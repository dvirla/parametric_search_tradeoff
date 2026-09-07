#!/usr/bin/env python3
"""Print the special-token metadata that decides whether a gemma-4 GGUF serves correctly.

The gemma-4 FRAMES-SFT GGUFs built with llama.cpp's convert_hf_to_gguf.py are mis-packaged in two
ways that only show up on the path where the model answers WITHOUT calling a tool (89.2% of
zero-search rows leak a raw `<channel|>` and truncate, vs 0.1% of rows with a search):

  * the six channel/tool control tokens are typed USER_DEFINED instead of CONTROL, because the
    converter classifies specials from tokenizer_config.json's `added_tokens_decoder`, which
    gemma-4 ships EMPTY -- its specials are flagged in tokenizer.json's `added_tokens` instead
    (llama.cpp issue #5838);
  * `tokenizer.ggml.eos_token_ids` is absent entirely, because gguf-py defines only the singular
    EOS_ID key -- so gemma-4's [1, 106, 50] cannot be represented.

Ollama's own converter (`ollama create --quantize`) does not have either gap. This script is the
acceptance test for a rebuild: point it at the old and new artifacts and diff the three rows.

    uv run python scripts/verify_gguf_special_tokens.py <file-or-tag> [<file-or-tag> ...]

A bare name is resolved as an ollama tag via $OLLAMA_MODELS (default ~/.ollama/models).
"""
from __future__ import annotations
import json, os, struct, sys

NAMES = {0: "UNDEFINED", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}
WATCH = [1, 2, 48, 49, 50, 51, 100, 101, 105, 106]
T_STR, T_ARR = 8, 9
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _s(f):
    return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")


def parse(path):
    """-> (kv dict; large arrays as ('LAZY', n, offset, elem_type), open file handle)."""
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        f.close(); return None, None
    struct.unpack("<I", f.read(4))[0]; struct.unpack("<Q", f.read(8))[0]
    nkv = struct.unpack("<Q", f.read(8))[0]
    kv = {}
    for _ in range(nkv):
        k = _s(f); t = struct.unpack("<I", f.read(4))[0]
        if t == T_STR:
            kv[k] = _s(f)
        elif t == T_ARR:
            et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
            off = f.tell()
            if et == T_STR:
                if n > 4096:
                    for _ in range(n):
                        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
                    kv[k] = ("LAZY", n, off, et)
                else:
                    kv[k] = [_s(f) for _ in range(n)]
            else:
                if n > 4096:
                    f.seek(n * SCALAR[et], 1); kv[k] = ("LAZY", n, off, et)
                else:
                    kv[k] = [struct.unpack(FMT[et], f.read(SCALAR[et]))[0] for _ in range(n)]
        else:
            kv[k] = struct.unpack(FMT[t], f.read(SCALAR[t]))[0]
    return kv, f


def strings_at(f, lazy, ids):
    _, n, off, _ = lazy; want = set(ids); out = {}
    f.seek(off)
    for i in range(n):
        ln = struct.unpack("<Q", f.read(8))[0]
        if i in want:
            out[i] = f.read(ln).decode("utf-8", "replace"); want.discard(i)
            if not want: break
        else:
            f.seek(ln, 1)
    return out


def nums_at(f, lazy, ids):
    _, n, off, et = lazy; out = {}
    for i in ids:
        f.seek(off + i * SCALAR[et]); out[i] = struct.unpack(FMT[et], f.read(SCALAR[et]))[0]
    return out


def resolve(name):
    if os.path.exists(name):
        return name
    store = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
    tag = name if ":" in name else name + ":latest"
    repo, t = tag.rsplit(":", 1)
    if "/" not in repo:
        repo = "library/" + repo
    p = f"{store}/manifests/registry.ollama.ai/{repo}/{t}"
    if not os.path.exists(p):
        return None
    man = json.load(open(p))
    for l in man.get("layers", []):
        if l["mediaType"].endswith(".model"):
            return f"{store}/blobs/" + l["digest"].replace(":", "-")
    return None


def main():
    for name in sys.argv[1:]:
        path = resolve(name)
        print("=" * 96)
        print(f"{name}\n  -> {path}")
        if not path:
            print("  UNRESOLVED (not a file, and no such ollama tag)"); continue
        kv, f = parse(path)
        if kv is None:
            print("  not a GGUF"); continue
        print(f"  tokenizer.ggml.model = {kv.get('tokenizer.ggml.model')!r}   pre = {kv.get('tokenizer.ggml.pre')!r}")
        print(f"  eos_token_id  = {kv.get('tokenizer.ggml.eos_token_id')}")
        eogs = kv.get("tokenizer.ggml.eos_token_ids")
        print(f"  eos_token_ids = {eogs if eogs is not None else 'ABSENT  <-- gemma-4 needs [1, 106, 50]'}")
        toks = kv.get("tokenizer.ggml.tokens"); tt = kv.get("tokenizer.ggml.token_type")
        if isinstance(toks, tuple) and isinstance(tt, tuple):
            s = strings_at(f, toks, WATCH); v = nums_at(f, tt, WATCH)
            bad = [i for i in WATCH if v[i] != 3]
            print(f"  {'id':>5} {'token':<22} type")
            for i in WATCH:
                flag = "" if v[i] == 3 else "   <-- should be CONTROL"
                print(f"  {i:>5} {s.get(i,''):<22} {NAMES.get(v[i], v[i])}{flag}")
            print(f"  VERDICT: {'OK -- all watched tokens are CONTROL' if not bad else f'{len(bad)} mis-typed: {bad}'}"
                  f"{' and eos_token_ids present' if eogs else ''}")
        f.close()


if __name__ == "__main__":
    main()
