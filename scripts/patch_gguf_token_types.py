#!/usr/bin/env python3
"""Retype mis-converted gemma-4 control tokens in a GGUF, in place, 4 bytes at a time.

WHY. The gemma-4 FRAMES-SFT GGUFs were converted with six special tokens typed USER_DEFINED (4)
instead of CONTROL (3):

    48 <|tool_call>   49 <tool_call|>   50 <|tool_response>
    51 <tool_response|>   100 <|channel>   101 <channel|>

USER_DEFINED tokens detokenize as literal text, so ollama's gemma4 PARSER never sees a channel
marker as a control signal. Measured consequence (scripts/diagnose_gemma_sft_channel.py, 20 probes,
no tools): base gemma4:31b -> 0/20 leaked markers and 20/20 with `thinking` populated; the SFT ->
20/20 leaked, 18/20 truncated on the marker, 0/20 with `thinking`. No Modelfile PARAMETER or
RENDERER/PARSER line fixes it (all were tried) because the defect is tokenizer metadata inside the
file. `tokenizer.ggml.eos_token_ids` was dropped by the same conversion; that needs a header
rewrite and is NOT attempted here (the hand-added `stop "<turn|>"` covers token 106).

WHY IN PLACE. token_type is a fixed-width numeric array, so each entry is a known absolute offset
and a patch touches 6*4 = 24 bytes -- no re-quantizing 18.7 GB and no new blob. Ollama addresses
blobs by sha256 filename and does not re-hash on load, so patching the blob keeps the existing
registration working; the digest no longer describes the content, which is acceptable only because
these models are local-only and were never pulled from a registry.

ALWAYS writes a JSON backup of the original values next to the target before touching it, so the
change reverts with --restore.

    uv run python scripts/patch_gguf_token_types.py <gguf>            # inspect only
    uv run python scripts/patch_gguf_token_types.py <gguf> --apply
    uv run python scripts/patch_gguf_token_types.py <gguf> --restore
"""
from __future__ import annotations
import argparse, json, os, struct, sys

CONTROL, USER_DEFINED = 3, 4
TARGET_IDS = [48, 49, 50, 51, 100, 101]
NAMES = {0: "UNDEFINED", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE"}
T_STR, T_ARR, T_U32, T_I32 = 8, 9, 4, 5
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def find_token_type_array(path):
    """-> (absolute offset of element 0, element type, count). Streams the KV block only."""
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    struct.unpack("<I", f.read(4))[0]
    struct.unpack("<Q", f.read(8))[0]
    nkv = struct.unpack("<Q", f.read(8))[0]
    found = None
    for _ in range(nkv):
        key = _str(f)
        t = struct.unpack("<I", f.read(4))[0]
        if t == T_STR:
            _str(f)
        elif t == T_ARR:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            off = f.tell()
            if et == T_STR:
                for _ in range(n):
                    f.seek(struct.unpack("<Q", f.read(8))[0], 1)
            else:
                f.seek(n * SCALAR[et], 1)
            if key == "tokenizer.ggml.token_type":
                found = (off, et, n)
        else:
            f.seek(SCALAR[t], 1)
    f.close()
    if not found:
        raise SystemExit(f"{path}: no tokenizer.ggml.token_type array")
    return found


def read_vals(path, off, et, ids):
    fmt = "<i" if et == T_I32 else "<I"
    sz = SCALAR[et]
    out = {}
    with open(path, "rb") as f:
        for i in ids:
            f.seek(off + i * sz)
            out[i] = struct.unpack(fmt, f.read(sz))[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--ids", type=int, nargs="*", default=TARGET_IDS)
    a = ap.parse_args()
    bak = a.gguf + ".token_type_backup.json"

    off, et, n = find_token_type_array(a.gguf)
    sz = SCALAR[et]
    print(f"{a.gguf}\n  token_type: {n} entries, elem={sz}B, base offset={off}")
    cur = read_vals(a.gguf, off, et, a.ids)
    for i in a.ids:
        print(f"    id {i:>4}  offset {off + i*sz:<14} value {cur[i]} ({NAMES.get(cur[i], '?')})")

    if a.restore:
        if not os.path.exists(bak):
            raise SystemExit(f"no backup at {bak}")
        orig = {int(k): v for k, v in json.load(open(bak))["original"].items()}
        with open(a.gguf, "r+b") as f:
            for i, v in orig.items():
                f.seek(off + i * sz)
                f.write(struct.pack("<i" if et == T_I32 else "<I", v))
        print(f"  RESTORED {len(orig)} entries from {bak}")
        return

    if not a.apply:
        print("\n  (inspect only; pass --apply to write)")
        return

    todo = [i for i in a.ids if cur[i] != CONTROL]
    if not todo:
        print("\n  nothing to do: all target ids are already CONTROL")
        return
    json.dump({"gguf": a.gguf, "offset": off, "elem_size": sz,
               "original": {str(i): cur[i] for i in a.ids}},
              open(bak, "w"), indent=2)
    print(f"\n  backup -> {bak}")
    with open(a.gguf, "r+b") as f:
        for i in todo:
            f.seek(off + i * sz)
            f.write(struct.pack("<i" if et == T_I32 else "<I", CONTROL))
    after = read_vals(a.gguf, off, et, a.ids)
    print(f"  patched {len(todo)} entries ({4*len(todo)} bytes)")
    for i in a.ids:
        print(f"    id {i:>4}  {NAMES.get(cur[i],'?')} -> {NAMES.get(after[i],'?')}")
    bad = [i for i in a.ids if after[i] != CONTROL]
    if bad:
        raise SystemExit(f"VERIFY FAILED for {bad}")
    print("  verified: all target ids are CONTROL")


if __name__ == "__main__":
    main()
