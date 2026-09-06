#!/usr/bin/env python3
"""Why do the gemma-4 FRAMES-SFT checkpoints emit a raw `<channel|>` and truncate?

The SFT GGUFs are missing `tokenizer.ggml.eos_token_ids` (base gemma4:31b has [1, 106, 50] =
<eos> / <turn|> / <|tool_response|>) AND they embed an 18.7k-char channel-based
`tokenizer.chat_template` that the base GGUF does not have. Under `no_search` (no tools registered)
74-97% of responses end on a raw `<channel|>` with no answer; with tools they are ~0% except under
the confident_parametric cue, which tells the model not to search.

HYPOTHESIS: the embedded Jinja template makes ollama take the generic Jinja path instead of its
built-in gemma4 channel renderer, so channel markers land in `content` instead of being routed.

This probes variants of the SAME weights (each `FROM` the existing tag, so no blob is duplicated
and no disk is consumed) and reports the contamination rate of each, no-tools, on real questions:

  current   the registration as it stands  (PARAMETER stop "<turn|>")
  nostop    same, stop parameter removed             -> is the stop itself truncating?
  eogstop   stops on all three of the base's EOG strings
  native    RENDERER/PARSER copied from base gemma4:31b, if it declares them

Run inside the Athena job scripts/athena_diagnose_gemma_channel.job (needs ollama 0.32.5 + a GPU).
"""
from __future__ import annotations
import json, os, re, subprocess, sys, urllib.request

CHAN = re.compile(r"<\|?channel\|?>")
BASE_TAG = "gemma4:31b"
SFT_TAG = "gemma4-frames-robust-q4km:latest"
HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
N = int(os.environ.get("N_PROBE", "20"))


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def modelfile(tag):
    return sh("ollama", "show", "--modelfile", tag)


def chat(model, prompt):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://{HOST}/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d.get("message", {})
    # a correctly-parsed channel model puts reasoning in `thinking`, not in `content`
    return m.get("content") or "", m.get("thinking") or "", d.get("done_reason")


def questions(n):
    p = "data/hotpotqa_300.jsonl"
    out = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out.append(r.get("problem") or r.get("question") or r.get("text"))
            if len(out) >= n:
                break
    return out


def probe(tag, qs):
    hit = ends = empty = think = 0
    ex = None
    for q in qs:
        try:
            c, t, dr = chat(tag, q)
        except Exception as e:
            print(f"      [{tag}] request failed: {type(e).__name__}: {e}")
            continue
        if CHAN.search(c):
            hit += 1
            if CHAN.search(c.rstrip()[-40:]):
                ends += 1
            if ex is None:
                ex = c.rstrip()[-120:]
        if not c.strip():
            empty += 1
        if t.strip():
            think += 1
    n = len(qs)
    print(f"  {tag:<46} channel={hit:>3}/{n}  ends_on_it={ends:>3}  empty={empty:>3}  has_thinking={think:>3}")
    if ex:
        print(f"      e.g. ...{ex!r}")
    return hit, ends


def create(name, body):
    path = f"/tmp/Modelfile.{name}"
    open(path, "w").write(body)
    r = subprocess.run(["ollama", "create", name, "-f", path], capture_output=True, text=True)
    if r.returncode:
        print(f"  [create {name} FAILED] {r.stderr.strip()[:300]}")
        return False
    return True


def main():
    print("=" * 96)
    print(f"BASE {BASE_TAG} modelfile\n{'-'*96}\n{modelfile(BASE_TAG)}")
    print("=" * 96)
    print(f"SFT {SFT_TAG} modelfile\n{'-'*96}\n{modelfile(SFT_TAG)}")
    print("=" * 96)

    base_mf = modelfile(BASE_TAG)
    carry = "\n".join(l for l in base_mf.splitlines()
                      if l.strip().upper().startswith(("RENDERER", "PARSER", "TEMPLATE")))
    print(f"renderer/parser/template lines the BASE declares:\n{carry or '  (none)'}\n")

    variants = {
        "sftdiag-current": f'FROM {SFT_TAG}\n',
        "sftdiag-nostop":  f'FROM {SFT_TAG}\nPARAMETER stop ""\n',
        "sftdiag-eogstop": f'FROM {SFT_TAG}\nPARAMETER stop "<turn|>"\nPARAMETER stop "<|tool_response>"\nPARAMETER stop "<eos>"\n',
    }
    if carry.strip():
        variants["sftdiag-native"] = f'FROM {SFT_TAG}\n{carry}\n'

    qs = questions(N)
    print(f"probing {len(qs)} questions, NO tools registered\n")
    print(f"--- reference ---")
    probe(BASE_TAG, qs)
    probe(SFT_TAG, qs)
    print(f"--- variants (same weights, no blob duplicated) ---")
    for name, body in variants.items():
        if create(name, body):
            probe(name, qs)
    print("\ncleanup:", " ".join(f"ollama rm {v}" for v in variants))


if __name__ == "__main__":
    main()
