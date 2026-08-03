"""
Convert `<think>`-style SFT ChatML into gemma-4 canonical form.

Our rollout collector stored reasoning Qwen-style: `<think>...</think>` inside the assistant
`content`. gemma-4's canonical chat template instead wants reasoning in a **`reasoning`** field,
which it renders in the `<|channel>thought ... <channel|>` block **before** tool_calls (correct
causal order: reason -> act). If fed `<think>`-in-content, the template leaves the literal
`<think>` tags in `content`, rendered AFTER the tool_call — wrong order and wrong tokens.

gemma-4's `thinking_gate` keeps reasoning for every assistant turn that follows the last user
message (in our single-user agentic traces, that is every assistant turn). We move each turn's
`<think>` into `reasoning` and leave `content` as the post-`</think>` answer text.

Usage:
    uv run python scripts/gemmify_sft_chatml.py \
        --in  data/sft/frames_gemma4/procedure1_onpolicy_sft_rewired.jsonl \
        --out data/sft/frames_gemma4/procedure1_gemma_sft.jsonl
"""
import argparse
import json
import os
import re

_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
_THINK_STRIP = re.compile(r"<think>.*?</think>", re.S)


def split_think(content):
    if not content:
        return "", ""
    thinks = [t.strip() for t in _THINK_BLOCK.findall(content)]
    rest = _THINK_STRIP.sub("", content).strip()
    return "\n".join(t for t in thinks if t).strip(), rest


def to_gemma(messages):
    """Move <think> in assistant content -> `reasoning` field; content := answer text."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            think, answer = split_think(m.get("content"))
            m["content"] = answer
            if think:
                m["reasoning"] = think
            else:
                m.pop("reasoning", None)
        out.append(m)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = with_reasoning = empty_final = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = to_gemma(d["messages"])
            asst = [m for m in g if m.get("role") == "assistant"]
            if asst:
                if asst[-1].get("reasoning"):
                    with_reasoning += 1
                if not (asst[-1].get("content") or "").strip():
                    empty_final += 1
            fout.write(json.dumps({"messages": g}) + "\n")
            n += 1

    print(f"converted {n} examples -> {args.out}")
    print(f"  final turn with reasoning: {with_reasoning}")
    print(f"  final turn with EMPTY answer content: {empty_final}"
          + ("  <-- inspect" if empty_final else ""))


if __name__ == "__main__":
    main()
