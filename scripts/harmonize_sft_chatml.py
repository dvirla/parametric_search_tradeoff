"""
Convert `<think>`-style SFT ChatML into gpt-oss (OpenAI harmony) format.

Our rollout collector stored reasoning Qwen-style: `<think>...</think>` inside the
assistant `content`. gpt-oss's chat template instead wants reasoning in a separate
`thinking` field, which it renders in the `analysis` channel; `content` becomes the
`final` channel. If fed the `<think>`-in-content form, the gpt-oss template DROPS the
reasoning of tool-call turns entirely and renders final-turn reasoning as literal
`<think>` tags in the answer — both wrong for training.

Harmony also keeps the `analysis` channel of only the LAST assistant turn (intermediate
reasoning is ephemeral by design). We therefore **strip `thinking` from all but the final
assistant turn**. This is not a loss (the template drops it anyway) and it is REQUIRED for
correct loss masking: train_sft.py falls back to prefix-retokenization (gpt-oss has no
`{% generation %}` markers), which needs `render(messages[:i+1])` to be a token-prefix of
the full render. If an intermediate turn kept `thinking`, it would render WITH an analysis
channel as the last message of a prefix but WITHOUT one in the full sequence, breaking the
prefix alignment and corrupting the mask.

What is trained after conversion (verified): the search tool-call commentary, the final
analysis, and the final answer. Tool results / user / system stay masked.

Usage:
    uv run python scripts/harmonize_sft_chatml.py \
        --in data/sft/frames/procedure1_onpolicy_sft_rewired.jsonl \
        --out data/sft/frames_gptoss/procedure1_onpolicy_sft_rewired.jsonl
"""

import argparse
import json
import os
import re

_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
_THINK_STRIP = re.compile(r"<think>.*?</think>", re.S)


def split_think(content):
    """Return (joined_thinking, answer_text) from a `<think>`-annotated content string."""
    if not content:
        return "", ""
    thinks = [t.strip() for t in _THINK_BLOCK.findall(content)]
    rest = _THINK_STRIP.sub("", content).strip()
    return "\n".join(t for t in thinks if t).strip(), rest


def to_harmony(messages):
    """Reformat one ChatML message list into gpt-oss harmony form."""
    asst_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    last = asst_idx[-1] if asst_idx else -1
    out = []
    for i, m in enumerate(messages):
        m = dict(m)
        if m.get("role") == "assistant":
            think, answer = split_think(m.get("content"))
            m["content"] = answer  # final channel (empty for tool-call turns)
            if i == last and think:
                m["thinking"] = think  # analysis channel, kept only on the final turn
            else:
                m.pop("thinking", None)  # intermediate reasoning: dropped by harmony anyway
        out.append(m)
    return out


def _length_filter(tokenizer_name, max_tokens):
    """Return a predicate keep(messages)->bool that drops examples over max_tokens.

    Over-long examples must be DROPPED, not truncated: train_sft truncates the tail, which
    is the final answer we most want to train on. Tokenizer import is lazy so the plain
    conversion path needs no transformers.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def keep(messages):
        m = [{**x, "content": ""} if x.get("role") == "system" else x for x in messages]
        s = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        return len(tok(s, add_special_tokens=False)["input_ids"]) <= max_tokens
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, help="input <think>-style ChatML jsonl")
    p.add_argument("--out", dest="out", required=True, help="output harmony ChatML jsonl")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="drop examples whose rendered length exceeds this (needs --tokenizer). "
                        "Dropping beats truncating: truncation would cut the final answer.")
    p.add_argument("--tokenizer", default="openai/gpt-oss-20b",
                   help="HF tokenizer used for --max-tokens filtering")
    args = p.parse_args()

    keep = _length_filter(args.tokenizer, args.max_tokens) if args.max_tokens else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = kept_think = empty_final = dropped_long = 0
    by_cond = {}
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            h = to_harmony(d["messages"])
            if keep is not None and not keep(h):
                dropped_long += 1
                continue
            fa = [m for m in h if m.get("role") == "assistant"]
            if fa:
                if fa[-1].get("thinking"):
                    kept_think += 1
                if not (fa[-1].get("content") or "").strip():
                    empty_final += 1
            fout.write(json.dumps({"messages": h}) + "\n")
            n += 1

    print(f"converted {n} examples -> {args.out}")
    if keep is not None:
        print(f"  dropped (over {args.max_tokens} tokens): {dropped_long}")
    print(f"  final turn with analysis (thinking): {kept_think}")
    print(f"  final turn with EMPTY answer content: {empty_final}"
          + ("  <-- inspect; final answer should be non-empty" if empty_final else ""))


if __name__ == "__main__":
    main()
