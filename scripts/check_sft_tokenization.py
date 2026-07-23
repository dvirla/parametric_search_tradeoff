"""
Dry-run tokenization check for a ChatML SFT file against a given HF model.

Validates the riskiest part of Step 0 before committing to a training run:
that the model's chat template accepts our tool-call ChatML AND that
assistant-only loss masking works. gpt-oss uses a channel/"harmony" template,
so `return_assistant_tokens_mask` may or may not yield usable masks — this
tells you which, and whether the prefix-retokenization fallback is needed.

Mirrors train_sft.py: clears system prompts to empty, normalizes
tool_calls[].function.arguments from JSON string back to dict for Jinja.

Usage:
    uv run python scripts/check_sft_tokenization.py \
        --data data/sft/frames/procedure1_onpolicy_sft_rewired.jsonl \
        --model openai/gpt-oss-20b --n 3
"""

import argparse
import json
import sys


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Clear system content (as train_sft does) and parse tool-call arguments to dict."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "system":
            m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        pass
                tc["function"] = fn
                new_tcs.append(tc)
            m["tool_calls"] = new_tcs
        out.append(m)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ChatML SFT jsonl ({'messages': [...]}).")
    p.add_argument("--model", required=True, help="HF model id or local path (for the tokenizer).")
    p.add_argument("--n", type=int, default=3, help="How many lines to test.")
    p.add_argument("--show", action="store_true", help="Print the rendered prompt for line 1.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    has_template = bool(getattr(tok, "chat_template", None))
    print(f"model: {args.model}")
    print(f"chat_template present: {has_template}")
    if not has_template:
        sys.exit("ERROR: tokenizer has no chat_template — cannot SFT this model as-is.")

    lines = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
            if len(lines) >= args.n:
                break
    if not lines:
        sys.exit("ERROR: no lines in data file.")

    n_render_ok = 0
    n_mask_ok = 0
    mask_supported = None
    for i, obj in enumerate(lines):
        msgs = normalize_messages(obj["messages"])
        # 1) Does the template render at all (tool calls included)?
        try:
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            n_render_ok += 1
        except Exception as e:
            print(f"  line {i}: RENDER FAILED — {type(e).__name__}: {e}")
            continue
        if args.show and i == 0:
            print("\n----- rendered line 0 -----")
            print(rendered[:2000])
            print("----- end -----\n")
        # 2) Does assistant-token masking produce a usable (non-trivial) mask?
        try:
            enc = tok.apply_chat_template(
                msgs, tokenize=True, return_assistant_tokens_mask=True,
                return_dict=True, add_generation_prompt=False,
            )
            mask = enc.get("assistant_masks")
            if mask is not None and sum(mask) > 0 and sum(mask) < len(mask):
                n_mask_ok += 1
                mask_supported = True
            else:
                mask_supported = mask_supported or False
                print(f"  line {i}: assistant mask degenerate "
                      f"(sum={None if mask is None else sum(mask)}, len={None if mask is None else len(mask)}) "
                      f"— template likely lacks {{% generation %}} markers; prefix-retokenization fallback needed.")
        except Exception as e:
            mask_supported = mask_supported or False
            print(f"  line {i}: MASK path errored — {type(e).__name__}: {e} "
                  f"— prefix-retokenization fallback needed.")

    print(f"\nrendered OK: {n_render_ok}/{len(lines)}")
    print(f"assistant-mask OK (native): {n_mask_ok}/{len(lines)}")
    if n_render_ok == len(lines) and n_mask_ok == len(lines):
        print("PASS: template renders and native assistant masking works — train_sft can use the fast path.")
    elif n_render_ok == len(lines):
        print("PARTIAL: template renders, but native masking is unavailable — "
              "train_sft's prefix-retokenization fallback must handle masking (verify it triggers).")
    else:
        print("FAIL: template does not render our tool-call ChatML — fix formatting before training.")


if __name__ == "__main__":
    main()
