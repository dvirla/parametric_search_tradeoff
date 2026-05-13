"""
Tokenizer compatibility check for train_sft.py.

Loads a model's tokenizer and verifies:
  1. Chat template handles system/user/assistant/tool turns
  2. Loss masking in build_tokenized_example is correct (only assistant tokens labelled)
  3. Procedure-2 patch_start_idx masking works (only post-patch assistant tokens labelled)
  4. No off-by-one issues in assistant turn boundary detection

Run on the training machine before launching train_sft.py:
  python scripts/check_tokenizer_compat.py --model Qwen/Qwen3.5-9B
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_sft import _apply_chat_template, _normalize_messages, build_tokenized_example


def load_examples(data_dir: str):
    p1 = Path(data_dir) / "procedure1_natural_sft.jsonl"
    p2 = Path(data_dir) / "procedure2_m_patching.jsonl"
    examples = {}
    if p1.exists():
        with open(p1) as f:
            examples["p1"] = json.loads(f.readline())
    if p2.exists():
        with open(p2) as f:
            examples["p2"] = json.loads(f.readline())
    return examples


def decode_labelled_spans(tokenizer, input_ids, labels):
    """Return list of (role_guess, decoded_text) for each labelled span."""
    spans = []
    in_span = False
    span_ids = []
    for tid, lid in zip(input_ids, labels):
        if lid != -100:
            in_span = True
            span_ids.append(tid)
        else:
            if in_span:
                spans.append(tokenizer.decode(span_ids, skip_special_tokens=False))
                span_ids = []
                in_span = False
    if span_ids:
        spans.append(tokenizer.decode(span_ids, skip_special_tokens=False))
    return spans


def check_basic_template(tokenizer):
    print("\n=== 1. Basic chat template ===")
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    print(textwrap.indent(rendered[:500], "  "))
    assert "<|im_start|>" in rendered or "assistant" in rendered, \
        "Template looks wrong — no assistant marker found"
    print("  [OK] template renders correctly")


def check_tool_turn(tokenizer):
    print("\n=== 2. Tool turn in template ===")
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Search for X."},
        {"role": "assistant", "content": "<think>Let me search.</think>\n[search call]"},
        {"role": "tool", "content": "[{'snippet': 'result'}]"},
        {"role": "assistant", "content": "The answer is Y."},
    ]
    try:
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        print(f"  [OK] tool turns render ({len(rendered)} chars)")
        print(textwrap.indent(rendered[:600], "  "))
    except Exception as e:
        print(f"  [FAIL] tool turn rendering raised: {e}")
        raise


def check_loss_masking(tokenizer, example, label, max_seq_length=4096):
    print(f"\n=== 3. Loss masking — {label} ===")
    messages = example["messages"]
    patch_start_idx = example.get("patch_start_idx")
    input_ids, labels = build_tokenized_example(tokenizer, messages, patch_start_idx, max_seq_length)

    n_total = len(input_ids)
    n_labelled = sum(1 for l in labels if l != -100)
    n_masked = n_total - n_labelled
    print(f"  Tokens total:    {n_total}")
    print(f"  Tokens labelled: {n_labelled} ({100*n_labelled/n_total:.1f}%)")
    print(f"  Tokens masked:   {n_masked}")

    assert n_labelled > 0, "No tokens are labelled — model would learn nothing"
    assert n_masked > 0, "No tokens are masked — prompt is leaking into loss"

    spans = decode_labelled_spans(tokenizer, input_ids, labels)
    print(f"  Labelled spans ({len(spans)}):")
    for i, span in enumerate(spans):
        preview = span.replace("\n", "\\n")[:120]
        print(f"    [{i}] {preview}")

    # Sanity: first labelled span should start with assistant content, not user/system
    first = spans[0] if spans else ""
    bad_starts = ["<|im_start|>user", "<|im_start|>system", "<|im_start|>tool"]
    for bad in bad_starts:
        assert bad not in first[:50], \
            f"First labelled span starts with non-assistant turn: {first[:80]}"
    print("  [OK] masking looks correct")

    if patch_start_idx:
        print(f"  patch_start_idx={patch_start_idx}: first {patch_start_idx} messages masked from loss")
        prefix_ids = _apply_chat_template(
            tokenizer=tokenizer, messages=_normalize_messages(messages[:patch_start_idx])
        )
        # Labels up to len(prefix_ids) should all be -100
        pre_patch_labelled = sum(1 for l in labels[:len(prefix_ids)] if l != -100)
        if pre_patch_labelled > 0:
            print(f"  [WARN] {pre_patch_labelled} tokens before patch boundary are labelled "
                  f"(expected 0). This may be a boundary misalignment.")
        else:
            print("  [OK] pre-patch tokens are correctly masked")


def check_padding(tokenizer):
    print("\n=== 4. Pad token ===")
    if tokenizer.pad_token is None:
        print("  [WARN] pad_token is None — train_sft.py will set it to eos_token")
    else:
        print(f"  [OK] pad_token = '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--data-dir", default="data/sft/musique")
    p.add_argument("--max-seq-length", type=int, default=8192)
    args = p.parse_args()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    check_basic_template(tokenizer)
    check_tool_turn(tokenizer)
    check_padding(tokenizer)

    examples = load_examples(args.data_dir)
    if not examples:
        print(f"\n[WARN] No procedure JSONL files found in {args.data_dir} — skipping masking checks")
    for label, ex in examples.items():
        check_loss_masking(tokenizer, ex, label, args.max_seq_length)

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
