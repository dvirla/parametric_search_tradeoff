"""
Merge a gemma-4 LoRA adapter into its bf16 base and save a plain bf16 HF checkpoint.

train_sft's --merge-at-end already does this inline right after training, but if that step
failed for an unrelated reason (e.g. disk quota) after training itself succeeded, re-running
the whole multi-hour training just to redo the merge is wasteful -- the adapter is already
saved on disk. This script replicates train_sft.py's merge block standalone: load the base
model to CPU in bf16 (device_map="cpu", NOT "auto" -- gemma-4's tied embed_tokens/lm_head land
on the meta device under "auto" and crash on merge), apply + merge the adapter, save.

Usage:
    uv run --no-sync python scripts/merge_gemma_lora.py <adapter_dir> <out_dir> [base_model]
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out = sys.argv[1], sys.argv[2]
base_id = sys.argv[3] if len(sys.argv) > 3 else "google/gemma-4-31B-it"

print(f"base model: {base_id}", flush=True)
print("loading base (bf16, cpu)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("applying + merging adapter...", flush=True)
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()

os.makedirs(out, exist_ok=True)
print(f"saving merged bf16 -> {out} ...", flush=True)
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("MERGE_OK", flush=True)
