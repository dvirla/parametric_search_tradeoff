"""
Merge a gpt-oss LoRA adapter into a BF16 base and save a plain bf16 HF checkpoint.

train_sft's --merge-at-end crashes for gpt-oss: merging into the OFFICIAL openai/gpt-oss-20b
(MXFP4) keeps the MXFP4 quantization_config on the model, so save_pretrained calls
revert_weight_conversion to re-encode MXFP4 and raises NotImplementedError. Fix: merge into a
genuine BF16 base (unsloth/gpt-oss-20b-BF16 = the same weights upcast from MXFP4, no quantizer)
so save_pretrained has nothing to revert and writes a clean bf16 checkpoint that
llama.cpp convert_hf_to_gguf turns into a GGUF (experts kept MXFP4).

Usage:
    uv run --no-sync python scripts/merge_gptoss_lora.py <adapter_dir> <out_dir> [base_model]
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out = sys.argv[1], sys.argv[2]
# Default to the BF16 base (no MXFP4 config) so the save doesn't try to revert to MXFP4.
base_id = sys.argv[3] if len(sys.argv) > 3 else "unsloth/gpt-oss-20b-BF16"
print(f"base model: {base_id}  (adapter trained vs "
      f"{json.load(open(os.path.join(adapter, 'adapter_config.json'))).get('base_model_name_or_path')})",
      flush=True)

print("loading base (bf16, cpu)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print("applying + merging adapter...", flush=True)
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()

# Drop the MXFP4 quantization metadata so save_pretrained writes plain bf16 (no revert).
cfg = model.config
if getattr(cfg, "quantization_config", None) is not None:
    try:
        delattr(cfg, "quantization_config")
    except Exception:
        cfg.quantization_config = None
    print("stripped quantization_config", flush=True)

os.makedirs(out, exist_ok=True)
print(f"saving merged bf16 -> {out} ...", flush=True)
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("MERGE_OK", flush=True)
