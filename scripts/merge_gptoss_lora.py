"""
Merge a gpt-oss LoRA adapter into the base and save a plain bf16 HF checkpoint.

train_sft's --merge-at-end crashes for gpt-oss: the base loads dequantized to bf16 (no MXFP4
Triton kernels), then save_pretrained calls revert_weight_conversion to re-encode MXFP4 and
raises NotImplementedError. Fix: drop the MXFP4 quantization_config before saving so it writes
a normal bf16 checkpoint (which llama.cpp convert_hf_to_gguf can then turn into a GGUF).

Usage:
    uv run --no-sync python scripts/merge_gptoss_lora.py <adapter_dir> <out_dir>
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter, out = sys.argv[1], sys.argv[2]
base_id = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
print(f"base model: {base_id}", flush=True)

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
