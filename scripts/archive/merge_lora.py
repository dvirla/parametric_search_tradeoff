"""
Merge a saved LoRA adapter into its base model and write the result as a
standard HuggingFace checkpoint (Safetensors).

Usage (no accelerate needed — runs on a single process):

  PYTHONUNBUFFERED=1 HF_HOME=/work/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
  uv run python scripts/merge_lora.py \
      --adapter-dir /work/models/nemotron-musique-lora-v2 \
      --output-dir  /work/models/nemotron-musique-lora-v2_merged \
      --hf-cache-dir /work/hf_cache

The base model name is read from the adapter's adapter_config.json
(field "base_model_name_or_path") so you don't need to pass it explicitly.
Override with --base-model if the path recorded in the config is stale.

Memory note: the 30B Nemotron model weighs ~60 GB in bf16. merge_and_unload()
requires the full model in one contiguous address space, so the base model is
loaded on CPU first (avoids VRAM fragmentation during the merge). The script
then moves the merged result to CPU before saving. This needs ~120 GB RAM
(base + adapter copy during merge). If your machine does not have that, pass
--device cuda to load on GPU instead (requires ~60 GB VRAM split across GPUs).
"""

import argparse
import json
import os
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def save_model_safetensors(model, output_dir: str, max_shard_size_gb: int = 5) -> None:
    """Save model weights as safetensors shards, bypassing transformers' tied-weights
    check which crashes on Nemotron (returns a list where a dict is expected)."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    max_bytes = max_shard_size_gb * 1024 ** 3

    # Build shards greedily.
    shards: list[dict] = []
    current: dict = {}
    current_bytes = 0
    for key, tensor in state_dict.items():
        size = tensor.nbytes
        if current and current_bytes + size > max_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    n = len(shards)
    if n == 1:
        path = os.path.join(output_dir, "model.safetensors")
        save_file(shards[0], path)
        log(f"      Saved 1 shard → model.safetensors")
    else:
        index: dict = {
            "metadata": {"total_size": sum(t.nbytes for t in state_dict.values())},
            "weight_map": {},
        }
        for i, shard in enumerate(shards):
            fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            save_file(shard, os.path.join(output_dir, fname))
            for key in shard:
                index["weight_map"][key] = fname
            log(f"      Saved shard {i + 1}/{n} → {fname}")
        idx_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

    # Config + generation config (no state dict involved — safe to call directly).
    model.config.save_pretrained(output_dir)
    try:
        model.generation_config.save_pretrained(output_dir)
    except Exception:
        pass


def load_base_model_name(adapter_dir: str) -> str:
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    name = cfg.get("base_model_name_or_path", "")
    if not name:
        raise ValueError("base_model_name_or_path is empty in adapter_config.json")
    return name


def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--adapter-dir", required=True,
                   help="Directory produced by train_sft.py (contains adapter_config.json)")
    p.add_argument("--output-dir", required=True,
                   help="Where to write the merged checkpoint (Safetensors format)")
    p.add_argument("--base-model", default=None,
                   help="Override base model name / path (default: read from adapter_config.json)")
    p.add_argument("--hf-cache-dir", default=None,
                   help="Override HF_HOME, e.g. /work/hf_cache")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Dtype for loading the base model (default: bf16)")
    p.add_argument("--device", default="cpu",
                   help="Device for loading the base model: 'cpu' (default, safe for merge) "
                        "or 'auto' (spread across GPUs, needs ~60 GB VRAM total)")
    args = p.parse_args()

    # Force unbuffered output regardless of how the process was launched.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        log(f"HF cache   : {args.hf_cache_dir}")

    base_model_name = args.base_model or load_base_model_name(args.adapter_dir)
    log(f"Base model : {base_model_name}")
    log(f"Adapter    : {args.adapter_dir}")
    log(f"Output     : {args.output_dir}")
    log(f"Device     : {args.device}")
    log(f"Dtype      : {args.dtype}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    device_map = args.device if args.device != "cpu" else None

    log(f"\n[1/4] Loading base model ({args.dtype}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    log("      Base model loaded.")

    log("[2/4] Loading LoRA adapter ...")
    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    log("      Adapter loaded.")

    log("[3/4] Merging LoRA weights into base model ...")
    merged = peft_model.merge_and_unload()
    log("      Merge complete.")

    os.makedirs(args.output_dir, exist_ok=True)
    log(f"[4/4] Saving merged checkpoint to {args.output_dir} ...")
    save_model_safetensors(merged, args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    log("      Saved.")

    log(f"\nDone. Merged checkpoint at: {args.output_dir}")
    log("\nRegister with Ollama:")
    log(f"  bash scripts/register_nemotron_finetuned.sh \\")
    log(f"      --model-dir {args.output_dir} \\")
    log(f"      --model-name nemotron-3-nano-musique:30b")


if __name__ == "__main__":
    main()
