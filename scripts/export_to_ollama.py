"""Merge LoRA adapter into base model, convert to GGUF, and register with Ollama.

Steps:
  1. Load base model + LoRA adapter
  2. Merge and unload → full model weights
  3. Save merged model to disk
  4. Convert to GGUF using llama.cpp's convert_hf_to_gguf.py
  5. Create Ollama Modelfile and register the model
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_model_name: str, adapter_dir: str, merged_dir: str):
    """Load base model + adapter, merge, and save."""
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Merge on CPU to avoid GPU memory issues
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("Merge complete.")


def convert_to_gguf(merged_dir: str, gguf_path: str, llama_cpp_dir: str, quantize: str = "bf16"):
    """Convert merged HF model to GGUF format using llama.cpp."""
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"Clone llama.cpp and provide --llama-cpp-dir."
        )

    # Output type mapping
    outtype_map = {
        "bf16": "bf16",
        "f16": "f16",
        "f32": "f32",
        "q8_0": "q8_0",
    }
    outtype = outtype_map.get(quantize, "bf16")

    cmd = [
        sys.executable,
        convert_script,
        merged_dir,
        "--outfile", gguf_path,
        "--outtype", outtype,
    ]

    print(f"Converting to GGUF ({outtype})...")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # If user wants further quantization (e.g., Q4_K_M), use llama-quantize
    if quantize.upper().startswith("Q") and quantize.upper() != "Q8_0":
        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_bin):
            quantize_bin = shutil.which("llama-quantize")

        if quantize_bin:
            quantized_path = gguf_path.replace(".gguf", f"-{quantize}.gguf")
            qcmd = [quantize_bin, gguf_path, quantized_path, quantize.upper()]
            print(f"Quantizing to {quantize.upper()}...")
            subprocess.run(qcmd, check=True)
            # Replace the bf16 file with quantized
            os.replace(quantized_path, gguf_path)
        else:
            print(f"WARNING: llama-quantize not found, skipping {quantize} quantization.")

    print(f"GGUF saved to {gguf_path}")


def register_ollama(gguf_path: str, model_tag: str, temperature: float = 0.6):
    """Create Ollama Modelfile and register the model."""
    gguf_abs = os.path.abspath(gguf_path)

    modelfile_content = f"""FROM {gguf_abs}
PARAMETER temperature {temperature}
PARAMETER top_p 0.95
"""

    modelfile_path = os.path.join(os.path.dirname(gguf_abs), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\nModelfile written to {modelfile_path}")
    print(f"Registering model as '{model_tag}' with Ollama...")

    cmd = ["ollama", "create", model_tag, "-f", modelfile_path]
    subprocess.run(cmd, check=True)

    print(f"Model '{model_tag}' registered successfully.")
    print(f"Test with: ollama run {model_tag} \"What genre is Fablehaven?\"")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter, convert to GGUF, and register with Ollama"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="zai-org/GLM-4.7-Flash",
        help="HuggingFace base model ID",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="models/glm-4.7-flash-lora",
        help="Directory containing the LoRA adapter",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="models/glm-4.7-flash-lora-merged",
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--gguf-path",
        type=str,
        default="models/glm-4.7-flash-lora.gguf",
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=os.path.expanduser("~/llama.cpp"),
        help="Path to llama.cpp repo (for convert_hf_to_gguf.py)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="bf16",
        choices=["bf16", "f16", "f32", "q8_0", "Q4_K_M", "Q5_K_M", "Q6_K"],
        help="Quantization type for GGUF output",
    )
    parser.add_argument(
        "--ollama-tag",
        type=str,
        default="glm-4.7-flash-lora:bf16",
        help="Ollama model tag to register",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step (use existing merged model)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip GGUF conversion (use existing GGUF)",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama registration",
    )
    args = parser.parse_args()

    # Step 1: Merge LoRA
    if not args.skip_merge:
        merge_lora(args.base_model, args.adapter_dir, args.merged_dir)
    else:
        print("Skipping merge (--skip-merge).")

    # Step 2: Convert to GGUF
    if not args.skip_convert:
        convert_to_gguf(args.merged_dir, args.gguf_path, args.llama_cpp_dir, args.quantize)
    else:
        print("Skipping GGUF conversion (--skip-convert).")

    # Step 3: Register with Ollama
    if not args.skip_ollama:
        register_ollama(args.gguf_path, args.ollama_tag)
    else:
        print("Skipping Ollama registration (--skip-ollama).")

    print("\nDone!")


if __name__ == "__main__":
    main()
