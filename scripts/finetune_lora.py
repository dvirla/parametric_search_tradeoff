"""LoRA fine-tuning of GLM-4.7-Flash to address performative ignorance.

Uses HuggingFace transformers + PEFT + TRL SFTTrainer, with TensorBoard tracking.
The model is loaded in full bf16 (no quantization) since we have ~160GB GPU.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def inspect_model_modules(model) -> list[str]:
    """Print and return linear module names for LoRA target selection."""
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the last part of the module name
            parts = name.split(".")
            linear_modules.add(parts[-1])

    print("\nDiscovered linear module types:")
    for name in sorted(linear_modules):
        print(f"  {name}")
    return sorted(linear_modules)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for GLM-4.7-Flash")
    parser.add_argument(
        "--model-name",
        type=str,
        default="zai-org/GLM-4.7-Flash",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/sft/glm_4_7_popqa_sft.jsonl",
        help="Path to SFT JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/glm-4.7-flash-lora",
        help="Output directory for LoRA adapter",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16, help="LoRA rank (r)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="all-linear",
        help="Comma-separated target modules, or 'all-linear'",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per-device train batch size"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only load model and print module names, then exit",
    )
    args = parser.parse_args()

    # --- Load tokenizer ---
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model in bf16 ---
    print(f"Loading model {args.model_name} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Inspect modules ---
    module_names = inspect_model_modules(model)

    if args.inspect_only:
        print("\n--inspect-only flag set. Printing full named modules with 'proj' or 'dense':")
        for name, _ in model.named_modules():
            if any(kw in name for kw in ("proj", "dense", "linear", "qkv", "query", "key", "value")):
                print(f"  {name}")
        return

    # --- Parse target modules ---
    if args.target_modules == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",")]

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"\nLoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"targets={target_modules}")

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load dataset ---
    print(f"\nLoading dataset from {args.dataset_path}...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Nothing to train on.")
        return

    # --- Training config ---
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_dir=os.path.join(args.output_dir, "tensorboard"),
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        report_to="tensorboard",
        dataset_text_field=None,  # We use the chat template formatting
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    # Save the adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training info
    info = {
        "base_model": args.model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules if isinstance(target_modules, list) else "all-linear",
        "training_loss": train_result.training_loss,
        "dataset_size": len(dataset),
        "epochs": args.epochs,
        "metrics": train_result.metrics,
    }
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nTraining complete. Adapter saved to {args.output_dir}")
    print(f"TensorBoard logs: {os.path.join(args.output_dir, 'tensorboard')}")
    print(f"View with: tensorboard --logdir {os.path.join(args.output_dir, 'tensorboard')}")


if __name__ == "__main__":
    main()
