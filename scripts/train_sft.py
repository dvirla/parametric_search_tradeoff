"""
SFT training for MusiQue search-augmented QA using two-procedure rejection-sampled data.

Loads procedure1_natural_sft.jsonl and procedure2_m_patching.jsonl from --data-dir,
combines them, tokenizes with per-example loss masking, and runs LoRA fine-tuning via
TRL SFTTrainer.

Procedure-2 examples carry a `patch_start_idx` field indicating the ChatML message
index where the patch begins. Tokens belonging to messages before that index are masked
from the loss (label = -100) so the model only trains on the patched + continuation
portion.

Tracking: TensorBoard + MLflow (both enabled by default).
Multi-GPU: DeepSpeed ZeRO-3 via --deepspeed configs/deepspeed_zero3.json.

NOTE: Do NOT use device_map="auto" with DeepSpeed ZeRO-3. Accelerate handles sharding.

Launch (single node, 8 GPUs):
  accelerate launch --config_file configs/accelerate_zero3.yaml scripts/train_sft.py \\
      --model-name Qwen/Qwen2.5-72B-Instruct \\
      --data-dir data/sft/musique \\
      --output-dir models/qwen-musique-lora \\
      --deepspeed configs/deepspeed_zero3.json \\
      --merge-at-end

Single GPU (small model, dry-run):
  python scripts/train_sft.py \\
      --model-name Qwen/Qwen2.5-7B-Instruct \\
      --data-dir data/sft/musique_val_test \\
      --output-dir /tmp/test_run \\
      --epochs 1 --save-steps 999
"""

import argparse
import json
import os

import mlflow
import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer


_ARM_FILES = [
    "procedure1_natural_sft.jsonl",
    "procedure2_m_patching.jsonl",
]


def load_sft_data(data_dir: str, seed: int):
    datasets = []
    for fname in _ARM_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        ds = load_dataset("json", data_files=path, split="train")
        print(f"  {fname}: {len(ds)} examples")
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No procedure JSONL files found in {data_dir}")

    combined = concatenate_datasets(datasets).shuffle(seed=seed)
    print(f"  Total: {len(combined)} examples")
    return combined


def build_tokenized_example(
    tokenizer,
    messages: list[dict],
    patch_start_idx: int | None,
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one ChatML example and return (input_ids, labels).

    Labels follow standard SFT masking (non-assistant turns = -100).
    For procedure-2 examples, assistant turns whose ChatML index is before
    patch_start_idx are additionally masked so the model does not train on
    the prefix that was never patched.
    """
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )[:max_seq_length]

    labels = [-100] * len(full_ids)

    # Token position up to which everything is masked (patch prefix).
    mask_before = 0
    if patch_start_idx:
        prefix_ids = tokenizer.apply_chat_template(
            messages[:patch_start_idx],
            tokenize=True,
            add_generation_prompt=False,
        )
        mask_before = len(prefix_ids)

    # Unmask assistant-turn tokens that fall at or after the patch boundary.
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        # Tokenize up to (not including) this turn with generation prompt to
        # get the exact position where the assistant content starts.
        start_ids = tokenizer.apply_chat_template(
            messages[:i],
            tokenize=True,
            add_generation_prompt=True,
        )
        end_ids = tokenizer.apply_chat_template(
            messages[:i + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        start = len(start_ids)
        end = len(end_ids)
        if end <= mask_before:
            continue  # entire assistant turn is in the masked prefix
        for j in range(max(start, mask_before), min(end, len(labels))):
            labels[j] = full_ids[j]

    return full_ids, labels


def preprocess_dataset(dataset, tokenizer, max_seq_length: int):
    """Tokenize all examples and bake per-example loss masks into labels.

    Returns a dataset with only `input_ids` and `labels` columns so TRL does
    not attempt a second round of tokenisation.
    """
    patch_col_exists = "patch_start_idx" in dataset.column_names

    def process_batch(batch):
        all_input_ids, all_labels = [], []
        patch_starts = (
            batch["patch_start_idx"] if patch_col_exists
            else [None] * len(batch["messages"])
        )
        for messages, psi in zip(batch["messages"], patch_starts):
            ids, lbls = build_tokenized_example(tokenizer, messages, psi, max_seq_length)
            all_input_ids.append(ids)
            all_labels.append(lbls)
        return {"input_ids": all_input_ids, "labels": all_labels}

    return dataset.map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def setup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA SFT training on MusiQue three-arm data")
    p.add_argument("--model-name", required=True,
                   help="HuggingFace model ID or local checkpoint path")
    p.add_argument("--data-dir", default="data/sft/musique",
                   help="Directory containing arm*.jsonl files")
    p.add_argument("--output-dir", default="models/qwen-musique-lora",
                   help="Output directory for LoRA adapter")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear",
                   help="Comma-separated module names or 'all-linear'")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device train batch size (use 1 for 122B)")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--deepspeed", default=None,
                   help="Path to DeepSpeed JSON config (required for ZeRO-3 multi-GPU)")
    p.add_argument("--mlflow-uri", default="./mlruns",
                   help="MLflow tracking URI")
    p.add_argument("--mlflow-experiment", default="sft_musique",
                   help="MLflow experiment name")
    p.add_argument("--merge-at-end", action="store_true",
                   help="Merge LoRA weights into base model after training")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    print(f"Loading SFT data from {args.data_dir}...")
    dataset = load_sft_data(args.data_dir, args.seed)
    # Tokenizer must be loaded before preprocessing so we can bake masks in.
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizing dataset and baking loss masks...")
    dataset = preprocess_dataset(dataset, tokenizer, args.max_seq_length)

    # --- Model ---
    # IMPORTANT: no device_map with DeepSpeed ZeRO-3; accelerate manages placement.
    use_deepspeed = args.deepspeed is not None
    print(f"Loading model {args.model_name} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        **({"device_map": "auto"} if not use_deepspeed else {}),
    )

    # --- LoRA ---
    target_modules = (
        "all-linear" if args.target_modules == "all-linear"
        else [m.strip() for m in args.target_modules.split(",")]
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, targets={target_modules}")

    # --- Training config ---
    # TensorBoard logging dir via env var (warmup_ratio deprecated in TRL v5.2+)
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(args.output_dir, "tensorboard")
    total_steps = (len(dataset) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard", "mlflow"],
        deepspeed=args.deepspeed,
        dataset_text_field=None,
        seed=args.seed,
    )

    # Pre-tokenized dataset: use Seq2Seq collator so label padding is -100,
    # not the pad token id, which would incorrectly contribute to the loss.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # --- MLflow ---
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=os.path.basename(args.output_dir)):
        mlflow.log_params({
            "model_name": args.model_name,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": str(target_modules),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "dataset_size": len(dataset),
            "data_dir": args.data_dir,
            "deepspeed": args.deepspeed or "none",
        })

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        print(f"\nTrainable parameters:")
        trainer.model.print_trainable_parameters()

        print("\nStarting training...")
        train_result = trainer.train()

        mlflow.log_metric("train_loss", train_result.training_loss)

    # --- Save ---
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    info = {
        "base_model": args.model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules if isinstance(target_modules, list) else "all-linear",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "dataset_size": len(dataset),
        "training_loss": train_result.training_loss,
        "metrics": train_result.metrics,
    }
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # --- Optional merge ---
    # With ZeRO-3, this must run only on rank 0 after the barrier.
    # accelerate's is_main_process guard handles that correctly.
    if args.merge_at_end:
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            is_main = accelerator.is_main_process
        except Exception:
            is_main = True

        if is_main:
            print("\nMerging LoRA weights into base model...")
            merged = trainer.model.merge_and_unload()
            merged_path = args.output_dir + "_merged"
            merged.save_pretrained(merged_path, safe_serialization=True)
            tokenizer.save_pretrained(merged_path)
            print(f"Merged checkpoint saved to {merged_path}")
            print(f"\nServe with vLLM:\n"
                  f"  vllm serve {merged_path} --served-model-name <slug> "
                  f"--tensor-parallel-size <n_gpus>")

    print(f"\nTraining complete. Adapter saved to {args.output_dir}")
    print(f"TensorBoard: tensorboard --logdir {os.path.join(args.output_dir, 'tensorboard')}")
    print(f"MLflow UI:   mlflow ui --backend-store-uri {args.mlflow_uri}")


if __name__ == "__main__":
    main()
