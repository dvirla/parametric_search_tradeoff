"""
SFT training for MusiQue search-augmented QA using on-policy rejection-sampled data.

Loads procedure1_onpolicy_sft.jsonl from --data-dir, tokenizes with standard SFT
assistant-turn masking, and runs LoRA fine-tuning via TRL SFTTrainer.

System prompts in the training data are replaced with empty content so the model
learns to search intrinsically rather than only when prompted by the specific
collection-time system instruction.

Tracking: TensorBoard + MLflow (both enabled by default).
Multi-GPU: DeepSpeed ZeRO-3 via --deepspeed configs/deepspeed_zero3.json.

NOTE: Do NOT use device_map="auto" with DeepSpeed ZeRO-3. Accelerate handles sharding.

Launch (single node, 8 GPUs):
  accelerate launch --config_file configs/accelerate_zero3.yaml scripts/train_sft.py \\
      --model-name Qwen/Qwen2.5-72B-Instruct \\
      --data-dir data/sft/musique_onpolicy \\
      --output-dir models/qwen-musique-lora \\
      --deepspeed configs/deepspeed_zero3.json \\
      --merge-at-end

Single GPU (small model, dry-run):
  python scripts/train_sft.py \\
      --model-name Qwen/Qwen2.5-7B-Instruct \\
      --data-dir data/sft/musique_onpolicy \\
      --output-dir /tmp/test_run \\
      --epochs 1 --save-steps 999
"""

import argparse
import json
import os
import random

import mlflow
import torch
from datasets import Features, Value, concatenate_datasets, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer


_ARM_FILES = [
    "procedure1_onpolicy_sft_rewired.jsonl",
]


# Unified schema for the `messages` column. Some files (no-search) carry only
# role+content per message; tool-use files also carry tool_calls/tool_call_id.
# Cast everything to the richer schema so heterogeneous JSONL files can be
# concatenated without alignment errors.
_MESSAGES_FEATURES = Features({
    "messages": [{
        "role": Value("string"),
        "content": Value("string"),
        "tool_calls": [{
            "id": Value("string"),
            "type": Value("string"),
            "function": {
                "name": Value("string"),
                "arguments": Value("string"),
            },
        }],
        "tool_call_id": Value("string"),
    }]
})


def _align_messages_schema(example):
    out = []
    for msg in example["messages"]:
        out.append({
            "role": msg.get("role") or "",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "tool_call_id": msg.get("tool_call_id") or "",
        })
    return {"messages": out}


def _load_paraphrase_map(path: str) -> dict[str, str]:
    """Build {natural_text -> original_formal_question} from musique_train_natural.jsonl."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--augment-formal-questions requires {path}; not found. "
            "Pass --paraphrase-map-path to override."
        )
    mapping: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            nat, orig = d.get("text"), d.get("original_question")
            if nat and orig:
                mapping[nat] = orig
    print(f"  Paraphrase map: {len(mapping)} natural→formal pairs from {os.path.basename(path)}")
    return mapping


def _swap_phrasing(example: dict, mapping: dict[str, str], rng: random.Random) -> dict:
    """50/50 per-example: keep natural OR rewrite all content with natural→formal text replace.

    Many assistant messages quote the user's natural-phrased question verbatim inside
    <think> blocks, so a user-only swap would leave inconsistent supervision. We do a
    plain str.replace on every message's content; tool messages and search results
    are unaffected (they don't contain the question).
    """
    msgs = example["messages"]
    user = next((m for m in msgs if m.get("role") == "user"), None)
    if not user:
        return example
    nat = user.get("content") or ""
    formal = mapping.get(nat)
    if formal is None or rng.random() < 0.5:
        return example  # keep natural (no match, or coin says so)
    rewritten = []
    for m in msgs:
        m = dict(m)
        if m.get("content"):
            m["content"] = m["content"].replace(nat, formal)
        rewritten.append(m)
    return {"messages": rewritten}


def _has_search(example: dict) -> bool:
    """An example is search-bearing if it contains a tool message or an assistant tool_call."""
    for m in example["messages"]:
        if m.get("role") == "tool":
            return True
        if m.get("tool_calls"):
            return True
    return False


def _apply_search_ratio(dataset, target_ratio: float, seed: int):
    """Downsample the larger of {search, no-search} so the final mix matches target_ratio.

    target_ratio is the desired fraction of search examples in the output. No upsampling —
    duplication is deferred to whatever calls this. Logs what was dropped.
    """
    if not (0.0 < target_ratio < 1.0):
        raise ValueError(f"--search-ratio must be in (0, 1), got {target_ratio}")

    search_idx, nosearch_idx = [], []
    for i, ex in enumerate(dataset):
        (search_idx if _has_search(ex) else nosearch_idx).append(i)

    n_s, n_ns = len(search_idx), len(nosearch_idx)
    if n_s == 0 or n_ns == 0:
        print(f"  [search-ratio] one group is empty (search={n_s}, no_search={n_ns}); skipping rebalance")
        return dataset

    # Solve: n_s' / (n_s' + n_ns') = target.  Downsample only — keep all of the
    # constraining group, shrink the other.
    target_search_frac = target_ratio
    # If we keep all search: required no_search count = n_s * (1-r)/r
    keep_all_search_needs_ns = int(round(n_s * (1 - target_search_frac) / target_search_frac))
    # If we keep all no-search: required search count = n_ns * r/(1-r)
    keep_all_ns_needs_s = int(round(n_ns * target_search_frac / (1 - target_search_frac)))

    rng = random.Random(seed)
    if keep_all_search_needs_ns <= n_ns:
        keep_s, keep_ns = n_s, keep_all_search_needs_ns
        rng.shuffle(nosearch_idx); nosearch_idx = nosearch_idx[:keep_ns]
    else:
        keep_s, keep_ns = keep_all_ns_needs_s, n_ns
        rng.shuffle(search_idx); search_idx = search_idx[:keep_s]

    chosen = sorted(search_idx + nosearch_idx)
    new_ds = dataset.select(chosen)
    actual_ratio = keep_s / max(1, keep_s + keep_ns)
    print(f"  [search-ratio] before: search={n_s} no_search={n_ns}  "
          f"after: search={keep_s} no_search={keep_ns} (ratio={actual_ratio:.3f}, target={target_ratio})")
    return new_ds


def load_sft_data(
    data_dir: str,
    seed: int,
    extra_data: list[str] | None = None,
    augment_formal_questions: bool = False,
    paraphrase_map_path: str = "data/musique_train_natural.jsonl",
    search_ratio: float | None = None,
):
    datasets = []
    for fname in _ARM_FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found")
            continue
        ds = load_dataset("json", data_files=path, split="train")
        print(f"  {fname}: {len(ds)} examples")
        datasets.append(ds)

    for path in (extra_data or []):
        if not os.path.exists(path):
            print(f"  [skip] extra-data {path} not found")
            continue
        ds = load_dataset("json", data_files=path, split="train")
        print(f"  {os.path.basename(path)}: {len(ds)} examples")
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No JSONL files found in {data_dir} (and no --extra-data provided)")

    aligned = [
        ds.map(_align_messages_schema, desc="Aligning message schema").cast(_MESSAGES_FEATURES)
        for ds in datasets
    ]
    combined = concatenate_datasets(aligned)

    if augment_formal_questions:
        mapping = _load_paraphrase_map(paraphrase_map_path)
        rng = random.Random(seed)
        # Stable, deterministic order so each example gets a reproducible coin flip.
        combined = combined.map(
            lambda ex: _swap_phrasing(ex, mapping, rng),
            desc="Swapping natural↔formal phrasing (50/50)",
        )

    if search_ratio is not None:
        combined = _apply_search_ratio(combined, search_ratio, seed)

    combined = combined.shuffle(seed=seed)
    print(f"  Total: {len(combined)} examples")
    return combined


def _apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Wrapper around apply_chat_template that normalises the return type.

    Newer transformers (>=4.48 from git) return List[List[int]] for a single
    conversation, while older versions return List[int]. Both are handled here.
    """
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # BatchEncoding / dict path
    if hasattr(result, "input_ids"):
        result = result.input_ids
    # Batched path: [[tok1, tok2, ...]]
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
        return list(result[0])
    return list(result)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize pydantic-ai message format for apply_chat_template compatibility.

    pydantic-ai stores tool_calls[].function.arguments as a JSON string.
    Qwen3.5's Jinja template calls `arguments | items` expecting a dict —
    parse it here so the template receives the right type.
    """
    out = []
    for msg in messages:
        if msg.get("tool_calls"):
            msg = dict(msg)
            fixed_calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"query": args}
                fixed_calls.append({**tc, "function": {**fn, "arguments": args}})
            msg["tool_calls"] = fixed_calls
        out.append(msg)
    return out


def build_tokenized_example(
    tokenizer,
    messages: list[dict],
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one ChatML example and return (input_ids, labels).

    System messages are cleared so the model learns to search intrinsically
    rather than only when prompted by the collection-time system instruction.
    Labels follow standard SFT masking: only assistant turns are unmasked.

    Fast path: single apply_chat_template call using return_assistant_tokens_mask
    (transformers >= 4.47). Falls back to re-tokenizing prefixes per assistant turn
    if the tokenizer doesn't support it.
    """
    messages = _normalize_messages(messages)
    # Erase system prompt content — keep the role so the template renders correctly.
    messages = [
        {**m, "content": ""} if m.get("role") == "system" else m
        for m in messages
    ]

    # Fast path: single call, O(N) instead of O(N * num_assistant_turns).
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        # Mask key name varies across transformers versions: `assistant_masks`
        # (current) vs `assistant_tokens_mask` (older). Try both.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            assistant_mask = (
                getattr(encoded, "assistant_masks", None)
                or getattr(encoded, "assistant_tokens_mask", None)
            )
        else:
            input_ids = encoded["input_ids"]
            assistant_mask = (
                encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
            )
        if assistant_mask is None:
            raise KeyError("no assistant mask key in apply_chat_template output")
        # Unwrap batched shape ([[...]]) returned by newer transformers.
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
            assistant_mask = assistant_mask[0]
        input_ids = list(input_ids)[:max_seq_length]
        assistant_mask = list(assistant_mask)[:max_seq_length]
        # An all-zero mask means the chat template lacks `{% generation %}` markers
        # (e.g. Nemotron-3-Nano). Fall through to the prefix-re-tokenization path;
        # otherwise every label would be -100 and the model would receive no signal.
        if not any(assistant_mask):
            raise ValueError("assistant mask all zeros — template lacks {% generation %}")
        labels = [id_ if m else -100 for id_, m in zip(input_ids, assistant_mask)]
        return input_ids, labels
    except (TypeError, KeyError, AttributeError, ValueError):
        pass

    # Fallback: re-tokenize prefixes to locate assistant turn boundaries.
    full_ids = _apply_chat_template(messages=messages, tokenizer=tokenizer)[:max_seq_length]
    labels = [-100] * len(full_ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        start_ids = _apply_chat_template(messages=messages[:i], tokenizer=tokenizer, add_generation_prompt=True)
        end_ids = _apply_chat_template(messages=messages[:i + 1], tokenizer=tokenizer)
        start = len(start_ids)
        end = len(end_ids)
        for j in range(start, min(end, len(labels))):
            labels[j] = full_ids[j]

    return full_ids, labels


def preprocess_dataset(dataset, tokenizer, max_seq_length: int, num_proc: int = 1):
    """Tokenize all examples and bake per-example loss masks into labels.

    Returns a dataset with only `input_ids` and `labels` columns so TRL does
    not attempt a second round of tokenisation.
    """
    def process_batch(batch):
        all_input_ids, all_labels = [], []
        for messages in batch["messages"]:
            ids, lbls = build_tokenized_example(tokenizer, messages, max_seq_length)
            all_input_ids.append(ids)
            all_labels.append(lbls)
        return {"input_ids": all_input_ids, "labels": all_labels}

    # Disable HuggingFace fast-tokenizer thread pool inside each worker to avoid
    # nested parallelism warnings when num_proc > 1.
    if num_proc > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return dataset.map(
        process_batch,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
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
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="Token cap per example. Default 8192 covers ~99.4%% of the "
                        "rewired onpolicy set with the Nemotron-3-Nano tokenizer "
                        "(4096 truncated ~16%% of examples, cutting the final "
                        "assistant turn).")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--deepspeed", default=None,
                   help="Path to DeepSpeed JSON config (required for ZeRO-3 multi-GPU)")
    p.add_argument("--mlflow-uri", default="./mlruns",
                   help="MLflow tracking URI")
    p.add_argument("--mlflow-experiment", default="sft_musique",
                   help="MLflow experiment name")
    p.add_argument("--merge-at-end", action="store_true",
                   help="Merge LoRA weights into base model after training")
    p.add_argument("--qlora", action="store_true",
                   help="Load base model in 4-bit NF4 (QLoRA). Disables DeepSpeed; "
                        "uses device_map=auto to spread model across GPUs.")
    p.add_argument("--full-finetune", action="store_true",
                   help="Train all model parameters (no LoRA adapter). Requires ZeRO-3 "
                        "with CPU offload for 30B+ models. Incompatible with --qlora "
                        "and --merge-at-end. Suggested LR: 5e-6 to 1e-5.")
    p.add_argument("--extra-data", nargs="*", default=None,
                   help="Additional JSONL file(s) to mix in (e.g. nosearch_sft.jsonl from a "
                        "different directory). Each file must have a 'messages' column.")
    p.add_argument("--augment-formal-questions", action="store_true",
                   help="Augmentation: for each example whose user message matches a natural "
                        "paraphrase, flip a coin (50/50) to substitute the original formal "
                        "MusiQue question (replaces natural-text everywhere it appears, including "
                        "inside assistant <think> quotes). Teaches phrasing-invariant behavior.")
    p.add_argument("--paraphrase-map-path", default="data/musique_train_natural.jsonl",
                   help="JSONL with 'text' (natural) and 'original_question' (formal) per row. "
                        "Used only when --augment-formal-questions is set.")
    p.add_argument("--search-ratio", type=float, default=None,
                   help="Target fraction of search-bearing examples in the final mix "
                        "(e.g. 0.5 = 50/50). Downsamples the larger of {search, no-search} "
                        "after augmentation. Unset = use whatever the loaded files contain.")
    p.add_argument("--optimizer", default="adamw_torch",
                   help="HF Trainer optim string. Use 'paged_adamw_8bit' (bitsandbytes) to "
                        "halve optimizer-state memory — needed for full SFT of 30B+ models "
                        "when CPU AdamW state otherwise exceeds the cgroup limit.")
    p.add_argument("--dataloader-num-workers", type=int, default=2,
                   help="Per-rank DataLoader fork count. Default 2 to cap CPU memory under "
                        "ZeRO-3 with optimizer offload (forks copy the dataset).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-cache-dir", default=None,
                   help="Override HF_HOME (model + dataset cache). Use group/work storage "
                        "when home quota is tight, e.g. /home/dvirla/work/hf_cache")
    p.add_argument("--tokenize-workers", type=int, default=min(16, os.cpu_count() or 4),
                   help="Parallel processes for dataset tokenization. Defaults to min(16, cpu_count).")
    args = p.parse_args()
    if args.full_finetune and args.qlora:
        p.error("--full-finetune and --qlora are mutually exclusive (QLoRA is 4-bit + LoRA).")
    if args.full_finetune and args.merge_at_end:
        p.error("--full-finetune and --merge-at-end are mutually exclusive (nothing to merge).")
    return args


def main():
    args = setup_args()

    # Redirect HuggingFace cache before any HF library calls touch disk.
    # Qwen2.5-72B-Instruct is ~144 GB in bf16 — must land outside the home quota.
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.makedirs(args.hf_cache_dir, exist_ok=True)
        print(f"HF cache redirected to {args.hf_cache_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    print(f"Loading SFT data from {args.data_dir}...")
    dataset = load_sft_data(
        args.data_dir,
        args.seed,
        extra_data=args.extra_data,
        augment_formal_questions=args.augment_formal_questions,
        paraphrase_map_path=args.paraphrase_map_path,
        search_ratio=args.search_ratio,
    )
    # Tokenizer must be loaded before preprocessing so we can bake masks in.
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizing dataset and baking loss masks (num_proc={args.tokenize_workers})...")
    dataset = preprocess_dataset(dataset, tokenizer, args.max_seq_length, num_proc=args.tokenize_workers)

    # --- Model ---
    # IMPORTANT: no device_map with DeepSpeed ZeRO-3; accelerate manages placement.
    # With --qlora, DeepSpeed is ignored and device_map=auto spreads the 4-bit model
    # across all available GPUs (model parallel), keeping CPU RAM usage negligible.
    use_deepspeed = args.deepspeed is not None and not args.qlora
    if args.qlora:
        print(f"Loading model {args.model_name} in 4-bit NF4 (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # device_map="auto" can spill modules to CPU, which bitsandbytes 4-bit
        # does not support. We build max_memory explicitly:
        #   - Use total_memory * 0.90 per GPU (stable; from device properties,
        #     unlike mem_get_info which may return near-zero before CUDA context init)
        #   - Zero out GPUs already >50% occupied by other processes (e.g. Ollama)
        #   - Block CPU and disk entirely
        # Tip: use CUDA_VISIBLE_DEVICES to restrict to free GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("QLoRA requires at least one CUDA GPU. None visible.")
        max_memory = {}
        for i in range(n_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            try:
                with torch.cuda.device(i):
                    free_mem, _ = torch.cuda.mem_get_info()
                if (total_mem - free_mem) / total_mem > 0.5:
                    max_memory[i] = "0GiB"
                    continue
            except Exception:
                pass  # can't query free mem; assume GPU is available
            actual_gib = int(total_mem * 0.90 / 1024 ** 3)
            # infer_auto_device_map sizes modules using the model's bf16 dtype, not
            # the quantized 4-bit size. A 122B model appears as ~244 GiB in bf16 but
            # only ~61 GiB when quantized. Inflate the virtual budget by 4x so all
            # modules are assigned to GPU; actual VRAM consumed will be ~4x lower.
            virtual_gib = (actual_gib - 10) * 4  # -10 GiB reserve for activations
            max_memory[i] = f"{virtual_gib}GiB"
        max_memory["cpu"] = "0GiB"
        usable = {k: v for k, v in max_memory.items() if isinstance(k, int) and v != "0GiB"}
        if not usable:
            raise RuntimeError(
                f"QLoRA: all visible GPUs are occupied. max_memory={max_memory}. "
                "Set CUDA_VISIBLE_DEVICES to only include free GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1"
            )
        actual_map = {k: f"{int(v[:-3]) // 4 + 10}GiB" for k, v in usable.items()}
        print(f"  QLoRA device map budget: virtual(bf16-inflated)={dict(usable)}, actual_physical≈{actual_map}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        if use_deepspeed:
            with open(args.deepspeed) as _f:
                _ds_full = json.load(_f)
            _zero_stage = _ds_full.get("zero_optimization", {}).get("stage", 3)

            if _zero_stage == 3:
                # ZeRO-3 requires model params to be created inside deepspeed.zero.Init()
                # so they start in partitioned+offloaded form. Without this, DeepSpeed
                # calls model.to(device) during engine setup to convert CPU params, which
                # needs to fit one rank's full shard in VRAM.
                #
                # We must also init distributed first so world_size is correct — zero.Init()
                # validates train_batch_size == micro_batch * grad_acc * world_size.
                # Accelerator() triggers torch.distributed init; SFTTrainer reuses the state.
                from accelerate import Accelerator
                import torch.distributed as _dist
                _accelerator = Accelerator()
                _world_size = _dist.get_world_size() if _dist.is_initialized() else 1
                print(f"Loading model {args.model_name} in bf16 via zero.Init() "
                      f"(world_size={_world_size})...")
                import deepspeed as _ds
                # Force CPU offload during zero.Init regardless of training config.
                # Without this, ZeRO-3 broadcasts each parameter via NCCL as it is loaded
                # inside from_pretrained. MoE models run distributed ops in their custom
                # __init__ code at the same time, deadlocking the NCCL communicator.
                # With cpu offload, parameters are partitioned on CPU — no NCCL during load.
                # The training engine (SFTTrainer) then moves them to GPU using the
                # user-supplied --deepspeed config (which may or may not have offload).
                _zero_init_config = {
                    "zero_optimization": {
                        **_ds_full["zero_optimization"],
                        "offload_param": {"device": "cpu", "pin_memory": True},
                    },
                    "train_batch_size": args.batch_size * args.grad_accum * _world_size,
                    "train_micro_batch_size_per_gpu": args.batch_size,
                    "gradient_accumulation_steps": args.grad_accum,
                    "bf16": {"enabled": True},
                }
                with _ds.zero.Init(config_dict_or_path=_zero_init_config):
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_name,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                    )
            else:
                # ZeRO-2: parameters are replicated on every rank — no zero.Init needed
                # and no parameter ALLGATHER during the forward pass. This avoids the
                # NCCL communicator collision that occurs when MoE models issue their own
                # distributed ops (expert routing) interleaved with ZeRO-3's per-layer
                # parameter all-gathers. SFTTrainer initialises distributed + wraps the
                # model in the DeepSpeed engine internally.
                print(f"Loading model {args.model_name} in bf16 "
                      f"(ZeRO-{_zero_stage}, no zero.Init)...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
        else:
            print(f"Loading model {args.model_name} in bf16...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )

    # --- LoRA (skipped when --full-finetune) ---
    if args.full_finetune:
        lora_config = None
        target_modules = None
        print("Full SFT: all model parameters will be trained (no LoRA adapter).")
    else:
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
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard", "mlflow"],
        deepspeed=args.deepspeed if not args.qlora else None,
        dataset_text_field=None,
        seed=args.seed,
        optim=args.optimizer,
        dataloader_num_workers=args.dataloader_num_workers,
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
            "training_mode": "full_finetune" if args.full_finetune else "lora",
            "lora_rank": "n/a" if args.full_finetune else args.lora_rank,
            "lora_alpha": "n/a" if args.full_finetune else args.lora_alpha,
            "target_modules": str(target_modules),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "dataset_size": len(dataset),
            "data_dir": args.data_dir,
            "deepspeed_config": args.deepspeed or "none",
            "qlora": str(args.qlora),
            "augment_formal_questions": str(args.augment_formal_questions),
            "search_ratio": "none" if args.search_ratio is None else f"{args.search_ratio:.3f}",
            "optimizer": args.optimizer,
            "dataloader_num_workers": args.dataloader_num_workers,
        })

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        if lora_config is not None:
            trainer_kwargs["peft_config"] = lora_config
        trainer = SFTTrainer(**trainer_kwargs)

        if args.full_finetune:
            # Under ZeRO-3 with zero.Init, params are partitioned; param.numel()
            # returns the local shard size (often 0). Use ds_numel — the original
            # unpartitioned size DeepSpeed stashes on the parameter — when present.
            def _full_numel(p):
                return getattr(p, "ds_numel", None) or p.numel()
            n_trainable = sum(_full_numel(p) for p in trainer.model.parameters() if p.requires_grad)
            n_total = sum(_full_numel(p) for p in trainer.model.parameters())
            print(f"\nTrainable parameters: {n_trainable:,} / {n_total:,} (100% — full FT)")
        else:
            print(f"\nTrainable parameters:")
            trainer.model.print_trainable_parameters()

        print("\nStarting training...")
        train_result = trainer.train()

        mlflow.log_metric("train_loss", train_result.training_loss)

    # --- Save ---
    # Full FT: HF Trainer + DeepSpeed handles ZeRO-3 shard gathering correctly
    # when `stage3_gather_16bit_weights_on_model_save: true` is set in the DS config.
    # LoRA + ZeRO-3: trainer.save_model() routes through PEFT's save_pretrained which
    # reads parameter.data directly — only the local rank-0 shard, not the gathered
    # tensor. Must gather via accelerator.get_state_dict() before saving the adapter.
    if args.full_finetune:
        trainer.save_model(args.output_dir)
        if not trainer.is_deepspeed_enabled or trainer.accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
    elif trainer.is_deepspeed_enabled:
        state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        if trainer.accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    info = {
        "base_model": args.model_name,
        "training_mode": "full_finetune" if args.full_finetune else "lora",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "dataset_size": len(dataset),
        "training_loss": train_result.training_loss,
        "metrics": train_result.metrics,
    }
    if not args.full_finetune:
        info["lora_rank"] = args.lora_rank
        info["lora_alpha"] = args.lora_alpha
        info["target_modules"] = (
            target_modules if isinstance(target_modules, list) else "all-linear"
        )
    main_process = (not trainer.is_deepspeed_enabled) or trainer.accelerator.is_main_process
    if main_process:
        with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    # --- Optional merge ---
    if args.merge_at_end:
        if trainer.accelerator.is_main_process:
            print("\nMerging LoRA weights into base model...")
            # Load the adapter we just saved from disk into a fresh CPU model so the
            # full (gathered) weights are present before merge_and_unload().
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base, args.output_dir)
            merged = peft_model.merge_and_unload()
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
