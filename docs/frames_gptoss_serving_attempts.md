# Serving the fine-tuned gpt-oss:20b — attempts & failures

Companion to [frames_cue_robustness_sft.md](frames_cue_robustness_sft.md). Training **succeeded**;
the blocker has been **deploying the fine-tuned model for eval**. gpt-oss's MXFP4 format + MoE +
an old cluster GPU driver make every standard export/serve route hit a wall. This records what was
tried so we don't repeat it.

## What works (keep these)
- **LoRA adapter (PEFT)** — `models/gpt-oss-20b-frames-robust/` (`adapter_model.safetensors`, 8M
  params, 0.038%). The training artifact; intact.
- **Adapter → GGUF** — `models/frames_robust_adapter-F16.gguf` (192 tensors, 15.9 MB) via
  `llama.cpp/convert_lora_to_gguf.py`. Conversion itself is fine; the problem is *applying* it.
- Pipeline scripts: `create_frames_sft_data.py`, `curate_frames_sft_data.py`,
  `harmonize_sft_chatml.py`, `check_sft_tokenization.py`, `athena_frames_gptoss_sft.job` (training),
  `athena_frames_cue_eval.job` (eval, waiting on a working served model).

## Failed serving attempts

| # | Approach | Script(s) | Failure |
|---|----------|-----------|---------|
| 1 | Ollama runtime LoRA (`FROM gpt-oss:20b` + `ADAPTER`) | `athena_gptoss_adapter_gguf.job` | Ollama returns `failed to initialize model: loras are not yet implemented` — on **both** 0.18.2 (Athena) and 0.22.0 (nlp-srv3). Ollama does not apply LoRA to gpt-oss, any version. |
| 2 | transformers merge → bf16 → GGUF → Ollama full model | `merge_gptoss_lora.py`, `athena_gptoss_merge_gguf.job` | `save_pretrained` → `revert_weight_conversion` → **`NotImplementedError`**: the base loads dequantized to bf16 (no MXFP4 Triton kernels), then transformers tries to re-encode MXFP4 on save and can't. Stripping `quantization_config` did **not** help (the reversion is registered on the model instance). Produced only a 12K dir (config, no weights), twice. |
| 3 | vLLM runtime LoRA (PEFT adapter, no merge) | `athena_vllm_smoke.job` | Two sub-failures: (a) **dep conflict** — vLLM's transformers needs `huggingface-hub>=1.5`, but `pydantic_ai` (eval client) needs `<1.0`; cannot share `.venv` → fixed with an isolated `.vllm_venv` (reusing the existing 3.11 interpreter, no download). (b) **CUDA-driver wall** — n315's GPU driver is CUDA **12.8** (`found version 12080`); vLLM ≥0.20 ships **cu13** wheels (`libcudart.so.13`), and 0.26.0's only alt wheel is **cu129** — both need a newer driver. `--torch-backend cu128` fixes torch but not vLLM's own compiled kernels. No cu128 wheel exists for modern vLLM. |

## Key facts learned
- **Ollama cannot apply a LoRA to gpt-oss** (MoE), independent of version.
- **transformers cannot save a merged/dequantized gpt-oss as bf16** (`revert_weight_conversion`
  `NotImplementedError`). Unsloth notes it needed a special "on-demand MXFP4 dequant during LoRA
  merge" to make this work — i.e. the wall is real, not a mistake in our code.
- **n315's driver is CUDA 12.8** — older than any modern vLLM wheel (cu129/cu13). The base training
  venv works because its `torch 2.10.0+cu128` matches; vLLM's newer torch/kernels do not.

## Untried options (next)
- **A. `llama.cpp` `llama-export-lora`** — merge the adapter GGUF into the base gpt-oss GGUF →
  full-model GGUF → Ollama serves it as a plain model (**no runtime LoRA, no transformers save**).
  Sidesteps walls #1 and #2. Cost: build that one llama.cpp tool (compiles against local CUDA 12.8,
  so no wheel mismatch). Verify llama.cpp supports export-lora for gpt-oss MoE first.
- **B. vLLM on a newer-driver node** — if any Athena partition has CUDA ≥12.9, default vLLM 0.26
  works there (reuses the PEFT adapter directly, keeps MXFP4). Cheap to check driver versions.
- **C. Older cu128 vLLM (≤0.19)** — has cu128 wheels + gpt-oss, but LoRA-on-gpt-oss support in those
  versions is uncertain (base support landed ~0.10.1; MoE-LoRA later).
- **D. Unsloth export** — load base + our adapter in Unsloth, `save_pretrained_gguf` (its dequant-on-
  merge handles the gpt-oss save transformers can't), → Ollama. Standard-quant GGUF (q8_0/q4_k_m),
  not MXFP4, so re-run the baseline at the same quant for parity.

## Cleanup done at this checkpoint
- Removed empty merged dirs (`models/gpt-oss-20b-frames-robust_merged{,_bf16}`, 12K each), the failed
  `.vllm_venv`, temp `Modelfile.*`, and the throwaway `athena_test_adapter.job` on Athena; the dead
  `gpt-oss-frames-robust` model + `Modelfile.frames_robust` on nlp-srv3. Kept the PEFT adapter and the
  adapter GGUF (both reusable for options A/B/D).
