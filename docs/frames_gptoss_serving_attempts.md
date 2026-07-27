# Serving the fine-tuned gpt-oss:20b — attempts & failures

> **RESOLVED 2026-07-27.** The fine-tuned model was served, both Q4 variants + a Q4 vanilla control
> were evaluated on the full 7-condition cue grid, and the analysis is complete. The path that
> worked and every bug fixed along the way are at the **bottom** of this file
> ("## RESOLVED — the serving path that worked" + "## Bugs solved 2026-07-27"). The failure log
> below is kept so we don't retry the dead ends.

Companion to [frames_cue_robustness_sft.md](frames_cue_robustness_sft.md). Training **succeeded**;
the blocker was **deploying the fine-tuned model for eval**. gpt-oss's MXFP4 format + MoE +
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

---

## RESOLVED — the serving path that worked (2026-07-27)

Option **A/D hybrid**: merge the LoRA into a *clean bf16 base that carries no MXFP4 quantizer*, convert
to GGUF, quantize with the **prebuilt** `llama-quantize`, and register in Ollama with the base model's
harmony template + explicit EOG stops. No runtime LoRA, no `transformers` MXFP4 re-encode, no vLLM.

1. **Merge into `unsloth/gpt-oss-20b-BF16`** (not the MXFP4 `openai/gpt-oss-20b`) — this base loads as
   plain bf16 with **no quantizer registered**, so `save_pretrained` never calls
   `revert_weight_conversion` → sidesteps wall #2. `scripts/merge_gptoss_lora.py`
   (`base_id = argv[3] or "unsloth/gpt-oss-20b-BF16"`), `scripts/athena_gptoss_merge_gguf.job`.
2. **Convert** the merged dir → GGUF with `llama.cpp/convert_hf_to_gguf.py --outtype auto` (keeps
   experts MXFP4-precision, ~41.8 GB). Same job.
3. **MXFP4 quantize is impossible** — `llama-quantize <src> <out> MXFP4` → `invalid ftype 'MXFP4'`,
   and Ollama `--quantize mxfp4` also rejects it. MXFP4 is a valid *runtime* tensor type but **not a
   valid `llama-quantize` target**. → fall back to **Q4_K_M + Q4_K_S** via the prebuilt CPU
   `llama-quantize` (release b10142, the container has no compiler). `scripts/athena_q4_quantize.job`,
   `scripts/athena_mxfp4_quantize.job` (records the MXFP4 failure).
4. **Register in Ollama with the base template** — a bare `FROM x.gguf` Modelfile yields
   `registry ... does not support tools`. Fix: `ollama show --modelfile gpt-oss:20b` → `sed` the
   `FROM` line to our GGUF (keeps the harmony `TEMPLATE`). `scripts/athena_reregister_tools.job`.
5. **Add EOG stops** — the converted GGUF lacks gpt-oss's harmony end-of-generation metadata, so the
   model never stops at `<|call|>` and emits a malformed tool call (Ollama 500 "error parsing tool
   call"). Fix: append `PARAMETER stop "<|call|>"` and `PARAMETER stop "<|return|>"` to the Modelfile.
   Verified: clean `tool_calls` with `done_reason: stop`.

Because the served quant (Q4_K) ≠ the baseline's native MXFP4, an apples-to-apples comparison needs a
**Q4 vanilla control**: the un-fine-tuned base run through the *identical* recipe (steps 2–5) minus the
merge. `scripts/athena_gptoss_vanilla_q4.job` builds `gpt-oss-vanilla-q4km/q4ks`. See the results doc
for why this control mattered (it reattributed the whole accuracy drop to quantization).

## Bugs solved 2026-07-27 (deployment + eval)

- **Ollama port collision (host networking).** Two eval jobs on the *same* node each `ollama serve` on
  11434 (apptainer uses host net) → the second wedges at 0 rows. Fix in `athena_frames_cue_eval.job`:
  `PORT=$((11500 + SLURM_JOB_ID % 4000))`, export `OLLAMA_HOST`/`OLLAMA_BASE_URL`, wait on
  `/api/version`.
- **`.venv` torch broken by the vLLM experiments** — `libcudnn.so.9`, then `libcusparseLt.so.0`
  missing. Fix: `uv sync --reinstall` restored `torch 2.10.0+cu128` (CUDA True). Training/merge depend
  on this venv, so it can't be left broken. `scripts/athena_fix_torch.job`.
- **Logfire wedge → disable → re-enable.** In the Athena container the OTLP HTTPS exporter threw on
  every export (missing `certifi/cacert.pem`) and **wedged the workers** → disabled via
  `LOGFIRE_API_KEY=''`. After `uv sync --reinstall` restored `certifi`, removing that override
  re-enabled logfire (key comes from `.env`) with **no re-wedge**. Only disable logfire on Athena if
  `certifi` is actually missing.
- **Incomplete baseline eval.** The vanilla `gpt-oss:20b` grid (run earlier) was missing
  `verbose_direct` entirely and half of `verbose_query` (51/102). Resubmitting the *same* eval job
  with `MODEL=gpt-oss:20b` `--resume`d and filled only the gaps (EvaluationService skips done
  `example_id`s per condition file).
- **Q4-vanilla convert: base HF snapshot has no tokenizer.** The cached `unsloth/gpt-oss-20b-BF16`
  snapshot holds only `config.json` + safetensors — **no tokenizer files** — so
  `convert_hf_to_gguf.py` died with `Couldn't instantiate the backend tokenizer`. Fix: stage a dir =
  base weights (symlinked from the snapshot) + `tokenizer.json`/`tokenizer_config.json`/
  `chat_template.jinja` **copied from the merged dir** (LoRA never changes the tokenizer, so they're
  byte-identical). Also `set -e` did **not** catch the failure because the convert was piped to
  `grep` (pipe exit code = grep's `0`) → added explicit `[ -f "$GGUF" ] || exit 1` gates after
  convert and each quantize.
- **Disk quota (`OSError: … 0 written`).** `/home` is a **personal quota: 300 G soft / 330 G hard**;
  hitting 330 G* makes writes return 0 bytes (looks like a mysterious `OSError` mid-write, not
  "disk full"). `df` shows the *filesystem* (TBs free), not the quota — check `quota -s`. `/work` is a
  symlink to `/rg/reichart_prj/dvirla` (the `reichart_prj` **group** share; `df` on it read
  inconsistently, 48 G vs 36 T free). Freed space by deleting redundant BF16 intermediates
  (`gpt-oss-frames-robust.gguf` 40 G, the merged `model.safetensors` 40 G) — both reproducible from
  the 769 MB LoRA adapter. Note: `ollama create` copies the GGUF into `/work/ollama` (on `/rg`), so
  even registration consumes group space.

## Cleanup / reproducibility notes
- Kept (small, regenerates everything): the **LoRA adapter** `models/gpt-oss-20b-frames-robust/`
  (769 MB) and the merged dir's tokenizer files. Everything large (BF16 GGUF, merged safetensors) is
  reproducible: adapter → `merge_gptoss_lora.py` → `convert_hf_to_gguf.py` → `llama-quantize`.
- Registered Ollama models (on Athena `/work/ollama`): `gpt-oss-frames-robust-q4km/q4ks` (SFT) and
  `gpt-oss-vanilla-q4km/q4ks` (control), all with harmony template + EOG stops.
