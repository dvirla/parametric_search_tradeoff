# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating tradeoffs between parametric (model-internal) knowledge and search-augmented QA systems. Evaluates how different LLMs balance internal knowledge against external search results when answering questions.

## Running Commands

All commands use **UV** as the package manager (not pip):

```bash
# Install dependencies
uv sync

# Run a QA evaluation experiment
uv run python scripts/run_qa_eval_experiment.py \
  --agent_type iterative \
  --model_name gemini-3-pro-preview \
  --dataset facts-search \
  --num_examples 100

# Agent types: baseline, no_search, iterative, generalized
# Datasets: facts-param, facts-search, nq

# Downloading search agent traces from logfire
uv run python scripts/download_traces.py --agent-name baseline_agent --output-dir <dir> --model-name gemini-3-pro-preview

# Analysis of pre-search reasoning step
uv run python scripts/analyze_misalignment.py --traces <traces_downloaded_previously> --agent_eval <evaluation_json_path> --output <csv_name_and_path>

# Semantic entropy calculation
uv run python scripts/calculate_semantic_entropy.py

# Unified analysis (replaces visualize_results.py and analyze_semantic_entropy.py)
uv run python scripts/unified_analysis.py \
  --model-name "Gemini 3 Pro" \
  --datasets \
    "facts_one_hop:path/analysis.csv:path/json_dir:path/traces.json:path/entropy.csv" \
    "popqa:path/to/baseline_run_*_analysis.csv:path/json_dir::path/entropy.csv" \
  --output-dir results/gemini_3_pro \
  --aggregate
```

There is no test suite, linter configuration, or build step.

## Architecture

### Core Services (`src/services/`)

- **`base_agent.py`** — `BaseAgent`: Unified LLM interface wrapping pydantic-ai Agent. Supports Google, OpenAI, Anthropic, and Ollama providers with thinking mode, tool integration, and retry logic.
- **`agent_sampler.py`** — `AgentAsSampler`: Adapter making `BaseAgent` compatible with the `SamplerBase` evaluation interface. Tracks search tool usage and integrates with Logfire observability.
- **`service_types.py`** — Core type definitions: `Message`, `SamplerBase`, `EvalResult`, `SingleEvalResult`, `Eval`.
- **`iterative_search_agent.py`** — Main agentic QA loop: generates multiple drafts, clusters them semantically, calculates uncertainty, and triggers search when uncertain. Uses sub-agents for clustering, distillation, gap analysis, and validation. Max 4 steps.
- **`generalized_iterative_search_agent.py`** — Extended agent that decomposes complex multi-step questions into atomic queries before synthesis.
- **`ollama_thinking_agent.py`** — Native Ollama integration with first-class thinking/reasoning support (deepseek-r1, qwq).
- **`qa_eval.py`** — `EvaluationService`: Systematic QA evaluation across datasets (FACTS-Parametric, FACTS-Search, Natural Questions). LLM-based grading, result persistence, resumable runs.
- **`brave_search.py`** — `BraveSearchService`: Web search with pagination and exponential backoff.
- **`common.py`** — Utilities for answer extraction/normalization, HTML report generation, parallel evaluation, and statistics.

### Analysis Scripts (`scripts/`)

- **`run_qa_eval_experiment.py`** — Main entry point for running experiments. CLI-driven with model, dataset, and agent type selection.
- **`calculate_semantic_entropy.py`** — Clusters equivalent answers and computes semantic entropy per problem.
- **`unified_analysis.py`** — Per-model analysis combining epistemic state, semantic entropy, stability (5-run), and cross-dataset aggregation. Generates plots, CSVs, and a Markdown report.
- **`agent_comparison_analysis.py`** — Cross-model/cross-agent behavioral analysis.
- **`analyze_misalignment.py`** — Detects when models ignore or contradict search findings.
- **`re_evaluate_logs.py`** — Re-grades existing logs with different judge models.

### Data Flow

1. `run_qa_eval_experiment.py` orchestrates evaluation using `EvaluationService`
2. `EvaluationService` loads datasets and runs an agent (baseline/iterative/generalized) via `AgentAsSampler`
3. The agent uses `BaseAgent` for LLM calls and optionally `BraveSearchService` for web search
4. Results are persisted as JSON in `logs/<dataset>/<model>/`
5. Analysis scripts consume these logs to produce metrics, CSVs, and visualizations

## Key Conventions

- Multi-provider LLM access is abstracted through pydantic-ai — model switching is done via string model names
- Environment variables for API keys are loaded from `.env` (TAVILY_API_KEY, GOOGLE_API_KEY, BRAVE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Experiment results go in `logs/` organized by `<dataset>/<model>/`
- Python 3.11+ required

---

## MusiQue SFT Pipeline

End-to-end supervised fine-tuning pipeline for training a search-augmented QA model on MusiQue. Validates the hypothesis that training on targeted search traces (covering missed reasoning hops) improves multi-hop QA performance.

### Overview

Three independent stages that can run in parallel where noted:

```
Stage 1 (parallel):
  A) run_musique_parametric_uncertainty.py  →  per-hop uncertainty JSON
  B) paraphrase_musique_natural.py          →  natural-language paraphrases JSONL

Stage 2 (depends on A+B):
  create_musique_sft_data.py (full scale)
  create_musique_sft_data_from_existing.py (val-set validation, no new rollouts)

Stage 3 (GPU machine):
  train_sft.py  →  LoRA checkpoint

Stage 4 (GPU machine):
  eval_checkpoint.sh  →  results comparable to vanilla baseline
```

### Stage 1A — Parametric Uncertainty

```bash
uv run python scripts/run_musique_parametric_uncertainty.py \
    --model_name qwen3.5:122b \
    --provider ollama \
    --num_runs 5 \
    --staleness_csv data/musique_train_staleness.csv \
    --output_dir results/musique_parametric \
    --skip-aggregate \
    --resume
```

Outputs `results/musique_parametric/musique_parametric_uncertainty_qwen3.5_122b.json`.

**`--skip-aggregate`**: Always use this flag when generating SFT data — it skips the search-augmented aggregate run which is redundant (the SFT script runs its own rollouts). Without it you waste Brave API credits.

### Stage 1B — Natural Paraphrases (runs in parallel with 1A)

```bash
uv run python scripts/paraphrase_musique_natural.py \
    --staleness-csv data/musique_train_staleness.csv \
    --output data/musique_train_natural.jsonl \
    --all-hops \
    --model gpt-oss:20b
```

Loads directly from the staleness CSV + HuggingFace dataset — **no dependency on Stage 1A**. Uses `--staleness-csv` mode (not `--source`). The two input modes are mutually exclusive.

### Stage 2 — SFT Data Collection (full scale)

Requires: Stage 1A output + Stage 1B output + Ollama server running.

```bash
uv run python scripts/create_musique_sft_data.py \
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_qwen3.5_122b.json \
    --natural-jsonl data/musique_train_natural.jsonl \
    --model qwen3.5:122b \
    --provider ollama \
    --k 5 \
    --output-dir data/sft/musique \
    --resume
```

**Two-procedure rejection sampling** (K=5 rollouts each):
- **Procedure 1** (calibration-filtered natural): K natural rollouts → R1: correct + R2: all uncertain hops searched (semantic_entropy > 0) → keep verbatim. Optional `--r3-strict`: also reject if certain hops were over-searched.
- **Procedure 2** (M-patching): K natural rollouts → R1: correct + R2: ≥1 missed uncertain hop → for each missed hop: find canonical query in formal trace → consistency check (reject if trace reached wrong intermediate entity) → splice at correct reasoning position + bridge thinking → K continuation rollouts → keep if correct + no new M-violation. Each (rollout, missed-hop) pair is a separate training example.

**Budget optimisation**: formal rollout is only run per-example when at least one natural rollout is a P2 candidate (correct + missed uncertain hop). `--k-formal` controls how many formal rollouts to attempt (default 1).

All models default to `ollama` provider. **No cloud API calls happen by default.**  The script fails immediately with a clear error if Ollama is not running.

**Uncertainty definition**: a hop is uncertain if `semantic_entropy > 0` from the parametric probe output. Certain hops skipped by the model are classified as CP (correct parametric), not M.

**Hop attribution** uses the same `ATTRIBUTION_PROMPT` and `QueryAttribution` pydantic model as `analyze_parametric_search_interplay.py` — LLM-based, not keyword heuristics. Attribution is pre-computed once per rollout and reused for both procedures.

**Canonical query source**: for Procedure 2, the search query spliced in comes from the actual search query in the formal trace (not the formal sub-question text). This prevents the mediator from redirecting reasoning to formal-phrased entities.

**Consistency check**: before patching, the mediator verifies that the canonical query is consistent with what the natural trace established (e.g., if the trace found entity X but the canonical query asks about entity Y, the sample is rejected).

**Grading natural answers** is open-ended: checks whether the gold answer string appears explicitly in the model's free-form response text.

Outputs:
- `data/sft/musique/procedure1_natural_sft.jsonl`
- `data/sft/musique/procedure2_m_patching.jsonl`

Each line: `{"messages": [...]}` in ChatML format (Qwen-compatible).

### Stage 2 (alt) — Small-scale Validation from Existing Val-set Traces

Use this to validate the pipeline **before** committing to a full-scale data collection run. Reuses existing `results/musique_parametric/` and `results/musique-natural/` traces — no new rollouts for Arms 1 & 2.

```bash
uv run python scripts/create_musique_sft_data_from_existing.py \
    --uncertainty-dir results/musique_parametric \
    --natural-dir results/musique-natural \
    --natural-jsonl data/musique_val_natural.jsonl \
    --model-slug qwen3.5_122b \
    --continuation-model qwen3.5:122b \
    --continuation-provider ollama \
    --output-dir data/sft/musique_val_test \
    --resume
```

**Model slug vs. model name**: `--model-slug` is the filename stem (underscores, e.g. `qwen3.5_122b`); the eval JSON uses colons (e.g. `qwen3.5:122b`). The loader tries both forms automatically when matching natural eval files.

**Natural dir naming**: the directory is `results/musique-natural` (dash), not `results/musique_natural` (underscore). Easy to get wrong.

Expected yields from 600 val examples (K=1): Procedure 1 varies by uncertainty profile; Procedure 2 depends on consistency check pass rate and continuation model availability.

### Stage 3 — Training (GPU machine)

Install training dependencies first:

```bash
uv sync --extra training
```

```bash
accelerate launch --config_file configs/accelerate_zero3.yaml scripts/train_sft.py \
    --model-name Qwen/Qwen2.5-72B-Instruct \
    --data-dir data/sft/musique \
    --output-dir models/qwen-musique-lora \
    --deepspeed configs/deepspeed_zero3.json \
    --merge-at-end
```

Key training flags:
- `--merge-at-end`: merges LoRA adapter into base weights after training; saves to `<output-dir>_merged` in HF format ready for GGUF conversion or vLLM serving
- `--deepspeed configs/deepspeed_zero3.json`: required for 122B; uses ZeRO-3 with CPU offload. Drop `offload_optimizer`/`offload_param` blocks in the JSON if you have 8× H100 80GB (pure GPU ZeRO-3 is faster)
- **Do NOT pass `device_map="auto"` with DeepSpeed** — the script handles this correctly already

**MLflow**: tracked automatically. View with `mlflow ui --backend-store-uri ./mlruns`. TRL auto-logs step-level metrics; hyperparams are logged explicitly at run start.

**Serving the checkpoint via Ollama** (preferred — keeps eval apples-to-apples with baseline):

```bash
# Convert merged HF checkpoint to GGUF (use same quantization as baseline)
python llama.cpp/convert_hf_to_gguf.py models/qwen-musique-lora_merged --outtype q8_0

# Register with Ollama
echo "FROM ./models/qwen-musique-lora_merged/model-Q8_0.gguf" > Modelfile
ollama create qwen3.5_musique_lora -f Modelfile
```

Alternatively, serve via vLLM (no quantization, full bf16):

```bash
vllm serve models/qwen-musique-lora_merged \
    --served-model-name qwen3.5_musique_lora \
    --tensor-parallel-size 8 \
    --port 8001
# Then set OLLAMA_BASE_URL=http://localhost:8001/v1 before running eval
```

### Stage 4 — Checkpoint Evaluation

```bash
# With Ollama (model already registered via ollama create):
bash scripts/eval_checkpoint.sh qwen3.5_musique_lora

# With vLLM (model served on a custom port):
OLLAMA_BASE_URL=http://localhost:8001/v1 bash scripts/eval_checkpoint.sh qwen3.5_musique_lora 8001
```

Runs three evals in sequence:
1. `run_musique_parametric_uncertainty.py` — per-hop semantic entropy
2. `run_musique_experiment.py --mode with_search`
3. `run_musique_experiment.py --mode no_search`

Results land in `results/musique_parametric/` and `results/musique/` with the model slug in the filename — identical naming to baseline results, so `unified_analysis.py` and `scripts/make_paper_figures.py` pick them up without changes.

### Paper figures

All figures + stats for the inside-out paper (`paper/draft_inside_out.tex`) are produced by a single script reading the precomputed producer outputs (`interplay_summary.csv`, `matched_examples_*.json`, `commitment_locus_*.csv`):

```bash
uv run python scripts/make_paper_figures.py --output-dir results/paper_figures
```

Palette / stats / cell-assignment helpers live in `src/viz.py` (4-colour ColorBrewer RdBu: `#0571b0` E, `#92c5de` CP, `#f4a582` PR, `#ca0020` M). The two upstream LLM producers are `analyze_parametric_search_interplay.py` (query→hop attribution) and `probe_commitment_locus.py` (commitment-locus judge).

### Key Pitfalls

| Pitfall | Detail |
|---------|--------|
| **No Ollama, no run** | Both SFT data scripts call `_check_ollama()` at startup and exit immediately if `localhost:11434` is unreachable. All models default to `ollama` provider. |
| **`--skip-aggregate` missing** | Omitting this from `run_musique_parametric_uncertainty.py` when generating SFT data triggers a full Brave search run per example — expensive and unnecessary. |
| **`musique-natural` vs `musique_natural`** | The results directory uses a dash. Passing underscore silently produces zero natural traces and zero Arm 1/3 data. |
| **Model slug vs. model name** | Slug uses underscores (`qwen3.5_122b`) for filenames; Ollama model name uses colons (`qwen3.5:122b`). The loader normalises automatically, but `--served-model-name` in vLLM must match exactly what `--model_name` you pass to eval scripts. |
| **ZeRO-3 + merge** | `stage3_gather_16bit_weights_on_model_save: true` must be set in the DeepSpeed config (it is, in `configs/deepspeed_zero3.json`). The `--merge-at-end` logic is guarded by `accelerator.is_main_process`. |
| **Arm 3 retry logic** | An example is only marked done if Arm 3 was ineligible or succeeded. If the continuation model is unavailable, examples stay pending and retry on `--resume`. Check progress with `python3 -c "import json; print(len(json.load(open('data/sft/.../progress.json'))))"`. |
| **Quantization consistency** | Fine-tuned model should use the same GGUF quantization level as the baseline Ollama model for fair comparison. Use `q8_0` if unsure. |
| **Cloud API calls** | `gpt-4.1-mini` was the original default for attribution/mediator — now changed to `gpt-oss:20b` via ollama. If you override to a cloud provider, ensure you have credits loaded. |

### File Locations

| Purpose | Path |
|---------|------|
| Staleness labels (train) | `data/musique_train_staleness.csv` |
| Natural paraphrases (train) | `data/musique_train_natural.jsonl` |
| Natural paraphrases (val) | `data/musique_val_natural.jsonl` |
| Uncertainty JSON (val, per model) | `results/musique_parametric/musique_parametric_uncertainty_{slug}.json` |
| Natural traces (val, per model) | `results/musique-natural/musique_val_search_{slug}_baseline_agent_run_1_traces_*.json` |
| Natural eval (val, per model) | `results/musique-natural/musique-natural_baseline_{model}_run_1.json` |
| SFT training data (full scale) | `data/sft/musique/procedure{1,2}_*.jsonl` |
| SFT training data (val test) | `data/sft/musique_val_test/procedure{1,2}_*.jsonl` |
| DeepSpeed ZeRO-3 config | `configs/deepspeed_zero3.json` |
| LoRA adapter (post-training) | `models/qwen-musique-lora/` |
| Merged checkpoint | `models/qwen-musique-lora_merged/` |
