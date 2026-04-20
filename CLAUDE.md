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
