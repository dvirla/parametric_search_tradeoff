#!/usr/bin/env bash
set -euo pipefail
cd ~/parametric_search_tradeoff      # adjust to your repo path

git pull                                        # needs commit 2da1515 (4th-model plotting support)

MODEL="nemotron-3-nano-musique-v3-aug:latest"   # Ollama name (colon), in your trace/eval filenames
SLUG="nemotron-3-nano-musique-v3-aug_latest"    # underscore slug used in all derived filenames

# ── Stage 0: normalize the tuned model's trace filenames to the underscore slug ──
cd results/curated_sharechat
for V in curated-sharechat curated-sharechat-benchmark; do
  cp "${V}_${MODEL}_baseline_agent_run_1_traces.json" "${V}_${SLUG}_baseline_agent_run_1_traces.json"
done
cd - >/dev/null

# ── Stage 1: reuse the BASE Nemotron uncertainty as the tuned model's reference ──
#   (LoRA leaves parametric knowledge ~unchanged; copy base -> tuned slug in BOTH dirs
#    so the interplay matches it to the tuned traces and joins by example_id)
cp results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_nemotron-3-nano_30b.json \
    results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_${SLUG}.json
cp results/sharechat_parametric/sharechat_parametric_uncertainty_nemotron-3-nano_30b.json \
    results/sharechat_parametric/sharechat_parametric_uncertainty_${SLUG}.json

# ── Stage 2: interplay attribution — RE-RUN ALL 4 MODELS into the same dirs ──
#   summary CSV is rewritten each run; the 3 base models reload from
#   matched_examples_*.json + attribution_cache.json (no new LLM calls); only the 4th is attributed.

# 2a. benchmark traces -> benchmark_interplay_analysis
uv run python scripts/analyze_parametric_search_interplay.py \
    --uncertainty-jsons \
      results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_gemini-3-pro-preview.json \
      results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_nemotron-3-nano_30b.json \
      results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_qwen3.5_122b.json \
      results/sharechat_benchmark_parametric/sharechat_parametric_uncertainty_${SLUG}.json \
    --traces \
      results/curated_sharechat/curated-sharechat-benchmark_gemini-3-pro-preview_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat-benchmark_nemotron-3-nano_30b_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat-benchmark_qwen3.5_122b_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat-benchmark_${SLUG}_baseline_agent_run_1_traces.json \
    --no-gold-answer \
    --output-dir results/curated_sharechat/benchmark_interplay_analysis \
    --use-llm --attribution-model gpt-oss:120b --attribution-provider ollama

# 2b. natural (real) traces -> interplay_analysis
uv run python scripts/analyze_parametric_search_interplay.py \
    --uncertainty-jsons \
      results/sharechat_parametric/sharechat_parametric_uncertainty_gemini-3-pro-preview.json \
      results/sharechat_parametric/sharechat_parametric_uncertainty_nemotron-3-nano_30b.json \
      results/sharechat_parametric/sharechat_parametric_uncertainty_qwen3.5_122b.json \
      results/sharechat_parametric/sharechat_parametric_uncertainty_${SLUG}.json \
    --traces \
      results/curated_sharechat/curated-sharechat_gemini-3-pro-preview_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat_nemotron-3-nano_30b_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat_qwen3.5_122b_baseline_agent_run_1_traces.json \
      results/curated_sharechat/curated-sharechat_${SLUG}_baseline_agent_run_1_traces.json \
    --no-gold-answer \
    --output-dir results/curated_sharechat/interplay_analysis \
    --use-llm --attribution-model gpt-oss:120b --attribution-provider ollama

# ── Stage 3: commitment-locus probe (4th model only) ──
uv run python scripts/probe_commitment_locus.py \
    --benchmark-matched "results/curated_sharechat/benchmark_interplay_analysis/matched_examples_${SLUG}.json" \
    --natural-matched   "results/curated_sharechat/interplay_analysis/matched_examples_${SLUG}.json" \
    --benchmark-traces  "results/curated_sharechat/curated-sharechat-benchmark_${SLUG}_baseline_agent_run_1_traces.json" \
    --natural-traces    "results/curated_sharechat/curated-sharechat_${SLUG}_baseline_agent_run_1_traces.json" \
    --model-slug "$SLUG" \
    --judge_model gpt-oss:120b --judge_provider ollama \
    --output-dir results/curated_sharechat/commitment_locus \
    --resume

# ── Stage 4: regenerate all ShareChat figures (now includes the 4th model) ──
uv run python scripts/make_paper_figures.py --mode sharechat