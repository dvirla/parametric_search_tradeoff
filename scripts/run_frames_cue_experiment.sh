#!/usr/bin/env bash
# Run the FRAMES prompt-cue sensitivity experiment: each cue-condition file produced by
# scripts/paraphrase_frames_cues.py is evaluated with the baseline search agent over the FREE
# local BM25 index (data/frames_index). The dependent variables (search_calls, correct) are
# recorded per problem by the eval. Resumable.
#
# Usage:
#   bash scripts/run_frames_cue_experiment.sh                      # all default models, all conditions
#   bash scripts/run_frames_cue_experiment.sh "gpt-oss:20b"        # one model
#   CONDITIONS="neutral epi_strong_hedge" bash scripts/run_frames_cue_experiment.sh "qwen3.5:122b"
#   NUM_EXAMPLES=20 bash scripts/run_frames_cue_experiment.sh "gpt-oss:20b"   # smoke test
#
# Env overrides: CONDITIONS, NUM_EXAMPLES, CUES_DIR, INDEX_DIR, LOCAL_BACKEND, RESULTS_ROOT, RUN_NAME_SUFFIX
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_MODELS=("gemini-3-pro-preview" "qwen3.5:122b" "nemotron-3-nano")
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

DEFAULT_CONDITIONS=(neutral epi_strong_boost epi_mild_boost epi_mild_hedge epi_strong_hedge \
                    framing_first_person mitigation_polite structural_implicit \
                    output_final_only output_detailed)
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-${DEFAULT_CONDITIONS[*]}}"

CUES_DIR="${CUES_DIR:-data/frames_cues}"
INDEX_DIR="${INDEX_DIR:-data/frames_index}"
LOCAL_BACKEND="${LOCAL_BACKEND:-bm25}"
RESULTS_ROOT="${RESULTS_ROOT:-results/frames_cues}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
# Grader LLM. Defaults to local ollama gpt-oss:20b; override when no local ollama, e.g.
#   GRADER_PROVIDER=Google GRADER_MODEL=gemini-3-flash-preview bash scripts/run_frames_cue_experiment.sh ...
GRADER_MODEL="${GRADER_MODEL:-gpt-oss:20b}"
GRADER_PROVIDER="${GRADER_PROVIDER:-ollama}"
EXTRA=(--grader_model "${GRADER_MODEL}" --grader_provider "${GRADER_PROVIDER}")
[[ -n "${NUM_EXAMPLES:-}" ]] && EXTRA+=(--num_examples "${NUM_EXAMPLES}")

# model name -> provider
provider_for() {
  case "$1" in
    gemini*|*gemini*) echo "Google" ;;
    gpt-4*|gpt-3*|o1*|o3*) echo "OpenAI" ;;
    claude*) echo "Anthropic" ;;
    *) echo "ollama" ;;   # qwen3.5:122b, nemotron-3-nano, gpt-oss:*, etc.
  esac
}

for model in "${MODELS[@]}"; do
  provider="$(provider_for "$model")"
  slug_model="${model//:/_}"; slug_model="${slug_model//\//_}"
  out_dir="${RESULTS_ROOT}/${slug_model}"
  mkdir -p "$out_dir"
  echo "================ MODEL: $model (provider=$provider) -> $out_dir ================"
  for cond in "${CONDITIONS_ARR[@]}"; do
    cond_file="${CUES_DIR}/${cond}.jsonl"
    if [[ ! -f "$cond_file" ]]; then
      echo "  [skip] $cond_file not found (run paraphrase_frames_cues.py first)"; continue
    fi
    echo "  ---- condition: $cond ----"
    uv run python scripts/run_qa_eval_experiment.py \
      --dataset frames-cues --dataset_path "$cond_file" \
      --search-backend local --index-dir "$INDEX_DIR" --local-backend "$LOCAL_BACKEND" \
      --agent_type baseline --model_name "$model" --provider_name "$provider" \
      --run_name "${cond}${RUN_NAME_SUFFIX}" --output_dir "$out_dir" \
      --num_workers "${NUM_WORKERS:-4}" --resume "${EXTRA[@]}"
  done
done
echo "Done. Results under ${RESULTS_ROOT}/<model>/frames-cues_baseline_<model>_<condition>.json"
