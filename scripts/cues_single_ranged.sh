#!/usr/bin/env bash
# Single-model cue sweep (no_search), generalized over an arbitrary run-index range.
# --run_name is "<cue>_run_<N>" (unique per cue, no collision with plain baseline or
# between cues -- see scripts/cluster_cues_llm_judge.py header for why that matters).
# Resumable per file: skips a (cue, run) pair whose output already exists at target size,
# and relies on run_qa_eval_experiment.py's own row-level resume for partial files.
#
# Usage: bash cues_single_ranged.sh <model> <workers> <repo_root> <run_start> <run_end>
#   e.g. bash cues_single_ranged.sh gpt-oss:20b 4 /data/home/dvirla/parametric_search_tradeoff 4 5
set -uo pipefail
MODEL="$1"; WORKERS="$2"; REPO="${3:-/data/home/dvirla/parametric_search_tradeoff}"
RUN_START="${4:-1}"; RUN_END="${5:-3}"
cd "$REPO"
FRAMES_FILE=data/frames_cues/orig_phrasing_full.jsonl
MULTITURN_HIST=data/frames_cues/chit_chat_multi_turn.json
SEARCHMULTI_HIST=data/frames_cues/search_multi_turn.json

declare -A CUES=(
  [elaborate]="elaborate "
  [direct]="direct "
  [confident_parametric]="confident_parametric "
  [multiturn]="plain $MULTITURN_HIST"
  [searchmulti]="plain $SEARCHMULTI_HIST"
)
CUE_ORDER=(elaborate direct confident_parametric multiturn searchmulti)

slug="${MODEL//:/_}"
echo "[$MODEL] START $(date -Is) runs ${RUN_START}-${RUN_END}"
for cue in "${CUE_ORDER[@]}"; do
  read -r template hist <<< "${CUES[$cue]}"
  hist_args=()
  [ -n "$hist" ] && hist_args=(--history_path "$hist")

  for ((i=RUN_START; i<=RUN_END; i++)); do
    r="${cue}_run_${i}"
    echo "[$MODEL] FRAMES $r"
    uv run python scripts/run_qa_eval_experiment.py \
      --dataset frames-cues --dataset_path "$FRAMES_FILE" \
      --query_template "$template" --agent_type no_search \
      --model_name "$MODEL" --provider_name ollama \
      --run_name "$r" --output_dir "results/frames_parametric/${slug}" \
      --num_workers "$WORKERS" --no_grader "${hist_args[@]}"

    echo "[$MODEL] MEDQA $r"
    uv run python scripts/run_qa_eval_experiment.py \
      --dataset medqa-500 \
      --query_template "$template" --agent_type no_search \
      --model_name "$MODEL" --provider_name ollama \
      --run_name "$r" --output_dir "results/medqa_parametric/${slug}" \
      --num_workers "$WORKERS" --no_grader "${hist_args[@]}"
  done
done
echo "[$MODEL] DONE $(date -Is)"
