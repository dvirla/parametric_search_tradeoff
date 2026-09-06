#!/usr/bin/env bash
# Single-model cue sweep (no_search), generalized over an arbitrary run-index range.
# --run_name is "<cue>_run_<N>" (unique per cue, no collision with plain baseline or
# between cues -- see scripts/cluster_cues_llm_judge.py header for why that matters).
# Resumable per file: skips a (cue, run) pair whose output already exists at target size,
# and relies on run_qa_eval_experiment.py's own row-level resume for partial files.
#
# Usage: bash cues_single_ranged.sh <model> <workers> <repo_root> <run_start> <run_end>
#   e.g. bash cues_single_ranged.sh gpt-oss:20b 4 /data/home/dvirla/parametric_search_tradeoff 4 5
#
# Env overrides (defaults preserve the original behaviour exactly):
#   CUES      subset/order of conditions, space separated. Default = the 5 cue conditions.
#             `plain` is also available and writes the NO-CUE run name (see PLAIN naming below).
#   DATASETS  "frames medqa" (default) | "frames" | "medqa".
#
# PLAIN NAMING IS NOT SYMMETRIC, and must not be "fixed". The plain parametric baselines were
# produced by scripts/run_parametric_probe_experiment.sh, which writes `..._run_<i>.json` with NO
# condition token, while every cue writes `..._<cue>_run_<i>.json`. Anything compared against those
# baselines has to reproduce that asymmetry or it will not pair by filename.
set -uo pipefail
MODEL="$1"; WORKERS="$2"; REPO="${3:-/data/home/dvirla/parametric_search_tradeoff}"
RUN_START="${4:-1}"; RUN_END="${5:-3}"
cd "$REPO"
FRAMES_FILE=data/frames_cues/orig_phrasing_full.jsonl
MULTITURN_HIST=data/frames_cues/chit_chat_multi_turn.json
SEARCHMULTI_HIST=data/frames_cues/search_multi_turn.json

declare -A CUE_MAP=(
  [plain]="plain "
  [elaborate]="elaborate "
  [direct]="direct "
  [confident_parametric]="confident_parametric "
  [multiturn]="plain $MULTITURN_HIST"
  [searchmulti]="plain $SEARCHMULTI_HIST"
)
read -r -a CUE_ORDER <<< "${CUES:-elaborate direct confident_parametric multiturn searchmulti}"
read -r -a DATASETS_ARR <<< "${DATASETS:-frames medqa}"
want_ds() { local d; for d in "${DATASETS_ARR[@]}"; do [[ "$d" == "$1" ]] && return 0; done; return 1; }

slug="${MODEL//:/_}"
echo "[$MODEL] START $(date -Is) runs ${RUN_START}-${RUN_END}"
for cue in "${CUE_ORDER[@]}"; do
  spec="${CUE_MAP[$cue]:-}"
  if [[ -z "$spec" ]]; then echo "[$MODEL]   [skip] unknown cue: $cue"; continue; fi
  read -r template hist <<< "$spec"
  hist_args=()
  [ -n "$hist" ] && hist_args=(--history_path "$hist")

  for ((i=RUN_START; i<=RUN_END; i++)); do
    # See "PLAIN NAMING" above: plain carries no condition token, every cue does.
    if [[ "$cue" == "plain" ]]; then r="run_${i}"; else r="${cue}_run_${i}"; fi
    if want_ds frames; then
    echo "[$MODEL] FRAMES $r"
    uv run python scripts/run_qa_eval_experiment.py \
      --dataset frames-cues --dataset_path "$FRAMES_FILE" \
      --query_template "$template" --agent_type no_search \
      --model_name "$MODEL" --provider_name ollama \
      --run_name "$r" --output_dir "results/frames_parametric/${slug}" \
      --num_workers "$WORKERS" --no_grader "${hist_args[@]}"
    fi

    if want_ds medqa; then
    echo "[$MODEL] MEDQA $r"
    uv run python scripts/run_qa_eval_experiment.py \
      --dataset medqa-500 \
      --query_template "$template" --agent_type no_search \
      --model_name "$MODEL" --provider_name ollama \
      --run_name "$r" --output_dir "results/medqa_parametric/${slug}" \
      --num_workers "$WORKERS" --no_grader "${hist_args[@]}"
    fi
  done
done
echo "[$MODEL] DONE $(date -Is)"
