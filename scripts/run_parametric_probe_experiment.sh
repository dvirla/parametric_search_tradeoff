#!/usr/bin/env bash
# Parametric knowledge probe: for each model, answer the PLAIN ORIGINAL FRAMES + MedQA questions
# with the search tool REMOVED (--agent_type no_search), repeated RUNS times (run_1..run_N) so we
# can gauge answer consistency/stability against the search baselines. This is the search-free
# counterpart of scripts/run_medqa_grid_experiment.sh / run_frames_grid_experiment.sh -- it reuses
# the same provider_for / rows_done / MAX_PASSES retry loop / parallel-vs-sequential Ollama block,
# but drops the search backend and the cue-condition sweep.
#
# "plain original" = the verbose/original phrasing sent through the plain passthrough template:
#   FRAMES -> --dataset frames-cues --dataset_path data/frames_cues/orig_phrasing_full.jsonl (501 rows;
#             frames-cues is force-plain, so --query_template plain is explicit-but-redundant here)
#   MedQA  -> --dataset medqa-500 --query_template plain (fixed 500-row original clinical vignettes)
# Both are paired on the SAME example_ids as the existing search grids.
#
# The no_search agent never touches the search tool, so NO search backend / index is needed
# (--search-backend defaults to brave, constructed-but-unused; no data/*_index on disk required).
# Grader held constant at gemini-3-flash-preview (Google) for apples-to-apples grading. Resumable.
#
# Output:
#   <RESULTS_ROOT_FRAMES>/<slug>/frames-cues_no_search_<model>_run_<r>.json
#   <RESULTS_ROOT_MEDQA>/<slug>/medqa-500_no_search_<model>_run_<r>.json
#
# Usage:
#   bash scripts/run_parametric_probe_experiment.sh                                   # default models
#   bash scripts/run_parametric_probe_experiment.sh gemma4:31b nemotron-3-nano:30b    # subset (srv3)
#   DATASETS=medqa RUNS=1 bash scripts/run_parametric_probe_experiment.sh gpt-oss:20b # one dataset, one run
#   NUM_EXAMPLES=5 RUNS=1 bash scripts/run_parametric_probe_experiment.sh gpt-oss:20b # smoke
#   DRYRUN=1 bash scripts/run_parametric_probe_experiment.sh gemma4:31b               # print commands only
#
# Env overrides: DATASETS (default "frames medqa"), RUNS (3), NUM_EXAMPLES, NUM_WORKERS (4),
#                MAX_PASSES (4), PARALLEL (auto), GRADER_MODEL, GRADER_PROVIDER, FRAMES_FILE,
#                RESULTS_ROOT_FRAMES, RESULTS_ROOT_MEDQA, DRYRUN.
set -euo pipefail
cd "$(dirname "$0")/.."

# Default = the two-box union (srv3 <=31B + Athena 120B-class). Pass a subset per machine.
DEFAULT_MODELS=("qwen3.5:122b" "gpt-oss:120b" "nemotron-3-super" \
                "gemma4:31b" "nemotron-3-nano:30b" "gpt-oss:20b")
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

read -r -a DATASETS_ARR <<< "${DATASETS:-frames medqa}"
RUNS="${RUNS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PASSES="${MAX_PASSES:-4}"
DRYRUN="${DRYRUN:-0}"
PARALLEL="${PARALLEL:-auto}"
# Grader held constant across models for apples-to-apples grading (compute-node internet works).
GRADER_MODEL="${GRADER_MODEL:-gemini-3-flash-preview}"
GRADER_PROVIDER="${GRADER_PROVIDER:-Google}"

FRAMES_FILE="${FRAMES_FILE:-data/frames_cues/orig_phrasing_full.jsonl}"
RESULTS_ROOT_FRAMES="${RESULTS_ROOT_FRAMES:-results/frames_parametric}"
RESULTS_ROOT_MEDQA="${RESULTS_ROOT_MEDQA:-results/medqa_parametric}"

provider_for() {
  case "$1" in
    gemini*|*gemini*) echo "Google" ;;
    gpt-4*|gpt-3*|o1*|o3*) echo "OpenAI" ;;
    claude*) echo "Anthropic" ;;
    *) echo "ollama" ;;   # qwen3.5:*, gemma4:*, nemotron-*, gpt-oss:*, etc.
  esac
}

# rows currently written to a results json (0 if absent)
rows_done() { uv run python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

# echo per-(model,dataset) the dataset args, result dir, filename prefix, and full-file target rows.
dataset_meta() {
  # sets globals: DS_ARGS (array), OUT_DIR, FILE_PREFIX, FULL_TARGET
  local model="$1" dataset="$2" slug="$3"
  case "$dataset" in
    frames)
      DS_ARGS=(--dataset frames-cues --dataset_path "$FRAMES_FILE")
      OUT_DIR="${RESULTS_ROOT_FRAMES}/${slug}"
      FILE_PREFIX="frames-cues_no_search_${model}"
      FULL_TARGET="$(wc -l < "$FRAMES_FILE" | tr -d ' ')"
      ;;
    medqa)
      DS_ARGS=(--dataset medqa-500)
      OUT_DIR="${RESULTS_ROOT_MEDQA}/${slug}"
      FILE_PREFIX="medqa-500_no_search_${model}"
      FULL_TARGET=500
      ;;
    *) echo "[skip] unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

# Run all datasets x runs for ONE model. Prefixes stdout so parallel logs stay readable.
run_model() {
  local model="$1"
  local provider; provider="$(provider_for "$model")"
  local slug="${model//:/_}"; slug="${slug//\//_}"
  echo "[$model] START (provider=$provider)"
  for dataset in "${DATASETS_ARR[@]}"; do
    dataset_meta "$model" "$dataset" "$slug" || continue
    mkdir -p "$OUT_DIR"
    # Full-file target unless NUM_EXAMPLES caps it (smoke).
    local target="$FULL_TARGET"
    [[ -n "${NUM_EXAMPLES:-}" && "${NUM_EXAMPLES}" -lt "$target" ]] && target="$NUM_EXAMPLES"
    for ((r=1; r<=RUNS; r++)); do
      local out_json="${OUT_DIR}/${FILE_PREFIX}_run_${r}.json"
      echo "[$model]   ---- $dataset run_${r} (target=$target) ----"
      for ((pass=1; pass<=MAX_PASSES; pass++)); do
        local cmd=(uv run python scripts/run_qa_eval_experiment.py
          "${DS_ARGS[@]}" --query_template plain --agent_type no_search
          --model_name "$model" --provider_name "$provider"
          --grader_provider "$GRADER_PROVIDER" --grader_model "$GRADER_MODEL"
          --run_name "run_${r}" --output_dir "$OUT_DIR"
          --num_workers "$NUM_WORKERS" --resume)
        [[ -n "${NUM_EXAMPLES:-}" ]] && cmd+=(--num_examples "$NUM_EXAMPLES")
        if [[ "$DRYRUN" == "1" ]]; then
          echo "[$model]     [dryrun] ${cmd[*]}"
          break 1
        fi
        "${cmd[@]}" || true
        local n; n=$(rows_done "$out_json")
        echo "[$model]     $dataset run_${r} pass $pass: $n/$target"
        [[ "$n" -ge "$target" ]] && break
      done
    done
  done
  echo "[$model] DONE"
}

# Decide parallel vs sequential. 'auto' => parallel when more than one model is given.
run_parallel=0
if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then run_parallel=1; fi
if [[ "$PARALLEL" == "auto" && ${#MODELS[@]} -gt 1 ]]; then run_parallel=1; fi
[[ "$DRYRUN" == "1" ]] && run_parallel=0   # keep dryrun output readable/sequential

if [[ "$run_parallel" -eq 1 ]]; then
  echo "Running ${#MODELS[@]} models IN PARALLEL against Ollama (ensure OLLAMA_MAX_LOADED_MODELS>=${#MODELS[@]} and enough VRAM)."
  pids=()
  for model in "${MODELS[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_model "$model" > "scratch_param_${slug}.log" 2>&1 &
    pids+=($!)
    echo "  launched $model (pid $!) -> scratch_param_${slug}.log"
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" -eq 1 ]] && echo "WARNING: at least one model run exited non-zero (check scratch_param_*.log)"
else
  for model in "${MODELS[@]}"; do run_model "$model"; done
fi
echo "Done. Results under ${RESULTS_ROOT_FRAMES}/<model>/ and ${RESULTS_ROOT_MEDQA}/<model>/"
