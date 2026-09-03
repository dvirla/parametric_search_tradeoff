#!/usr/bin/env bash
# HotpotQA cue-sensitivity grid: for each model, run the baseline search agent (local BM25 index
# over data/hotpotqa_index) across prompt-cue conditions, all PAIRED on the same example_ids --
# the fixed nested subsets built by scripts/build_hotpotqa_subset.py (hotpotqa-50/-300/-500).
# Dependent variable is `sampler_search_calls` per problem; correctness is left to an offline pass
# (NO_GRADER=1 by default, matching the FRAMES/MedQA cue grids).
#
# Cloned from scripts/run_medqa_grid_experiment.sh -- see that file for the shared mechanisms
# (provider_for, rows_done, MAX_PASSES retry loop, parallel-vs-sequential Ollama block). Unlike
# the FRAMES grid (where the cue lives in a per-condition JSONL) and the MedQA grid (where the
# phrasing selects the dataset), here the DATASET IS FIXED and the condition is purely the
# --query_template value, so condition name == template name.
#
# Output filename is derived by run_qa_eval_experiment.py as
#   <out_dir>/<dataset>_baseline_<model>_<run_name>.json
# i.e. hotpotqa-50_baseline_gemma4:31b_plain.json -- the tier is part of the name, which is why
# seed_reuse below splices a smaller tier's results into a bigger tier's file before resuming.
#
# Usage:
#   bash scripts/run_hotpotqa_cue_experiment.sh "gemma4:31b"
#   DATASET=hotpotqa-300 bash scripts/run_hotpotqa_cue_experiment.sh "gemma4:31b"
#   CONDITIONS="plain elaborate" DRYRUN=1 bash scripts/run_hotpotqa_cue_experiment.sh "gemma4:31b"
#
# Env overrides: DATASET, CONDITIONS, NUM_WORKERS, MAX_PASSES, INDEX_DIR, LOCAL_BACKEND,
#                RESULTS_ROOT, NO_GRADER, GRADER_MODEL, GRADER_PROVIDER, PARALLEL, DRYRUN
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_MODELS=("gemma4:31b")
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

DATASET="${DATASET:-hotpotqa-50}"
INDEX_DIR="${INDEX_DIR:-data/hotpotqa_index}"
LOCAL_BACKEND="${LOCAL_BACKEND:-bm25}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_cue_pilot}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PASSES="${MAX_PASSES:-4}"
NO_GRADER="${NO_GRADER:-1}"
# AGENT_TYPE=no_search removes the search tool entirely -> the PARAMETRIC probe, i.e. what the
# model answers from weights alone under each cue. Reusing THIS driver (rather than a separate
# script) is deliberate: the condition -> (template, history) map below is shared, so the
# no-search runs are prompt-for-prompt identical to the search runs and pair on example_id.
AGENT_TYPE="${AGENT_TYPE:-baseline}"
# RUNS>1 repeats every condition N times (run_1..run_N) to measure answer stability, matching the
# frames_parametric / medqa_parametric protocol. run_name becomes "<cond>_run_<r>".
RUNS="${RUNS:-1}"
DRYRUN="${DRYRUN:-0}"
PARALLEL="${PARALLEL:-auto}"
GRADER_MODEL="${GRADER_MODEL:-gemini-3-flash-preview}"
GRADER_PROVIDER="${GRADER_PROVIDER:-Google}"

# Row count of the selected tier -- read from the materialized file so it can't drift.
TIER_FILE="data/${DATASET/-/_}.jsonl"     # hotpotqa-50 -> data/hotpotqa_50.jsonl
if [[ ! -f "$TIER_FILE" ]]; then
  echo "ERROR: $TIER_FILE missing. Build it first: uv run python scripts/build_hotpotqa_subset.py" >&2
  exit 1
fi
TOTAL_ROWS=$(wc -l < "$TIER_FILE")

# Condition -> (--query_template value, optional --history_path). For the 7 pure TEMPLATE cues
# the condition name IS the template name. The two HISTORY cues (multiturn, searchmulti) keep the
# plain passthrough template and instead prepend a conversation before the question, so the cue
# lives in the dialogue context rather than the instruction -- they must NOT also carry a
# template cue, or the two manipulations are confounded.
declare -A COND_TMPL=(
  [plain]="plain" [natural]="natural" [elaborate]="elaborate" [polite]="polite"
  [direct]="direct" [confident_parametric]="confident_parametric" [query]="query"
  [multiturn]="plain" [searchmulti]="plain"
)
# Reused verbatim from the FRAMES/MedQA cue grids so the history cue is IDENTICAL across
# datasets. multiturn = one fixed chit-chat conversation; searchmulti = a POOL of mocked-search
# conversations, one picked per example seeded by example_id (1 search round -- the 2/3-round
# ablations are deliberately not run here, they were dropped as noisy).
declare -A COND_HISTORY=(
  [multiturn]="data/frames_cues/chit_chat_multi_turn.json"
  [searchmulti]="data/frames_cues/search_multi_turn.json"
)
DEFAULT_CONDITIONS=(plain natural elaborate polite direct confident_parametric query multiturn searchmulti)
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-${DEFAULT_CONDITIONS[*]}}"

provider_for() {
  case "$1" in
    gemini*|*gemini*) echo "Google" ;;
    gpt-4*|gpt-3*|o1*|o3*) echo "OpenAI" ;;
    claude*) echo "Anthropic" ;;
    *) echo "ollama" ;;
  esac
}

rows_done() { uv run python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

# --- Seed reuse across nested tiers. hotpotqa-50 is a strict PREFIX of hotpotqa-300, which is a
#     strict prefix of hotpotqa-500 (build_hotpotqa_subset.py guarantees this), so a smaller
#     tier's completed rows are all valid rows of a bigger tier -- copy them in so --resume only
#     runs the ids that are actually new. Idempotent: only copies when dest is absent.
seed_reuse() {
  local model="$1" out_dir="$2" cond src dest smaller
  # A dry run must not touch the filesystem -- this used to copy tier files during DRYRUN=1.
  [[ "$DRYRUN" == "1" ]] && return 0
  # Cross-tier splicing only makes sense for a single-run baseline sweep.
  [[ "$AGENT_TYPE" != "baseline" || "$RUNS" -ne 1 ]] && return 0
  case "$DATASET" in
    hotpotqa-300) smaller=(hotpotqa-50) ;;
    hotpotqa-500) smaller=(hotpotqa-300 hotpotqa-50) ;;
    *) return 0 ;;
  esac
  for cond in "${CONDITIONS_ARR[@]}"; do
    dest="${out_dir}/${DATASET}_baseline_${model}_${cond}.json"
    [[ -f "$dest" ]] && continue
    for s in "${smaller[@]}"; do
      src="${out_dir}/${s}_baseline_${model}_${cond}.json"
      if [[ -f "$src" ]]; then
        cp "$src" "$dest"
        echo "[seed_reuse] $src -> $dest ($(rows_done "$dest") rows)"
        break
      fi
    done
  done
}

run_model() {
  local model="$1"
  local provider; provider="$(provider_for "$model")"
  local slug_model="${model//:/_}"; slug_model="${slug_model//\//_}"
  local out_dir="${RESULTS_ROOT}/${slug_model}"
  mkdir -p "$out_dir"
  seed_reuse "$model" "$out_dir"
  echo "[$model] START (provider=$provider dataset=$DATASET rows=$TOTAL_ROWS) -> $out_dir"
  for cond in "${CONDITIONS_ARR[@]}"; do
    local tmpl="${COND_TMPL[$cond]:-}"
    if [[ -z "$tmpl" ]]; then echo "[$model]   [skip] unknown condition: $cond"; continue; fi
    local history="${COND_HISTORY[$cond]:-}"
    if [[ -n "$history" && ! -f "$history" ]]; then
      echo "[$model]   [skip] $cond: history file $history missing"; continue
    fi
    for ((r=1; r<=RUNS; r++)); do
      local run_name="$cond"
      [[ "$RUNS" -gt 1 ]] && run_name="${cond}_run_${r}"
      local out_json="${out_dir}/${DATASET}_${AGENT_TYPE}_${model}_${run_name}.json"
      echo "[$model]   ---- $run_name (agent=$AGENT_TYPE template=$tmpl history=${history:-none} target=$TOTAL_ROWS) ----"
      for ((pass=1; pass<=MAX_PASSES; pass++)); do
        local cmd=(uv run python scripts/run_qa_eval_experiment.py
          --dataset "$DATASET" --query_template "$tmpl"
          --agent_type "$AGENT_TYPE" --model_name "$model" --provider_name "$provider"
          --grader_provider "$GRADER_PROVIDER" --grader_model "$GRADER_MODEL"
          --run_name "$run_name" --output_dir "$out_dir"
          --num_workers "$NUM_WORKERS" --resume)
        # no_search never touches the search tool, so it needs no backend and no index on disk.
        if [[ "$AGENT_TYPE" != "no_search" ]]; then
          cmd+=(--search-backend local --index-dir "$INDEX_DIR" --local-backend "$LOCAL_BACKEND")
        fi
        if [[ -n "$history" ]]; then cmd+=(--history_path "$history"); fi
        if [[ "$NO_GRADER" == "1" ]]; then cmd+=(--no_grader); fi
        if [[ "$DRYRUN" == "1" ]]; then echo "[$model]     [dryrun] ${cmd[*]}"; break; fi
        "${cmd[@]}" || true
        local n; n=$(rows_done "$out_json")
        echo "[$model]     $run_name pass $pass: $n/$TOTAL_ROWS"
        [[ "$n" -ge "$TOTAL_ROWS" ]] && break
      done
    done
  done
  echo "[$model] DONE"
}

run_parallel=0
if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then run_parallel=1; fi
if [[ "$PARALLEL" == "auto" && ${#MODELS[@]} -gt 1 ]]; then run_parallel=1; fi

if [[ "$run_parallel" -eq 1 ]]; then
  echo "Running ${#MODELS[@]} models IN PARALLEL against Ollama (needs OLLAMA_MAX_LOADED_MODELS>=${#MODELS[@]})."
  pids=()
  for model in "${MODELS[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_model "$model" > "scratch_hotpotqa_cue_${slug}.log" 2>&1 &
    pids+=($!); echo "  launched $model (pid $!) -> scratch_hotpotqa_cue_${slug}.log"
  done
  fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" -eq 1 ]] && echo "WARNING: a model run exited non-zero (check scratch_hotpotqa_cue_*.log)"
else
  for model in "${MODELS[@]}"; do run_model "$model"; done
fi
echo "Done. Results under ${RESULTS_ROOT}/<model>/${DATASET}_${AGENT_TYPE}_<model>_<run_name>.json"
