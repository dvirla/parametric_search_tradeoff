#!/usr/bin/env bash
# MedQA cue-sensitivity GRID: for each model, evaluate the baseline search agent (local BM25
# index over data/medqa_index) across the full phrasing x output-template grid, all paired on
# the SAME 500 example_ids (medqa-500). Dependent variables (search_calls, correct) are recorded
# per problem. Grader stays gemini-3-flash-preview (Google) so grading is held constant across
# models. Resumable, with a retry loop to ride out transient grader/output-validation flakes.
# Cloned from scripts/run_frames_grid_experiment.sh -- see that file for the shared mechanisms
# (provider_for, rows_done, MAX_PASSES retry loop, parallel-vs-sequential Ollama block).
#
# Conditions (12 = {orig,terse} x {plain,natural,elaborate,polite,direct,query}):
#   Unlike the FRAMES grid (where phrasing = which JSONL file, dataset held constant at
#   frames-cues), here PHRASING SELECTS THE DATASET NAME:
#     orig_*   -> --dataset medqa-500   (verbose/original clinical-vignette phrasing)
#     terse_*  -> --dataset medqa-terse (terse paraphrase, fact-preserving)
#   Template (plain/natural/elaborate/polite/direct/query) is passed via --query_template.
#   No --dataset_path override is used -- medqa-500/medqa-terse are each a full self-contained
#   500-row dataset (registered in src/services/qa_eval.py::_load_dataset, in _self_path_datasets).
#
# IMPORTANT naming note (falls out of run_qa_eval_experiment.py, which this script does NOT
# edit): the eval script derives its output filename as
#   f"{dataset}_{agent_type}_{model_name}_{run_name}.json"
# i.e. purely from --dataset (not a fixed "medqa" literal). Because phrasing here selects the
# dataset name, the REAL files this driver produces/resumes are:
#   <out_dir>/medqa-500_baseline_<model>_<cond>.json    (orig_* conditions)
#   <out_dir>/medqa-terse_baseline_<model>_<cond>.json  (terse_* conditions)
# NOT "medqa_baseline_<model>_<cond>.json". The seed-reuse step below accounts for this: it
# splices the completed medqa_cue_200 results (which WERE produced with --dataset medqa) into
# the medqa-500-prefixed filename that orig_plain/orig_elaborate will actually resume from.
#
# Usage:
#   bash scripts/run_medqa_grid_experiment.sh                                  # default models
#   bash scripts/run_medqa_grid_experiment.sh "nemotron-3-nano:30b"           # one model
#   CONDITIONS="terse_plain orig_plain" bash scripts/run_medqa_grid_experiment.sh "gemma4:31b"
#   DRYRUN=1 bash scripts/run_medqa_grid_experiment.sh "gemma4:31b"           # print commands only
#
# Env overrides: CONDITIONS, GRADER_MODEL, GRADER_PROVIDER, NUM_WORKERS, MAX_PASSES,
#                INDEX_DIR, LOCAL_BACKEND, RESULTS_ROOT, SEED_DIR, DRYRUN
set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_MODELS=("gemini-3.1-pro-preview" "qwen3.5:122b" "qwen3.5:35b" "qwen3.5:4b" \
                "gemma4:31b" "gemma4:12b" "gemma4:e4b" "nemotron-3-nano:30b")
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

# PARALLEL=1 runs the given models concurrently against the same Ollama server (default ON when
# >1 model). VRAM budget (4x48GB = 192GB): the 122b-class model (qwen3.5:122b) does NOT co-reside
# with another large model -- pair it with at most one ~30B-or-smaller model at a time. Ollama
# must allow >=2 resident models: export OLLAMA_MAX_LOADED_MODELS=2 (and enough VRAM). With the
# full DEFAULT_MODELS set, batch the 122b model in its own invocation, e.g.:
#   bash scripts/run_medqa_grid_experiment.sh "qwen3.5:122b"
#   bash scripts/run_medqa_grid_experiment.sh "qwen3.5:35b" "qwen3.5:4b" "gemma4:31b" "gemma4:12b" "gemma4:e4b" "nemotron-3-nano:30b"
#   bash scripts/run_medqa_grid_experiment.sh "gemini-3.1-pro-preview"   # Google API, no VRAM concern
PARALLEL="${PARALLEL:-auto}"

INDEX_DIR="${INDEX_DIR:-data/medqa_index}"
LOCAL_BACKEND="${LOCAL_BACKEND:-bm25}"
RESULTS_ROOT="${RESULTS_ROOT:-results/medqa_grid}"
SEED_DIR="${SEED_DIR:-results/medqa_cue_200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PASSES="${MAX_PASSES:-4}"
DRYRUN="${DRYRUN:-0}"
# Grader held constant across models for apples-to-apples grading.
GRADER_MODEL="${GRADER_MODEL:-gemini-3-flash-preview}"
GRADER_PROVIDER="${GRADER_PROVIDER:-Google}"

# Full row count of medqa-500 / medqa-terse. Both are fixed 500-row datasets, so no
# --num_examples is passed -- the whole file is always run.
TOTAL_ROWS=500

# condition -> "<dataset> <template>". Phrasing (orig/terse) selects the dataset; the suffix
# after the phrasing prefix is the --query_template value.
declare -A COND_DATASET=(
  [orig_plain]="medqa-500"   [orig_natural]="medqa-500"   [orig_elaborate]="medqa-500"   [orig_polite]="medqa-500"   [orig_direct]="medqa-500"   [orig_confident_parametric]="medqa-500"   [orig_query]="medqa-500"   [orig_multiturn]="medqa-500"   [orig_searchmulti]="medqa-500"   [orig_searchmulti2]="medqa-500"   [orig_searchmulti3]="medqa-500"
  [terse_plain]="medqa-terse" [terse_natural]="medqa-terse" [terse_elaborate]="medqa-terse" [terse_polite]="medqa-terse" [terse_direct]="medqa-terse" [terse_query]="medqa-terse"
)
declare -A COND_TMPL=(
  [orig_plain]="plain"   [orig_natural]="natural"   [orig_elaborate]="elaborate"   [orig_polite]="polite"   [orig_direct]="direct"   [orig_confident_parametric]="confident_parametric"   [orig_query]="query"   [orig_multiturn]="plain"   [orig_searchmulti]="plain"   [orig_searchmulti2]="plain"   [orig_searchmulti3]="plain"
  [terse_plain]="plain"  [terse_natural]="natural"  [terse_elaborate]="elaborate"  [terse_polite]="polite"  [terse_direct]="direct"  [terse_query]="query"
)
# Conditions that prepend a conversation-history file before the question, via
# run_qa_eval_experiment.py's --history_path. orig_multiturn uses a single fixed chit-chat
# conversation; orig_searchmulti{,2,3} uses a POOL of mocked-search conversations (one picked per
# example, seeded by example_id) with 1/2/3 concatenated search rounds respectively (round-count
# ablation). Empty/unset for all other conditions.
declare -A COND_HISTORY=(
  [orig_multiturn]="data/frames_cues/chit_chat_multi_turn.json"
  [orig_searchmulti]="data/frames_cues/search_multi_turn.json"
  [orig_searchmulti2]="data/frames_cues/search_multi_turn_2round.json"
  [orig_searchmulti3]="data/frames_cues/search_multi_turn_3round.json"
)
DEFAULT_CONDITIONS=(orig_plain orig_natural orig_elaborate orig_polite orig_direct orig_confident_parametric orig_query orig_multiturn orig_searchmulti orig_searchmulti2 orig_searchmulti3 \
                    terse_plain terse_natural terse_elaborate terse_polite terse_direct terse_query)
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-${DEFAULT_CONDITIONS[*]}}"

provider_for() {
  case "$1" in
    gemini*|*gemini*) echo "Google" ;;
    gpt-4*|gpt-3*|o1*|o3*) echo "OpenAI" ;;
    claude*) echo "Anthropic" ;;
    *) echo "ollama" ;;   # qwen3.5:*, gemma4:*, nemotron-3-nano:30b, etc.
  esac
}

# rows currently written to a results json (0 if absent)
rows_done() { uv run python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

# --- Seed reuse: splice the already-completed medqa_cue_200 plain/elaborate results (200 of the
#     500 ids, produced with --dataset medqa) into the orig_plain / orig_elaborate cells of the
#     grid for the 3 original models, so --resume only has to run the remaining ~300 ids. Only
#     copies if the source exists and the dest does not (idempotent, safe to call every invocation).
seed_reuse() {
  local seed_models=("gemini-3.1-pro-preview" "qwen3.5:122b" "gemma4:31b")
  local m src dest slug out_dir cond ds
  for m in "${seed_models[@]}"; do
    slug="${m//:/_}"; slug="${slug//\//_}"
    out_dir="${RESULTS_ROOT}/${slug}"
    mkdir -p "$out_dir"
    for cond in orig_plain orig_elaborate; do
      local suffix="${cond#orig_}"   # plain | elaborate
      src="${SEED_DIR}/medqa_baseline_${m}_${suffix}.json"
      ds="${COND_DATASET[$cond]}"
      dest="${out_dir}/${ds}_baseline_${m}_${cond}.json"
      if [[ -f "$src" && ! -f "$dest" ]]; then
        cp "$src" "$dest"
        echo "[seed_reuse] copied $src -> $dest ($(rows_done "$dest") rows)"
      fi
    done
  done
}

# Run the full condition sweep for ONE model. Prefixes its stdout so parallel logs stay readable.
run_model() {
  local model="$1"
  local provider; provider="$(provider_for "$model")"
  local slug_model="${model//:/_}"; slug_model="${slug_model//\//_}"
  local out_dir="${RESULTS_ROOT}/${slug_model}"
  mkdir -p "$out_dir"
  echo "[$model] START (provider=$provider) -> $out_dir"
  for cond in "${CONDITIONS_ARR[@]}"; do
    local ds="${COND_DATASET[$cond]:-}"; local tmpl="${COND_TMPL[$cond]:-}"
    if [[ -z "$ds" || -z "$tmpl" ]]; then echo "[$model]   [skip] unknown condition: $cond"; continue; fi
    local out_json="${out_dir}/${ds}_baseline_${model}_${cond}.json"
    local history="${COND_HISTORY[$cond]:-}"
    echo "[$model]   ---- $cond (dataset=$ds template=$tmpl target=$TOTAL_ROWS) ----"
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
      local cmd=(uv run python scripts/run_qa_eval_experiment.py
        --dataset "$ds" --query_template "$tmpl"
        --search-backend local --index-dir "$INDEX_DIR" --local-backend "$LOCAL_BACKEND"
        --agent_type baseline --model_name "$model" --provider_name "$provider"
        --grader_provider "$GRADER_PROVIDER" --grader_model "$GRADER_MODEL"
        --run_name "$cond" --output_dir "$out_dir"
        --num_workers "$NUM_WORKERS" --resume)
      if [[ -n "$history" ]]; then cmd+=(--history_path "$history"); fi
      if [[ "${NO_GRADER:-0}" == "1" ]]; then cmd+=(--no_grader); fi
      if [[ "$DRYRUN" == "1" ]]; then
        echo "[$model]     [dryrun] ${cmd[*]}"
        break
      fi
      "${cmd[@]}" || true
      local n; n=$(rows_done "$out_json")
      echo "[$model]     $cond pass $pass: $n/$TOTAL_ROWS"
      [[ "$n" -ge "$TOTAL_ROWS" ]] && break
    done
  done
  echo "[$model] DONE"
}

seed_reuse

# Decide parallel vs sequential. 'auto' => parallel when more than one model is given.
run_parallel=0
if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then run_parallel=1; fi
if [[ "$PARALLEL" == "auto" && ${#MODELS[@]} -gt 1 ]]; then run_parallel=1; fi

if [[ "$run_parallel" -eq 1 ]]; then
  echo "Running ${#MODELS[@]} models IN PARALLEL against Ollama (ensure OLLAMA_MAX_LOADED_MODELS>=${#MODELS[@]} and enough VRAM)."
  pids=()
  for model in "${MODELS[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_model "$model" > "scratch_medqa_grid_${slug}.log" 2>&1 &
    pids+=($!)
    echo "  launched $model (pid $!) -> scratch_medqa_grid_${slug}.log"
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" -eq 1 ]] && echo "WARNING: at least one model run exited non-zero (check scratch_medqa_grid_*.log)"
else
  for model in "${MODELS[@]}"; do run_model "$model"; done
fi
echo "Done. Results under ${RESULTS_ROOT}/<model>/<dataset>_baseline_<model>_<condition>.json"
