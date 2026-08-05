#!/usr/bin/env bash
# FRAMES cue-sensitivity GRID: for each model, evaluate the baseline search agent (local BM25
# index) across the full phrasing x output-template grid PLUS the epistemic-cue conditions, all
# paired on the SAME 50 example_ids. Dependent variables (search_calls, correct) are recorded
# per problem. Grader stays gemini-3-flash-preview (Google) so grading is held constant across
# models. Resumable, with a retry loop to ride out transient grader/output-validation flakes.
#
# Conditions (all dataset=frames-cues; phrasing set by which file, template by --query_template):
#   phrasing x template grid (templates: plain / natural / elaborate / polite / query / direct):
#     verbose_*  (verbose original FRAMES phrasing)
#     terse_*    (terse benchmark-anchor phrasing)
#     direct = maximal answer-directive (no length/politeness/format) — stress-tests the
#              answer-directive hypothesis vs PLAIN (presence), NATURAL (minus length+please),
#              and POLITE (directive vs niceness).
#   epistemic cues (terse anchor + injected cue, PLAIN template):
#     epi_strong_boost / epi_strong_hedge
#   NOTE: terse_plain doubles as the neutral baseline for BOTH the grid and the epistemic cues.
#
# Usage:
#   bash scripts/run_frames_grid_experiment.sh                                  # default models
#   bash scripts/run_frames_grid_experiment.sh "nemotron-3-nano:30b"           # one model
#   CONDITIONS="terse_natural epi_strong_boost" bash scripts/run_frames_grid_experiment.sh ...
#
# Env overrides: CONDITIONS, GRADER_MODEL, GRADER_PROVIDER, NUM_WORKERS, MAX_PASSES,
#                CUES_DIR, INDEX_DIR, LOCAL_BACKEND, RESULTS_ROOT
set -euo pipefail
cd "$(dirname "$0")/.."

# nemotron-3-super is the 120B sibling of nemotron-3-nano:30b (same family). It routes to the
# ollama provider via provider_for's wildcard, like the other local models.
DEFAULT_MODELS=("nemotron-3-nano:30b" "gemma4:31b" "qwen-3.5:122b" "nemotron-3-super")
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

# PARALLEL=1 runs the given models concurrently against the same Ollama server (default ON when
# >1 model). VRAM budget (4x48GB = 192GB): pair ONE big model (qwen-3.5:122b OR nemotron-3-super)
# with one small (~30B) at a time — the two 120B-class models will NOT co-reside, so do not run them
# in the same parallel batch. Ollama must allow >=2 resident models: export OLLAMA_MAX_LOADED_MODELS=2
# (and enough VRAM). With the full DEFAULT_MODELS set, run big models in separate invocations, e.g.:
#   bash scripts/run_frames_grid_experiment.sh "nemotron-3-nano:30b" "qwen-3.5:122b"
#   bash scripts/run_frames_grid_experiment.sh "gemma4:31b" "nemotron-3-super"
PARALLEL="${PARALLEL:-auto}"

CUES_DIR="${CUES_DIR:-data/frames_cues}"
# SEARCH_BACKEND: local (default, offline BM25/dense index) or brave (paid live web API).
# Switching invalidates cross-run comparability against local-backend results (see CLAUDE.md
# pitfalls) — always route brave runs to their own RESULTS_ROOT.
SEARCH_BACKEND="${SEARCH_BACKEND:-local}"
INDEX_DIR="${INDEX_DIR:-data/frames_index}"
LOCAL_BACKEND="${LOCAL_BACKEND:-bm25}"
RESULTS_ROOT="${RESULTS_ROOT:-results/frames_cues}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PASSES="${MAX_PASSES:-4}"
# Grader held constant across models for apples-to-apples grading.
GRADER_MODEL="${GRADER_MODEL:-gemini-3-flash-preview}"
GRADER_PROVIDER="${GRADER_PROVIDER:-Google}"
NO_GRADER_FLAG=""; [[ "${NO_GRADER:-0}" == "1" ]] && NO_GRADER_FLAG="--no_grader"

# Phrasing-grid input files. Default to the paired 50-example set; scale up with SCALE=full (or
# SCALE=500) to use the full audited FRAMES set (501 rows) via *_full.jsonl. The full-scale files
# are strict supersets of the 50-example files with identical question text, so --resume reuses any
# already-saved 50-example results per condition and only runs the new rows. Override individual
# files with VERBOSE_FILE / TERSE_FILE.
SCALE="${SCALE:-50}"
if [[ "$SCALE" == "full" || "$SCALE" == "500" ]]; then
  VERBOSE_FILE="${VERBOSE_FILE:-orig_phrasing_full.jsonl}"
  TERSE_FILE="${TERSE_FILE:-neutral_matched_full.jsonl}"
else
  VERBOSE_FILE="${VERBOSE_FILE:-orig_phrasing50.jsonl}"
  TERSE_FILE="${TERSE_FILE:-neutral_matched50.jsonl}"
fi

# condition -> "<file> <template>".  All conditions are dataset=frames-cues.
declare -A COND_FILE=(
  [verbose_plain]="$VERBOSE_FILE"   [verbose_natural]="$VERBOSE_FILE"   [verbose_query]="$VERBOSE_FILE"   [verbose_elaborate]="$VERBOSE_FILE"   [verbose_polite]="$VERBOSE_FILE"   [verbose_direct]="$VERBOSE_FILE"   [verbose_confident_parametric]="$VERBOSE_FILE"   [verbose_multiturn]="$VERBOSE_FILE"   [verbose_searchmulti]="$VERBOSE_FILE"   [verbose_searchmulti2]="$VERBOSE_FILE"   [verbose_searchmulti3]="$VERBOSE_FILE"
  [terse_plain]="$TERSE_FILE"   [terse_natural]="$TERSE_FILE"   [terse_query]="$TERSE_FILE"   [terse_elaborate]="$TERSE_FILE"   [terse_polite]="$TERSE_FILE"   [terse_direct]="$TERSE_FILE"
  [epi_strong_boost]="epi_strong_boost.jsonl" [epi_strong_hedge]="epi_strong_hedge.jsonl"
)
declare -A COND_TMPL=(
  [verbose_plain]="plain"   [verbose_natural]="natural"   [verbose_query]="query"   [verbose_elaborate]="elaborate"   [verbose_polite]="polite"   [verbose_direct]="direct"   [verbose_confident_parametric]="confident_parametric"   [verbose_multiturn]="plain"   [verbose_searchmulti]="plain"   [verbose_searchmulti2]="plain"   [verbose_searchmulti3]="plain"
  [terse_plain]="plain"     [terse_natural]="natural"     [terse_query]="query"     [terse_elaborate]="elaborate"     [terse_polite]="polite"     [terse_direct]="direct"
  [epi_strong_boost]="plain" [epi_strong_hedge]="plain"
)
# Conditions that prepend a conversation-history file before the question, via
# run_qa_eval_experiment.py's --history_path. verbose_multiturn uses a single fixed chit-chat
# conversation; verbose_searchmulti{,2,3} uses a POOL of mocked-search conversations (one picked
# per example, seeded by example_id -- see EvaluationService.history_pool) with 1/2/3 concatenated
# search rounds respectively (round-count ablation). Empty/unset for all other conditions.
declare -A COND_HISTORY=(
  [verbose_multiturn]="${CUES_DIR}/chit_chat_multi_turn.json"
  [verbose_searchmulti]="${CUES_DIR}/search_multi_turn.json"
  [verbose_searchmulti2]="${CUES_DIR}/search_multi_turn_2round.json"
  [verbose_searchmulti3]="${CUES_DIR}/search_multi_turn_3round.json"
)
DEFAULT_CONDITIONS=(verbose_plain verbose_natural verbose_query verbose_elaborate verbose_polite verbose_direct verbose_confident_parametric verbose_multiturn verbose_searchmulti verbose_searchmulti2 verbose_searchmulti3 \
                    terse_plain terse_natural terse_query terse_elaborate terse_polite terse_direct \
                    epi_strong_boost epi_strong_hedge)
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-${DEFAULT_CONDITIONS[*]}}"

# --- Build the paired helper files (terse anchor + verbose original) from the epistemic id set,
#     so the grid is paired on the exact same 50 example_ids. No-op if they already exist.
build_matched() {
  local boost="${CUES_DIR}/epi_strong_boost.jsonl"
  local anchor="${CUES_DIR}/neutral_audited.jsonl"
  [[ -f "${CUES_DIR}/neutral_matched50.jsonl" && -f "${CUES_DIR}/orig_phrasing50.jsonl" ]] && return 0
  if [[ ! -f "$boost" || ! -f "$anchor" ]]; then
    echo "[build] cannot build matched files: need $boost and $anchor"; exit 1
  fi
  echo "[build] creating neutral_matched50.jsonl + orig_phrasing50.jsonl from epi_strong_boost ids"
  uv run python - "$boost" "$anchor" "$CUES_DIR" <<'PY'
import json, sys
boost, anchor, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
ids = [json.loads(l)["example_id"] for l in open(boost)]
rows = {json.loads(l)["example_id"]: json.loads(l) for l in open(anchor)}
with open(f"{outdir}/neutral_matched50.jsonl", "w") as fn, open(f"{outdir}/orig_phrasing50.jsonl", "w") as fv:
    for i in ids:
        r = rows[i]
        fn.write(json.dumps(r) + "\n")                       # terse anchor (text unchanged)
        v = dict(r); v["text"] = v["original_question"]      # verbose original phrasing
        v["cue_dimension"] = "phrasing_control"; v["cue_level"] = "verbose_original"
        fv.write(json.dumps(v) + "\n")
print("wrote", len(ids), "rows each")
PY
}

# --- Build the full-scale paired files from the ENTIRE audited FRAMES set (every example_id in
#     neutral_audited.jsonl), used when SCALE=full. Strict superset of the 50-example files with
#     identical question text, so --resume reuses already-saved 50-example results per condition.
build_full() {
  local anchor="${CUES_DIR}/neutral_audited.jsonl"
  [[ -f "${CUES_DIR}/neutral_matched_full.jsonl" && -f "${CUES_DIR}/orig_phrasing_full.jsonl" ]] && return 0
  if [[ ! -f "$anchor" ]]; then echo "[build] cannot build full files: need $anchor"; exit 1; fi
  echo "[build] creating neutral_matched_full.jsonl + orig_phrasing_full.jsonl from all $anchor ids"
  uv run python - "$anchor" "$CUES_DIR" <<'PY'
import json, sys
anchor, outdir = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(anchor)]
with open(f"{outdir}/neutral_matched_full.jsonl", "w") as fn, open(f"{outdir}/orig_phrasing_full.jsonl", "w") as fv:
    for r in rows:
        fn.write(json.dumps(r) + "\n")                       # terse anchor (text unchanged)
        v = dict(r); v["text"] = v["original_question"]      # verbose original phrasing
        v["cue_dimension"] = "phrasing_control"; v["cue_level"] = "verbose_original"
        fv.write(json.dumps(v) + "\n")
print("wrote", len(rows), "rows each")
PY
}

provider_for() {
  case "$1" in
    gemini*|*gemini*) echo "Google" ;;
    gpt-4*|gpt-3*|o1*|o3*) echo "OpenAI" ;;
    claude*) echo "Anthropic" ;;
    *) echo "ollama" ;;   # nemotron-3-nano:30b, qwen-3.5:122b, gpt-oss:*, etc.
  esac
}

# rows currently written to a results json (0 if absent)
rows_done() { uv run python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0; }

# Run the full condition sweep for ONE model. Prefixes its stdout so parallel logs stay readable.
run_model() {
  local model="$1"
  local provider; provider="$(provider_for "$model")"
  local slug_model="${model//:/_}"; slug_model="${slug_model//\//_}"
  local out_dir="${RESULTS_ROOT}/${slug_model}"
  mkdir -p "$out_dir"
  local backend_args=(--search-backend local --index-dir "$INDEX_DIR" --local-backend "$LOCAL_BACKEND")
  [[ "$SEARCH_BACKEND" == "brave" ]] && backend_args=(--search-backend brave)
  echo "[$model] START (provider=$provider, backend=$SEARCH_BACKEND) -> $out_dir"
  for cond in "${CONDITIONS_ARR[@]}"; do
    local file="${COND_FILE[$cond]:-}"; local tmpl="${COND_TMPL[$cond]:-}"
    if [[ -z "$file" || -z "$tmpl" ]]; then echo "[$model]   [skip] unknown condition: $cond"; continue; fi
    local cond_path="${CUES_DIR}/${file}"
    if [[ ! -f "$cond_path" ]]; then echo "[$model]   [skip] $cond_path not found"; continue; fi
    local target; target=$(wc -l < "$cond_path" | tr -d ' ')
    local out_json="${out_dir}/frames-cues_baseline_${model}_${cond}.json"
    local history="${COND_HISTORY[$cond]:-}"
    local history_args=(); [[ -n "$history" ]] && history_args=(--history_path "$history")
    echo "[$model]   ---- $cond (file=$file template=$tmpl target=$target) ----"
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
      uv run python scripts/run_qa_eval_experiment.py \
        --dataset frames-cues --dataset_path "$cond_path" --query_template "$tmpl" \
        "${backend_args[@]}" \
        --agent_type baseline --model_name "$model" --provider_name "$provider" \
        --grader_provider "$GRADER_PROVIDER" --grader_model "$GRADER_MODEL" \
        --run_name "$cond" --output_dir "$out_dir" "${history_args[@]}" \
        --num_workers "$NUM_WORKERS" ${NO_GRADER_FLAG} --resume || true
      local n; n=$(rows_done "$out_json")
      echo "[$model]     $cond pass $pass: $n/$target"
      [[ "$n" -ge "$target" ]] && break
    done
  done
  echo "[$model] DONE"
}

build_matched
if [[ "$SCALE" == "full" || "$SCALE" == "500" ]]; then build_full; fi

# Decide parallel vs sequential. 'auto' => parallel when more than one model is given.
run_parallel=0
if [[ "$PARALLEL" == "1" || "$PARALLEL" == "true" ]]; then run_parallel=1; fi
if [[ "$PARALLEL" == "auto" && ${#MODELS[@]} -gt 1 ]]; then run_parallel=1; fi

if [[ "$run_parallel" -eq 1 ]]; then
  echo "Running ${#MODELS[@]} models IN PARALLEL against Ollama (ensure OLLAMA_MAX_LOADED_MODELS>=${#MODELS[@]} and enough VRAM)."
  pids=()
  for model in "${MODELS[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_model "$model" > "scratch_grid_${slug}.log" 2>&1 &
    pids+=($!)
    echo "  launched $model (pid $!) -> scratch_grid_${slug}.log"
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  [[ "$fail" -eq 1 ]] && echo "WARNING: at least one model run exited non-zero (check scratch_grid_*.log)"
else
  for model in "${MODELS[@]}"; do run_model "$model"; done
fi
echo "Done. Results under ${RESULTS_ROOT}/<model>/frames-cues_baseline_<model>_<condition>.json"
