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
INDEX_DIR="${INDEX_DIR:-data/frames_index}"
LOCAL_BACKEND="${LOCAL_BACKEND:-bm25}"
RESULTS_ROOT="${RESULTS_ROOT:-results/frames_cues}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_PASSES="${MAX_PASSES:-4}"
# Grader held constant across models for apples-to-apples grading.
GRADER_MODEL="${GRADER_MODEL:-gemini-3-flash-preview}"
GRADER_PROVIDER="${GRADER_PROVIDER:-Google}"

# condition -> "<file> <template>".  All conditions are dataset=frames-cues.
declare -A COND_FILE=(
  [verbose_plain]="orig_phrasing50.jsonl"   [verbose_natural]="orig_phrasing50.jsonl"   [verbose_query]="orig_phrasing50.jsonl"   [verbose_elaborate]="orig_phrasing50.jsonl"   [verbose_polite]="orig_phrasing50.jsonl"   [verbose_direct]="orig_phrasing50.jsonl"
  [terse_plain]="neutral_matched50.jsonl"   [terse_natural]="neutral_matched50.jsonl"   [terse_query]="neutral_matched50.jsonl"   [terse_elaborate]="neutral_matched50.jsonl"   [terse_polite]="neutral_matched50.jsonl"   [terse_direct]="neutral_matched50.jsonl"
  [epi_strong_boost]="epi_strong_boost.jsonl" [epi_strong_hedge]="epi_strong_hedge.jsonl"
)
declare -A COND_TMPL=(
  [verbose_plain]="plain"   [verbose_natural]="natural"   [verbose_query]="query"   [verbose_elaborate]="elaborate"   [verbose_polite]="polite"   [verbose_direct]="direct"
  [terse_plain]="plain"     [terse_natural]="natural"     [terse_query]="query"     [terse_elaborate]="elaborate"     [terse_polite]="polite"     [terse_direct]="direct"
  [epi_strong_boost]="plain" [epi_strong_hedge]="plain"
)
DEFAULT_CONDITIONS=(verbose_plain verbose_natural verbose_query verbose_elaborate verbose_polite verbose_direct \
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
  echo "[$model] START (provider=$provider) -> $out_dir"
  for cond in "${CONDITIONS_ARR[@]}"; do
    local file="${COND_FILE[$cond]:-}"; local tmpl="${COND_TMPL[$cond]:-}"
    if [[ -z "$file" || -z "$tmpl" ]]; then echo "[$model]   [skip] unknown condition: $cond"; continue; fi
    local cond_path="${CUES_DIR}/${file}"
    if [[ ! -f "$cond_path" ]]; then echo "[$model]   [skip] $cond_path not found"; continue; fi
    local target; target=$(wc -l < "$cond_path" | tr -d ' ')
    local out_json="${out_dir}/frames-cues_baseline_${model}_${cond}.json"
    echo "[$model]   ---- $cond (file=$file template=$tmpl target=$target) ----"
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
      uv run python scripts/run_qa_eval_experiment.py \
        --dataset frames-cues --dataset_path "$cond_path" --query_template "$tmpl" \
        --search-backend local --index-dir "$INDEX_DIR" --local-backend "$LOCAL_BACKEND" \
        --agent_type baseline --model_name "$model" --provider_name "$provider" \
        --grader_provider "$GRADER_PROVIDER" --grader_model "$GRADER_MODEL" \
        --run_name "$cond" --output_dir "$out_dir" \
        --num_workers "$NUM_WORKERS" --resume || true
      local n; n=$(rows_done "$out_json")
      echo "[$model]     $cond pass $pass: $n/$target"
      [[ "$n" -ge "$target" ]] && break
    done
  done
  echo "[$model] DONE"
}

build_matched

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
