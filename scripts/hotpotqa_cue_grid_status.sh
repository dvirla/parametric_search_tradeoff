#!/usr/bin/env bash
# Progress of the HotpotQA cue grid across BOTH machines. Read-only; safe to run any time.
#   bash scripts/hotpotqa_cue_grid_status.sh
# Env: DATASET (hotpotqa-300), RESULTS_ROOT (results/hotpotqa_cue_grid), TARGET (300)
DATASET="${DATASET:-hotpotqa-300}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_cue_grid}"
TARGET="${TARGET:-300}"
# baseline = the search grid; no_search = the parametric probe (run names carry _run_<r>).
#   AGENT_TYPE=no_search RESULTS_ROOT=results/hotpotqa_parametric CONDS=20 bash scripts/hotpotqa_cue_grid_status.sh
AGENT_TYPE="${AGENT_TYPE:-baseline}"
CONDS="${CONDS:-9}"

# Counts rows in every condition file under a results root, one line per model.
REMOTE_SNIPPET='
python3 - <<PY
import glob, json, os, collections
root = "'"$RESULTS_ROOT"'"; target = '"$TARGET"'; conds = '"$CONDS"'
agent = "'"$AGENT_TYPE"'"
per = collections.defaultdict(dict)
for p in glob.glob(os.path.join(root, "*", "'"$DATASET"'_" + agent + "_*_*.json")):
    model = os.path.basename(os.path.dirname(p))
    stem = os.path.basename(p)[:-5]
    # run name is everything after "<dataset>_<agent>_<model>_" -- model may contain "_", so match
    # the known cue vocabulary (optionally suffixed with _run_<r>) from the right.
    cond = stem.split("_")[-1]
    for k in ("confident_parametric","searchmulti","multiturn","elaborate","natural","polite","direct","query","plain"):
        if stem.endswith("_"+k): cond = k; break
        import re as _re
        m = _re.search(r"_(" + k + r")_run_(\d+)$", stem)
        if m: cond = k + "_run_" + m.group(2); break
    try: n = len(json.load(open(p)))
    except Exception: n = -1
    per[model][cond] = n
if not per: print("  (no result files yet)")
for model in sorted(per):
    done = sum(1 for v in per[model].values() if v >= target)
    tot  = sum(max(v,0) for v in per[model].values())
    detail = " ".join(f"{c}:{n}" for c, n in sorted(per[model].items()))
    print(f"  {model:24s} {done}/{conds} complete, {tot} rows   [{detail}]")
PY
'

echo "=== ATHENA (SLURM) ==="
ssh -o ConnectTimeout=20 athen "squeue -u dvirla -h -o %T | sort | uniq -c; cd ~/parametric_search_tradeoff && $REMOTE_SNIPPET" 2>/dev/null | grep -v '^\*\*\|^\.\.\.'
echo
echo "=== NLP-SRV3 ==="
ssh -o ConnectTimeout=20 nlp-srv3 "cd /data/home/dvirla/parametric_search_tradeoff_hpqcue && $REMOTE_SNIPPET; echo '  driver alive:' \$(pgrep -f srv3_hotpotqa_cue_grid.sh | wc -l) 'proc(s)'" 2>/dev/null | grep -v '^\*\*\|^\.\.\.'
