#!/usr/bin/env bash
# Progress of the HotpotQA cue grid across BOTH machines. Read-only; safe to run any time.
#   bash scripts/hotpotqa_cue_grid_status.sh
# Env: DATASET (hotpotqa-300), RESULTS_ROOT (results/hotpotqa_cue_grid), TARGET (300)
DATASET="${DATASET:-hotpotqa-300}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_cue_grid}"
TARGET="${TARGET:-300}"

# Counts rows in every condition file under a results root, one line per model.
REMOTE_SNIPPET='
python3 - <<PY
import glob, json, os, collections
root = "'"$RESULTS_ROOT"'"; target = '"$TARGET"'
per = collections.defaultdict(dict)
for p in glob.glob(os.path.join(root, "*", "'"$DATASET"'_baseline_*_*.json")):
    model = os.path.basename(os.path.dirname(p))
    cond = os.path.basename(p)[:-5].split("_")[-1]
    for k in ("confident_parametric","searchmulti","multiturn","elaborate","natural","polite","direct","query","plain"):
        if os.path.basename(p)[:-5].endswith("_"+k): cond = k; break
    try: n = len(json.load(open(p)))
    except Exception: n = -1
    per[model][cond] = n
if not per: print("  (no result files yet)")
for model in sorted(per):
    done = sum(1 for v in per[model].values() if v >= target)
    tot  = sum(max(v,0) for v in per[model].values())
    detail = " ".join(f"{c}:{n}" for c, n in sorted(per[model].items()))
    print(f"  {model:24s} {done}/9 conds complete, {tot} rows   [{detail}]")
PY
'

echo "=== ATHENA (SLURM) ==="
ssh -o ConnectTimeout=20 athen "squeue -u dvirla -h -o %T | sort | uniq -c; cd ~/parametric_search_tradeoff && $REMOTE_SNIPPET" 2>/dev/null | grep -v '^\*\*\|^\.\.\.'
echo
echo "=== NLP-SRV3 ==="
ssh -o ConnectTimeout=20 nlp-srv3 "cd /data/home/dvirla/parametric_search_tradeoff_hpqcue && $REMOTE_SNIPPET; echo '  driver alive:' \$(pgrep -f srv3_hotpotqa_cue_grid.sh | wc -l) 'proc(s)'" 2>/dev/null | grep -v '^\*\*\|^\.\.\.'
