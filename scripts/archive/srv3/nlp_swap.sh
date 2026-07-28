#!/usr/bin/env bash
# Clean swap to the v2 wrapper: kill any wrapper + orphaned medqa evals, then launch exactly one
# fresh wrapper (setsid, survives ssh drop). run_medqa_grid_srv3.sh already holds the v2 grouping.
cd /data/home/dvirla/parametric_search_tradeoff
pkill -9 -f '[r]un_medqa_grid_srv3.sh' 2>/dev/null
pkill -9 -f '[r]un_medqa_grid_experiment.sh' 2>/dev/null
pkill -9 -f '[r]un_qa_eval_experiment.py --dataset medqa' 2>/dev/null
sleep 5
setsid nohup bash run_medqa_grid_srv3.sh >>results/medqa_grid_srv3.log 2>&1 </dev/null &
sleep 3
echo "SWAP_DONE wrapper=[$(pgrep -f '[r]un_medqa_grid_srv3.sh' | tr '\n' ' ')]"
