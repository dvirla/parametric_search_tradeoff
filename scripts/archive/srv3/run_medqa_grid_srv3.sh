#!/usr/bin/env bash
# nlp-srv3 3-way (v3): gemma4:31b + nemotron-3-nano:30b done. Run all 3 remaining models
# concurrently (3 slots): qwen3.5:35b + qwen3.5:4b + gemma4:e4b. All --resume.
set -u
cd /data/home/dvirla/parametric_search_tradeoff
export PARALLEL=1 NUM_WORKERS=2
D=scripts/run_medqa_grid_experiment.sh
echo "=== SRV3 3WAY v3 START $(date -Is) ==="
bash "$D" qwen3.5:35b qwen3.5:4b gemma4:e4b
echo "ALL_REMOTE_GRID_DONE $(date -Is)"
