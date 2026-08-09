#!/usr/bin/env bash
# Remote MedQA grid, 3-GPU parallel. Ollama runs with OLLAMA_MAX_LOADED_MODELS=3 on GPUs 0,1,2.
# Two concurrent tracks keep <=3 models resident:
#   Track A: qwen3.5:122b alone (the slow long pole, its own slot).
#   Track B: the other 5 models in pairs (<=2 concurrent) so A+B <= 3 loaded models.
# All --resume, so this picks up whatever the earlier sequential run already wrote.
set -u
cd /data/home/dvirla/parametric_search_tradeoff
export PARALLEL=1 NUM_WORKERS=2
D=scripts/run_medqa_grid_experiment.sh

echo "=== PARALLEL GRID START $(date -Is) (MAX_LOADED=3, GPUs 0,1,2) ==="

# Track A: 122b long pole
bash "$D" qwen3.5:122b &
A=$!

# Track B: remaining 5 models, at most 2 resident at a time
(
  bash "$D" qwen3.5:35b gemma4:31b
  bash "$D" qwen3.5:4b nemotron-3-nano:30b
  bash "$D" gemma4:e4b
) &
B=$!

wait "$A"; ra=$?
wait "$B"; rb=$?
echo "ALL_REMOTE_GRID_DONE $(date -Is) (trackA rc=$ra trackB rc=$rb)"
