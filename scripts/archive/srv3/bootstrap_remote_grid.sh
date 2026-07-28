#!/usr/bin/env bash
# Idempotent: ensure ollama is up with the 3-GPU config, then ensure the parallel grid wrapper
# is running. Safe to re-run if an ssh drops mid-way.
cd /data/home/dvirla/parametric_search_tradeoff

need_restart=1
if curl -s --max-time 5 localhost:11434/api/version >/dev/null 2>&1; then
  env=$(cat /proc/$(pgrep -o -f '[o]llama serve')/environ 2>/dev/null | tr '\0' '\n')
  if echo "$env" | grep -q 'OLLAMA_MAX_LOADED_MODELS=3' && echo "$env" | grep -q 'CUDA_VISIBLE_DEVICES=0,1,2'; then
    need_restart=0
  fi
fi

if [ "$need_restart" = 1 ]; then
  pkill -f '[o]llama serve' 2>/dev/null; sleep 5
  CUDA_VISIBLE_DEVICES=0,1,2 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL=2 \
    setsid nohup ollama serve >~/ollama_serve.log 2>&1 </dev/null &
  for i in $(seq 1 30); do
    curl -s --max-time 5 localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 2
  done
fi

if ! pgrep -f '[r]un_medqa_grid_parallel.sh' >/dev/null 2>&1; then
  setsid nohup bash run_medqa_grid_parallel.sh >results/medqa_grid_remote.log 2>&1 </dev/null &
  sleep 3
fi

echo "BOOTSTRAP_DONE ollama=$(curl -s --max-time 5 localhost:11434/api/version 2>/dev/null || echo DOWN)" \
     "env=[$(cat /proc/$(pgrep -o -f '[o]llama serve')/environ 2>/dev/null | tr '\0' '\n' | grep -E 'OLLAMA_MAX_LOADED|CUDA_VISIBLE|NUM_PARALLEL' | tr '\n' ' ')]" \
     "wrapper=[$(pgrep -f '[r]un_medqa_grid_parallel.sh' | tr '\n' ' ')]"
