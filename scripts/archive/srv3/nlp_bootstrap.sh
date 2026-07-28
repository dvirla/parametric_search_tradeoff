#!/usr/bin/env bash
# Idempotent recovery after nlp-srv3 reboot: ensure ollama is up with the 3-GPU config,
# then ensure the 5-model 3-way MedQA wrapper is running (--resume continues saved progress).
cd /data/home/dvirla/parametric_search_tradeoff

need=1
if curl -s --max-time 5 localhost:11434/api/version >/dev/null 2>&1; then
  env=$(cat /proc/$(pgrep -o -f '[o]llama serve')/environ 2>/dev/null | tr '\0' '\n')
  if echo "$env" | grep -q 'OLLAMA_MAX_LOADED_MODELS=3' && echo "$env" | grep -q 'CUDA_VISIBLE_DEVICES=0,1,2'; then
    need=0
  fi
fi
if [ "$need" = 1 ]; then
  pkill -f '[o]llama serve' 2>/dev/null; sleep 4
  CUDA_VISIBLE_DEVICES=0,1,2 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_NUM_PARALLEL=2 \
    setsid nohup ollama serve >~/ollama_serve.log 2>&1 </dev/null &
  for i in $(seq 1 30); do curl -s --max-time 5 localhost:11434/api/version >/dev/null 2>&1 && break; sleep 2; done
fi

if ! pgrep -f '[r]un_medqa_grid_srv3.sh' >/dev/null 2>&1; then
  setsid nohup bash run_medqa_grid_srv3.sh >>results/medqa_grid_srv3.log 2>&1 </dev/null &
  sleep 3
fi

echo "NLP_BOOTSTRAP_DONE ollama=$(curl -s --max-time 5 localhost:11434/api/version 2>/dev/null || echo DOWN)" \
     "wrapper=[$(pgrep -f '[r]un_medqa_grid_srv3.sh' | tr '\n' ' ')]"
echo "--- model progress (survived reboot on disk) ---"
for s in qwen3.5_35b gemma4_31b nemotron-3-nano_30b qwen3.5_4b gemma4_e4b; do
  f=$(for j in results/medqa_grid/$s/*.json; do python3 -c "import json;print(len(json.load(open(\"$j\"))))" 2>/dev/null; done | grep -c 500)
  echo "  $s ${f}/12"
done
echo "  gemini(done): $(for j in results/medqa_grid/gemini-3.1-pro-preview/*.json; do python3 -c "import json;print(len(json.load(open(\"$j\"))))" 2>/dev/null; done | grep -c 500)/12"
