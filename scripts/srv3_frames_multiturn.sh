#!/bin/bash
# nlp-srv3 driver: FRAMES verbose_multiturn condition (chit-chat history prefix + plain original
# question), for whichever models don't compete with the larger models' Athena jobs. Each model
# gets its own GPU (cycling 0,1) + private ollama daemon/port, run concurrently, <=2 at a time.
# verbose_plain baseline already exists in results/frames_cues_full for these models -- this only
# adds the new condition, writing into the SAME per-model directory so it lines up for comparison.
# Regex-graded only (--no_grader), matching the locked convention for these cue evals.
#
# Usage: bash scripts/srv3_frames_multiturn.sh [model ...]   # default: the original qwen pair
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

DEFAULT_MODELS=(qwen3.5:4b qwen3.5:35b)
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

run_one(){
  model="$1"; gpu="$2"; port="$3"
  slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES=$gpu
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=6 nohup ollama serve > /tmp/oll_multiturn_${slug}.log 2>&1 &
  ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port ollama_pid=$ollama_pid started $(date -Is)"

  uv run python scripts/run_qa_eval_experiment.py \
    --dataset frames-cues --dataset_path data/frames_cues/orig_phrasing_full.jsonl \
    --query_template plain --history_path data/frames_cues/chit_chat_multi_turn.json \
    --search-backend local --index-dir data/frames_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name verbose_multiturn --output_dir "results/frames_cues_full/${slug}" \
    --num_workers 6 --resume

  kill $ollama_pid 2>/dev/null
  echo "=== SRV3_MULTITURN_${slug}_DONE $(date -Is) ==="
}

port=11490
for model in "${MODELS[@]}"; do
  gpu=$(( (port - 11490) % 2 ))
  slug="${model//:/_}"; slug="${slug//\//_}"
  run_one "$model" "$gpu" "$port" > "/tmp/multiturn_${slug}.log" 2>&1 &
  port=$((port+1))
done
wait
echo "ALL_SRV3_MULTITURN_DONE $(date -Is)"
