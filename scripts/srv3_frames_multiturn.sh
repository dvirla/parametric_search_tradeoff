#!/bin/bash
# nlp-srv3 driver: FRAMES history-prefix condition (verbose_multiturn or verbose_searchmulti +
# plain original question) -- for SMALL models only (the larger ones belong on Athena). Each
# model gets its own GPU + private ollama daemon/port, run in batches of <=3 CONCURRENT models
# (never more than 3 GPUs in use at once -- if more than 3 models are passed, they run in
# sequential batches of 3). verbose_plain baseline already exists in results/frames_cues_full for
# these models -- this only adds the new condition, writing into the SAME per-model directory so
# it lines up for comparison. Regex-graded only (--no_grader), matching the locked convention.
#
# Usage: bash scripts/srv3_frames_multiturn.sh [model ...]   # default: the original qwen pair
# Env overrides: RUN_NAME (default verbose_multiturn), HISTORY_PATH (default
#                data/frames_cues/chit_chat_multi_turn.json; use search_multi_turn.json for the
#                mocked-search condition -> RUN_NAME=verbose_searchmulti)
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

RUN_NAME="${RUN_NAME:-verbose_multiturn}"
HISTORY_PATH="${HISTORY_PATH:-data/frames_cues/chit_chat_multi_turn.json}"
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
    --query_template plain --history_path "$HISTORY_PATH" \
    --search-backend local --index-dir data/frames_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name "$RUN_NAME" --output_dir "results/frames_cues_full/${slug}" \
    --num_workers 6 --resume

  kill $ollama_pid 2>/dev/null
  echo "=== SRV3_${RUN_NAME}_${slug}_DONE $(date -Is) ==="
}

MAX_CONCURRENT=3
port=11490
n=${#MODELS[@]}
for ((batch_start=0; batch_start<n; batch_start+=MAX_CONCURRENT)); do
  batch=("${MODELS[@]:batch_start:MAX_CONCURRENT}")
  echo "--- batch: ${batch[*]} ---"
  gpu=0
  for model in "${batch[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_one "$model" "$gpu" "$port" > "/tmp/${RUN_NAME}_${slug}.log" 2>&1 &
    gpu=$((gpu+1))
    port=$((port+1))
  done
  wait
done
echo "ALL_SRV3_${RUN_NAME}_DONE $(date -Is)"
