#!/bin/bash
# nlp-srv3 driver: confident_parametric condition ("you already have the relevant knowledge, no
# need to search" instruction cue) on BOTH FRAMES (verbose/original phrasing) and MedQA
# (orig/original phrasing) -- for SMALL models only (the larger ones belong on Athena, via
# scripts/athena_frames_confident_parametric_eval.job / athena_medqa_confident_parametric_eval.job).
# Each model gets its own GPU + private ollama daemon/port, run in batches of <=3 CONCURRENT
# models (never more than 3 GPUs in use at once -- if more than 3 models are passed, they run in
# sequential batches of 3). verbose_plain/orig_plain baselines already exist in
# results/frames_cues_full and results/medqa_grid for these models -- this only adds the new
# condition, writing into the SAME per-model directories so it lines up for comparison.
# Regex-graded only (--no_grader), matching the locked convention.
#
# Usage: bash scripts/srv3_confident_parametric.sh [model ...]   # default: the 4 small models
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

DEFAULT_MODELS=(qwen3.5:4b qwen3.5:35b gemma4:e4b gpt-oss:20b)
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

run_one(){
  model="$1"; gpu="$2"; port="$3"
  slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES=$gpu
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=6 nohup ollama serve > /tmp/oll_confparam_${slug}.log 2>&1 &
  ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port ollama_pid=$ollama_pid started $(date -Is)"

  uv run python scripts/run_qa_eval_experiment.py \
    --dataset frames-cues --dataset_path data/frames_cues/orig_phrasing_full.jsonl \
    --query_template confident_parametric \
    --search-backend local --index-dir data/frames_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name verbose_confident_parametric --output_dir "results/frames_cues_full/${slug}" \
    --num_workers 6 --resume

  uv run python scripts/run_qa_eval_experiment.py \
    --dataset medqa-500 --query_template confident_parametric \
    --search-backend local --index-dir data/medqa_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name orig_confident_parametric --output_dir "results/medqa_grid/${slug}" \
    --num_workers 6 --resume

  kill $ollama_pid 2>/dev/null
  echo "=== SRV3_confident_parametric_${slug}_DONE $(date -Is) ==="
}

MAX_CONCURRENT=3
port=11480
n=${#MODELS[@]}
for ((batch_start=0; batch_start<n; batch_start+=MAX_CONCURRENT)); do
  batch=("${MODELS[@]:batch_start:MAX_CONCURRENT}")
  echo "--- batch: ${batch[*]} ---"
  gpu=0
  for model in "${batch[@]}"; do
    slug="${model//:/_}"; slug="${slug//\//_}"
    run_one "$model" "$gpu" "$port" > "/tmp/confident_parametric_${slug}.log" 2>&1 &
    gpu=$((gpu+1))
    port=$((port+1))
  done
  wait
done
echo "ALL_SRV3_confident_parametric_DONE $(date -Is)"
