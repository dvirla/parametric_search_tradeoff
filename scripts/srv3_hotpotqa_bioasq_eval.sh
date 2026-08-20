#!/bin/bash
# nlp-srv3 driver: first-pass baseline eval on the new HotpotQA + BioASQ local indexes
# (data/hotpotqa_index, data/bioasq_index -- see scripts/build_hotpotqa_index.py /
# build_bioasq_index.py), for the small/mid-size open-source models. Larger models
# (qwen3.5:122b, gpt-oss:120b, nemotron-cascade-2:30b) run on Athena instead, via
# scripts/athena_hotpotqa_bioasq_eval.job.
#
# --num_examples 200 per dataset (matches the sample size already used to validate both
# indexes). --agent_type baseline. --no_grader (regex path skipped, grading is a separate
# offline pass -- see scripts/run_parametric_probe_experiment.sh precedent) -- correctness is
# NOT computed at launch time; this run is for accuracy/search-behavior data collection.
#
# GPU budget: uses GPUs 0, 2, 3 ONLY -- GPU 1 is occupied by a different session's ollama
# process (91GB, confirmed via nvidia-smi at launch time). Never assume GPU 1 is free; re-check
# with `nvidia-smi` before editing GPU_LIST below. Each model gets its own GPU + private ollama
# daemon/port, run concurrently (<=3 GPUs in use at once, matching the 3-GPU budget for this
# user on srv3).
#
# Usage: bash scripts/srv3_hotpotqa_bioasq_eval.sh [model ...]   # default: the 3 srv3 models
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

DEFAULT_MODELS=(gemma4:31b nemotron-3-nano:30b gpt-oss:20b)
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=("${DEFAULT_MODELS[@]}"); fi

# Free GPUs at launch time (0, 2, 3 -- GPU 1 in use by another session). One entry per model in
# MODELS, in order.
GPU_LIST=(0 2 3)

NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
RUN_NAME="${RUN_NAME:-plain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hotpotqa_bioasq_baseline}"

run_one(){
  model="$1"; gpu="$2"; port="$3"; workers="$4"
  slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES=$gpu
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=$workers nohup ollama serve > /tmp/oll_hbq_${slug}.log 2>&1 &
  ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port workers=$workers ollama_pid=$ollama_pid started $(date -Is)"
  echo "=== ensuring model is pulled ==="; ollama show "$model" >/dev/null 2>&1 || ollama pull "$model"

  uv run python scripts/run_qa_eval_experiment.py \
    --dataset hotpotqa --num_examples "$NUM_EXAMPLES" \
    --search-backend local --index-dir data/hotpotqa_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name "$RUN_NAME" --output_dir "${OUTPUT_ROOT}/${slug}" \
    --num_workers "$workers" --resume

  uv run python scripts/run_qa_eval_experiment.py \
    --dataset bioasq --num_examples "$NUM_EXAMPLES" \
    --search-backend local --index-dir data/bioasq_index --local-backend bm25 \
    --agent_type baseline --model_name "$model" --provider_name ollama \
    --no_grader --run_name "$RUN_NAME" --output_dir "${OUTPUT_ROOT}/${slug}" \
    --num_workers "$workers" --resume

  kill $ollama_pid 2>/dev/null
  echo "=== SRV3_hotpotqa_bioasq_${slug}_DONE $(date -Is) ==="
}

port=11560
i=0
for model in "${MODELS[@]}"; do
  gpu="${GPU_LIST[$i]}"
  slug="${model//:/_}"; slug="${slug//\//_}"
  # nemotron-3-nano is the slow/heavy model in this trio (see project_parametric_probe_launch
  # memory: ~74s/it) -- lower concurrency to avoid request-queue timeouts under --no_grader
  # (no grader-call throttle). gemma4:31b/gpt-oss:20b were fine at higher concurrency.
  case "$model" in
    nemotron-3-nano:30b) workers=2 ;;
    *) workers=4 ;;
  esac
  run_one "$model" "$gpu" "$port" "$workers" > "/tmp/hotpotqa_bioasq_${slug}.log" 2>&1 &
  i=$((i+1))
  port=$((port+1))
done
wait
echo "ALL_SRV3_hotpotqa_bioasq_DONE $(date -Is)"
