#!/bin/bash
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff
run_one(){
  model="$1"; gpu="$2"; port="$3"
  slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES=$gpu
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=6 nohup ollama serve > /tmp/oll_rr_${slug}.log 2>&1 &
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port started $(date -Is)"
  uv run python scripts/run_qa_eval_experiment.py --dataset medqa-500 --query_template plain \
    --search-backend local --index-dir data/medqa_index --local-backend bm25 \
    --agent_type baseline --model_name $model --provider_name ollama \
    --no_grader --run_name orig_plain --output_dir results/medqa_grid_rerun/${slug} \
    --num_workers 6 --resume
  uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues --dataset_path data/frames_cues/orig_phrasing_full.jsonl --query_template plain \
    --search-backend local --index-dir data/frames_index --local-backend bm25 \
    --agent_type baseline --model_name $model --provider_name ollama \
    --no_grader --run_name verbose_plain --output_dir results/frames_cues_rerun/${slug} \
    --num_workers 6 --resume
  echo "=== SRV3_RR_${slug}_DONE $(date -Is) ==="
}
port=11460
for model in gemma4:31b qwen3.5:122b; do
  echo "[queue] waiting for a free GPU slot (busy<3) for $model ..."
  while true; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
    if [ "$busy" -lt 3 ]; then
      free_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
      [ -n "$free_gpu" ] && break
    fi
    sleep 120
  done
  echo "[queue] launching $model on GPU $free_gpu"
  run_one $model $free_gpu $port > /tmp/rr_${model//:/_}.log 2>&1 &
  port=$((port+1))
  sleep 120
done
wait
echo "ALL_SRV3_PHASE2_DONE $(date -Is)"
