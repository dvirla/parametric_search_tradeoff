#!/bin/bash
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff
rows(){ python3 -c "import json,glob,sys;f=glob.glob(sys.argv[1]);print(len(json.load(open(f[0]))) if f else 0)" "$1" 2>/dev/null || echo 0; }
port=11470
for model in "$@"; do
  slug="${model//:/_}"; slug="${slug//\//_}"
  while true; do g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1;exit}'); [ -n "$g" ] && break; sleep 60; done
  export CUDA_VISIBLE_DEVICES=$g
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=6 nohup ollama serve > /tmp/oll_fill_${slug}.log 2>&1 &
  op=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  fill(){ ds=$1;dp=$2;idx=$3;rn=$4;od=$5;tgt=$6;gl=$7; for p in 1 2 3 4; do prev=$(rows "$gl"); uv run python scripts/run_qa_eval_experiment.py --dataset $ds $dp --query_template plain --search-backend local --index-dir $idx --local-backend bm25 --agent_type baseline --model_name $model --provider_name ollama --no_grader --run_name $rn --output_dir $od --num_workers 6 --resume; n=$(rows "$gl"); echo "[$slug $ds pass $p] $n/$tgt"; { [ "$n" -ge "$tgt" ] || [ "$n" -le "$prev" ]; } && break; done; }
  fill medqa-500 "" data/medqa_index orig_plain results/medqa_grid_rerun/$slug 500 "results/medqa_grid_rerun/$slug/*.json"
  fill frames-cues "--dataset_path data/frames_cues/orig_phrasing_full.jsonl" data/frames_index verbose_plain results/frames_cues_rerun/$slug 501 "results/frames_cues_rerun/$slug/*.json"
  kill $op 2>/dev/null; sleep 3; port=$((port+1))
done
echo "=== SRV3_FILL_QUEUE_DONE $(date -Is) ==="
