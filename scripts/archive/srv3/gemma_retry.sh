#!/bin/bash
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff
g=0; port=11480
export CUDA_VISIBLE_DEVICES=$g
OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=32768 nohup ollama serve > /tmp/oll_gemma_retry.log 2>&1 &
op=$!
for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
# LOAD-CHECK GUARD: only proceed if the model actually generates (no retry-storm on load failure)
warm=$(curl -s http://127.0.0.1:$port/api/generate -d '{"model":"gemma4:31b","prompt":"hi","stream":false,"options":{"num_predict":5}}' 2>/dev/null)
if ! echo "$warm" | grep -q '"response"'; then
  echo "GEMMA4:31B LOAD FAILED - ABORTING before eval (no Logfire flood): $(echo "$warm" | head -c 220)"
  kill $op 2>/dev/null; exit 1
fi
echo "gemma4:31b LOADED OK on gpu $g $(date -Is)"
export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
rows(){ python3 -c "import json,glob,sys;f=glob.glob(sys.argv[1]);print(len(json.load(open(f[0]))) if f else 0)" "$1" 2>/dev/null || echo 0; }
fill(){ ds=$1;dp=$2;idx=$3;rn=$4;od=$5;tgt=$6;gl=$7; for p in 1 2 3; do prev=$(rows "$gl"); uv run python scripts/run_qa_eval_experiment.py --dataset $ds $dp --query_template plain --search-backend local --index-dir $idx --local-backend bm25 --agent_type baseline --model_name gemma4:31b --provider_name ollama --no_grader --run_name $rn --output_dir $od --num_workers 4 --resume; n=$(rows "$gl"); echo "[gemma $ds pass $p] $n/$tgt"; { [ "$n" -ge "$tgt" ] || [ "$n" -le "$prev" ]; } && break; done; }
fill medqa-500 "" data/medqa_index orig_plain results/medqa_grid_rerun/gemma4_31b 500 "results/medqa_grid_rerun/gemma4_31b/*.json"
fill frames-cues "--dataset_path data/frames_cues/orig_phrasing_full.jsonl" data/frames_index verbose_plain results/frames_cues_rerun/gemma4_31b 501 "results/frames_cues_rerun/gemma4_31b/*.json"
kill $op 2>/dev/null
echo "=== GEMMA_RETRY_DONE $(date -Is) ==="
