#!/usr/bin/env bash
# nlp-srv3 driver for the HotpotQA cue grid (hotpotqa-300 x 9 cue conditions).
#
# GPU BUDGET -- re-check with `nvidia-smi` before every launch, this changes hour to hour.
# At launch (2026-09-02) srv3's four RTX PRO 6000 Blackwell (98GB) cards were:
#   GPU 0  19.2 GB shohamg, ~78 GB free   -> us (all three models share it, ~40 GB total)
#   GPU 1  83.1 GB another user's vLLM, ~15 GB free  -> DO NOT TOUCH
#   GPU 2  95.6 GB amosy3's ollama, ~2 GB free       -> DO NOT TOUCH
#   GPU 3  39.9 GB shohamg + 5.2 GB a stale ollama of ours (pid 1814375, deliberately left alone)
#
# GPU 1 was 893 MiB when this grid was planned and looked idle -- that vLLM had simply not
# allocated yet, and took 83 GB half an hour later. A near-empty vLLM process is NOT a free GPU;
# it preallocates most of the card once it starts serving. Always re-read `nvidia-smi` (and check
# free VRAM, not just who is listed) immediately before launching, and never widen GPU_LIST on
# the strength of an earlier reading.
#
# gemma4:31b / gemma4:e4b are NOT here: they crash on srv3's Blackwell GPUs ("llama-server
# process has terminated: signal: aborted") across ollama 0.22.0 and 0.32.14 alike, and with
# OLLAMA_FLASH_ATTENTION=0. Not reproducible on Athena. Route the gemma4 family, the 120B+ pair,
# and nemotron-cascade-2:30b (not pulled here) to scripts/athena_submit_hotpotqa_cue_grid.sh.
#
# Models are assigned round-robin to GPUs and run CONCURRENTLY, each with its own ollama daemon
# on its own port. With 3 models on 2 GPUs the third waits for GPU 0 to free up.
#
# Usage:
#   bash scripts/srv3_hotpotqa_cue_grid.sh                 # default 3 models
#   MODELS="qwen3.5:4b" bash scripts/srv3_hotpotqa_cue_grid.sh
#   DRYRUN=1 bash scripts/srv3_hotpotqa_cue_grid.sh
set -uo pipefail

# Isolated worktree so this doesn't fight whatever branch the main srv3 checkout is on.
REPO="${REPO:-/data/home/dvirla/parametric_search_tradeoff_hpqcue}"
cd "$REPO" || { echo "ERROR: $REPO missing. Create it with:"; \
  echo "  git -C /data/home/dvirla/parametric_search_tradeoff worktree add $REPO hotpotqa-cue-pilot"; exit 1; }

DATASET="${DATASET:-hotpotqa-300}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_cue_grid}"
CONDITIONS="${CONDITIONS:-plain natural elaborate polite direct confident_parametric query multiturn searchmulti}"
DRYRUN="${DRYRUN:-0}"
read -r -a GPU_LIST <<< "${GPUS:-0}"   # override with GPUS="0 1" once GPU 1 frees up

read -r -a MODELS_ARR <<< "${MODELS:-qwen3.5:35b gpt-oss:20b qwen3.5:4b}"
workers_for() { case "$1" in qwen3.5:4b) echo 6 ;; *) echo 4 ;; esac; }

run_one() {
  local model="$1" gpu="$2" port="$3"
  local workers; workers="$(workers_for "$model")"
  local slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES="$gpu"
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=$workers nohup ollama serve > /tmp/oll_hpqcue_${slug}.log 2>&1 &
  local ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port workers=$workers ollama_pid=$ollama_pid start $(date -Is)"
  ollama show "$model" >/dev/null 2>&1 || ollama pull "$model"

  DATASET="$DATASET" CONDITIONS="$CONDITIONS" RESULTS_ROOT="$RESULTS_ROOT" \
  INDEX_DIR=data/hotpotqa_index LOCAL_BACKEND=bm25 \
  NO_GRADER=1 NUM_WORKERS="$workers" PARALLEL=0 DRYRUN="$DRYRUN" \
    bash scripts/run_hotpotqa_cue_experiment.sh "$model"

  echo "[$slug] DONE $(date -Is)"
  kill "$ollama_pid" 2>/dev/null || true
}

pids=()
i=0
for model in "${MODELS_ARR[@]}"; do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  port=$((11700 + i))
  slug="${model//:/_}"; slug="${slug//\//_}"
  run_one "$model" "$gpu" "$port" > "scratch_hpqcue_${slug}.log" 2>&1 &
  pids+=($!)
  echo "launched $model on GPU $gpu (pid $!) -> scratch_hpqcue_${slug}.log"
  i=$((i+1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[[ "$fail" -eq 1 ]] && echo "WARNING: a model run exited non-zero (check scratch_hpqcue_*.log)"
echo "srv3 HotpotQA cue grid finished $(date -Is)"
