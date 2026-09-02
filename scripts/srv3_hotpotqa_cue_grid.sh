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
# gemma4 on srv3: the Blackwell load crash recorded on 2026-08-20 ("llama-server process has
# terminated: signal: aborted", across ollama 0.22.0 and a built 0.32.14) NO LONGER REPRODUCES --
# retested 2026-09-02 on srv3's default ollama 0.22.0: gemma4:31b loads in 6.5 s, reports
# `tools` among its capabilities, and generates normally. gemma4 is srv3-eligible again. If it
# ever aborts on load here again, re-test before assuming, and fall back to Athena.
#
# Only qwen3.5:122b (~81 GB) still requires Athena's h200/rtx6k partitions. nemotron-cascade-2:30b
# is simply not pulled on srv3. Both stay on scripts/athena_submit_hotpotqa_cue_grid.sh.
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
read -r -a GPU_LIST <<< "${GPUS:-0}"   # models are assigned round-robin over these
# Distinct port range per driver invocation, so several instances can run side by side (e.g. one
# pinned to GPU 1 for a 120B model, another pinned to GPU 3 for the gemma4 pair).
PORT_BASE="${PORT_BASE:-11700}"
# Cap the KV cache. Ollama sizes it as (context length x OLLAMA_NUM_PARALLEL), and some models
# declare an enormous default context -- gemma4:31b advertises 262144, which at 4 parallel asks
# for a 1,048,576-token cache and cannot fit beside another tenant on a 98GB card. Ollama then
# retry-loops the model load forever: no error, no completions, the job just sits at 0/N (this
# cost ~20 min of a gemma4:31b run before it was caught). The agent's real prompts are a few
# thousand tokens (question + <=10 BM25 passages of <=1500 chars + optional history), so 32768 is
# far above anything it sends and does NOT truncate -- it only stops the cache preallocation from
# being absurd. Raise it only if a model starts reporting truncated prompts.
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-32768}"

read -r -a MODELS_ARR <<< "${MODELS:-qwen3.5:35b gpt-oss:20b qwen3.5:4b}"
workers_for() { case "$1" in qwen3.5:4b) echo 6 ;; *) echo 4 ;; esac; }

run_one() {
  local model="$1" gpu="$2" port="$3"
  local workers; workers="$(workers_for "$model")"
  local slug="${model//:/_}"; slug="${slug//\//_}"
  # PORT COLLISION GUARD. srv3 accumulates stale `ollama serve` daemons from old sessions (16+
  # ports were bound at one point). If our chosen port is already taken, our `ollama serve` fails
  # to bind and exits, the eval happily connects to the SQUATTER instead, and the run silently
  # executes on whatever GPU that stale daemon was pinned to -- ignoring GPUS/CUDA_VISIBLE_DEVICES
  # entirely. That is how a gpt-oss:120b run meant for GPU 1 ended up on GPU 0 and pushed it to
  # 99.3%. Refuse to start rather than run somewhere unintended.
  if curl -s --max-time 3 "127.0.0.1:${port}/api/version" >/dev/null 2>&1; then
    echo "[$slug] ERROR: port $port is ALREADY SERVING (stale ollama from another session?)." >&2
    echo "[$slug]        Pick a free PORT_BASE, or kill the squatter. Refusing to start." >&2
    return 1
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=$workers OLLAMA_CONTEXT_LENGTH=$OLLAMA_CONTEXT_LENGTH \
    nohup ollama serve > /tmp/oll_hpqcue_${slug}.log 2>&1 &
  local ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  # Confirm the daemon answering on $port is the one WE started, and that it sees the GPU we asked
  # for -- a bind failure would otherwise surface only as mysteriously slow, mysteriously placed runs.
  if ! kill -0 "$ollama_pid" 2>/dev/null; then
    echo "[$slug] ERROR: our ollama serve (pid $ollama_pid) died -- port $port likely taken." >&2
    return 1
  fi
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
  port=$((PORT_BASE + i))
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
