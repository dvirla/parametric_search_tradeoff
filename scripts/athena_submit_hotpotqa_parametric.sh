#!/usr/bin/env bash
# Submit the HotpotQA PARAMETRIC (no_search) probe on Athena: one job per (model, condition),
# each doing RUNS repetitions of all 300 questions.
#
# 6 models x 4 conditions = 24 jobs; at RUNS=5 that is 6*4*5*300 = 36,000 rollouts.
# One job per CONDITION (not per run): 5 runs x 300 = 1500 rollouts per job fits the 12h QoS
# ceiling comfortably, while one-job-per-run would multiply the model-load overhead by 5.
#
# PARTITIONS. The ~30B models fit anywhere Athena offers, including the 40GB a100-public nodes,
# so they take the job file's full #SBATCH list. gpt-oss:120b (~65GB) and qwen3.5:122b (~81GB) do
# not fit 40GB, so they are pinned to `rtx6k-shared,h200-shared` -- BOTH, not h200 alone:
# rtx6k-shared has two ~98GB nodes and usually dequeues sooner than the single h200 node, which
# is where the search-grid's 122b jobs sat at (Priority) making zero progress.
#
# Usage:
#   bash scripts/athena_submit_hotpotqa_parametric.sh
#   DRYRUN=1 bash scripts/athena_submit_hotpotqa_parametric.sh
#   MODELS="gemma4:31b" CONDITIONS="plain" bash scripts/athena_submit_hotpotqa_parametric.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${DATASET:-hotpotqa-300}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_parametric}"
RUNS="${RUNS:-5}"
DRYRUN="${DRYRUN:-0}"

read -r -a MODELS_ARR <<< "${MODELS:-gemma4:31b nemotron-3-nano:30b nemotron-cascade-2:30b gpt-oss:20b gpt-oss:120b qwen3.5:122b}"
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-plain elaborate direct multiturn}"

# gemma4's chat template + tool parser are compiled into the ollama binary; 0.18.2 has no gemma4
# symbols and 400s with "does not support tools", which the eval SKIPS silently.
ollama_ver_for() { case "$1" in gemma4:*) echo "0.32.5" ;; *) echo "" ;; esac; }
# Only the 120B+ pair needs a big card; everything else takes the job file's full partition list.
partition_for()  { case "$1" in qwen3.5:122b|gpt-oss:120b) echo "rtx6k-shared,h200-shared" ;; *) echo "" ;; esac; }
# --no_grader removes the grader latency that throttled request rate; high concurrency then
# saturates ollama's queue and the client times out waiting. 122B -> 1, else 2.
workers_for()    { case "$1" in qwen3.5:122b) echo 1 ;; *) echo 2 ;; esac; }

n=0
for model in "${MODELS_ARR[@]}"; do
  ver="$(ollama_ver_for "$model")"; part="$(partition_for "$model")"; workers="$(workers_for "$model")"
  for cond in "${CONDITIONS_ARR[@]}"; do
    args=(--job-name="hpqp_$(echo "$model" | tr ':.' '__')_${cond}")
    [[ -n "$part" ]] && args+=(--partition="$part")
    exports="ALL,MODEL=${model},DATASET=${DATASET},CONDITION=${cond},RUNS=${RUNS},NUM_WORKERS=${workers},RESULTS_ROOT=${RESULTS_ROOT}"
    [[ -n "$ver" ]] && exports="${exports},OLLAMA_VER=${ver}"
    if [[ "$DRYRUN" == "1" ]]; then
      echo "sbatch ${args[*]} --export=${exports} scripts/athena_hotpotqa_parametric.job"
    else
      sbatch "${args[@]}" --export="$exports" scripts/athena_hotpotqa_parametric.job | tail -1
    fi
    n=$((n+1))
  done
done
echo "submitted $n job(s): dataset=$DATASET runs=$RUNS results=$RESULTS_ROOT"
