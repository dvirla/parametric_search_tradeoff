#!/usr/bin/env bash
# Submit the full HotpotQA cue grid on Athena: one SLURM job per (model, condition).
#
# WHY ONE JOB PER CONDITION rather than one per model: 9 conditions x 300 examples is ~9 h for a
# 30B model and far more for the 120B+ pair, which overruns even the 12h_4g QoS ceiling. Per
# condition each job is ~1-5 h, well inside the wall clock, and the qos's 4-GPU budget stays
# saturated because SLURM dequeues the next condition the moment one finishes. Every run is
# --resume, so a job that is preempted or times out costs only a resubmit of the same line.
#
# Model split: srv3 takes qwen3.5:35b / gpt-oss:20b / qwen3.5:4b (see
# scripts/srv3_hotpotqa_cue_grid.sh). Everything else runs here, because:
#   * gemma4:31b (and by extension gemma4:e4b -- same vision tower) CRASHES on srv3's Blackwell
#     GPUs, reproducibly, across ollama versions. Athena-only.
#   * nemotron-cascade-2:30b isn't pulled on srv3 at all.
#   * qwen3.5:122b / gpt-oss:120b need a 140GB h200 node -- on a 40GB a100-public node ollama
#     silently CPU-offloads half the layers and every request then times out forever.
#
# Usage:
#   bash scripts/athena_submit_hotpotqa_cue_grid.sh              # submit everything
#   DRYRUN=1 bash scripts/athena_submit_hotpotqa_cue_grid.sh     # print sbatch lines only
#   MODELS="gemma4:31b" bash scripts/athena_submit_hotpotqa_cue_grid.sh
#   CONDITIONS="plain query" bash scripts/athena_submit_hotpotqa_cue_grid.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${DATASET:-hotpotqa-300}"
RESULTS_ROOT="${RESULTS_ROOT:-results/hotpotqa_cue_grid}"
DRYRUN="${DRYRUN:-0}"

read -r -a MODELS_ARR <<< "${MODELS:-gemma4:31b gemma4:e4b nemotron-3-nano:30b nemotron-cascade-2:30b gpt-oss:120b qwen3.5:122b}"
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-plain natural elaborate polite direct confident_parametric query multiturn searchmulti}"

# Per-model launch parameters. OLLAMA_VER: gemma4's renderer/parser is compiled into the ollama
# binary and the container's built-in 0.18.2 has no gemma4 symbols -- it returns 400 "does not
# support tools", which the eval treats as non-retryable and SKIPS, so the job completes having
# done zero work. PARTITION: override only for the 120B+ pair. WORKERS: concurrency the model's
# ollama runner survives without being OOM-killed.
ollama_ver_for() { case "$1" in gemma4:*) echo "0.32.5" ;; *) echo "" ;; esac; }
partition_for()  { case "$1" in qwen3.5:122b|gpt-oss:120b) echo "h200-shared" ;; *) echo "" ;; esac; }
workers_for()    { case "$1" in qwen3.5:122b) echo 1 ;; gpt-oss:120b) echo 2 ;; gemma4:e4b|qwen3.5:4b) echo 6 ;; *) echo 4 ;; esac; }

n=0
for model in "${MODELS_ARR[@]}"; do
  ver="$(ollama_ver_for "$model")"; part="$(partition_for "$model")"; workers="$(workers_for "$model")"
  for cond in "${CONDITIONS_ARR[@]}"; do
    args=(--job-name="hpq_$(echo "$model" | tr ':.' '__')_${cond}")
    [[ -n "$part" ]] && args+=(--partition="$part")
    exports="ALL,MODEL=${model},DATASET=${DATASET},CONDITIONS=${cond},NUM_WORKERS=${workers},RESULTS_ROOT=${RESULTS_ROOT}"
    [[ -n "$ver" ]] && exports="${exports},OLLAMA_VER=${ver}"
    if [[ "$DRYRUN" == "1" ]]; then
      echo "sbatch ${args[*]} --export=${exports} scripts/athena_hotpotqa_cue_eval.job"
    else
      sbatch "${args[@]}" --export="$exports" scripts/athena_hotpotqa_cue_eval.job | tail -1
    fi
    n=$((n+1))
  done
done
echo "submitted $n job(s) (dataset=$DATASET results=$RESULTS_ROOT)"
