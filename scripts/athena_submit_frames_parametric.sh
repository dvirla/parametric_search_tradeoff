#!/usr/bin/env bash
# Submit the FRAMES PARAMETRIC (no_search) probe on Athena: one job per (model, condition),
# each doing RUNS repetitions of all 501 original-phrasing FRAMES questions.
#
# Twin of scripts/athena_submit_hotpotqa_parametric.sh. Same defaults, same conditions, so the
# FRAMES and HotpotQA parametric arms stay directly comparable.
#
# Usage:
#   bash scripts/athena_submit_frames_parametric.sh
#   DRYRUN=1 MODELS="gemma4-frames-robust-q4km:latest" bash scripts/athena_submit_frames_parametric.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${RUNS:-5}"
DRYRUN="${DRYRUN:-0}"

read -r -a MODELS_ARR <<< "${MODELS:-gemma4:31b}"
read -r -a CONDITIONS_ARR <<< "${CONDITIONS:-plain elaborate direct multiturn}"

# gemma4's chat template + tool parser are compiled into the ollama binary; 0.18.2 has no gemma4
# symbols and 400s with "does not support tools", which the eval SKIPS silently. The `gemma4-*` arm
# covers the FRAMES-SFT checkpoints (gemma4-frames-robust-*-q4km:latest), which are gemma4
# architecture but do not match `gemma4:*`.
ollama_ver_for() { case "$1" in gemma4:*|gemma4-*) echo "0.32.5" ;; *) echo "" ;; esac; }
partition_for()  { case "$1" in qwen3.5:122b|gpt-oss:120b) echo "rtx6k-shared,h200-shared" ;; *) echo "" ;; esac; }
workers_for()    { case "$1" in qwen3.5:122b) echo 1 ;; *) echo 2 ;; esac; }

n=0
for model in "${MODELS_ARR[@]}"; do
  ver="$(ollama_ver_for "$model")"; part="$(partition_for "$model")"; workers="$(workers_for "$model")"
  for cond in "${CONDITIONS_ARR[@]}"; do
    args=(--job-name="frmp_$(echo "$model" | tr ':.' '__')_${cond}")
    [[ -n "$part" ]] && args+=(--partition="$part")
    exports="ALL,MODEL=${model},CONDITION=${cond},RUNS=${RUNS},NUM_WORKERS=${workers}"
    [[ -n "$ver" ]] && exports="${exports},OLLAMA_VER=${ver}"
    if [[ "$DRYRUN" == "1" ]]; then
      echo "sbatch ${args[*]} --export=${exports} scripts/athena_frames_parametric.job"
    else
      sbatch "${args[@]}" --export="$exports" scripts/athena_frames_parametric.job | tail -1
    fi
    n=$((n+1))
  done
done
echo "submitted $n job(s): runs=$RUNS results=results/frames_parametric"
