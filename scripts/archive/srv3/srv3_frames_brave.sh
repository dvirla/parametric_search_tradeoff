#!/bin/bash
# FRAMES cue-briefing grid (7 conditions, matching scripts/make_cue_briefing_figures.py) over the
# Brave search backend, for the ollama-hosted models: gemma4:31b, qwen3.5:122b, gpt-oss:120b.
# Each model gets its own ollama daemon pinned to a free GPU + private port (GPU0 is busy with
# another user's job -- avoid it). Pattern mirrors srv3_phase2.sh from the local-index rerun.
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

CONDS="verbose_plain verbose_polite terse_plain verbose_natural verbose_elaborate verbose_query verbose_direct"

run_one(){
  model="$1"; gpu="$2"; port="$3"
  slug="${model//:/_}"; slug="${slug//\//_}"
  export CUDA_VISIBLE_DEVICES=$gpu
  OLLAMA_HOST=127.0.0.1:$port OLLAMA_NUM_PARALLEL=2 nohup ollama serve > /tmp/oll_brave_${slug}.log 2>&1 &
  ollama_pid=$!
  for i in $(seq 1 60); do curl -s 127.0.0.1:$port/api/version >/dev/null 2>&1 && break; sleep 2; done
  export OLLAMA_BASE_URL=http://127.0.0.1:$port/v1
  echo "[$slug] gpu=$gpu port=$port ollama_pid=$ollama_pid started $(date -Is)"

  SEARCH_BACKEND=brave SCALE=full NUM_WORKERS=2 NO_GRADER=1 PARALLEL=0 \
    RESULTS_ROOT=results/frames_cues_full_brave CONDITIONS="$CONDS" \
    bash scripts/run_frames_grid_experiment.sh "$model"

  SEARCH_BACKEND=brave SCALE=full NUM_WORKERS=2 NO_GRADER=1 PARALLEL=0 \
    RESULTS_ROOT=results/frames_cues_rerun_brave CONDITIONS="verbose_plain" \
    bash scripts/run_frames_grid_experiment.sh "$model"

  echo "=== SRV3_BRAVE_${slug}_DONE $(date -Is) ==="
}

run_one gemma4:31b   1 11480 > /tmp/brave_gemma4_31b.log   2>&1 &
run_one qwen3.5:122b 2 11481 > /tmp/brave_qwen3.5_122b.log 2>&1 &
run_one gpt-oss:120b 3 11482 > /tmp/brave_gpt-oss_120b.log 2>&1 &
wait
echo "ALL_SRV3_FRAMES_BRAVE_DONE $(date -Is)"
