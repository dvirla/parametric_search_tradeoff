#!/bin/bash
# FRAMES cue-briefing grid (7 conditions) over the Brave search backend, on nlp-srv3:
#   gemma4:31b   -> own GPU + private ollama daemon
#   gemini-3.5-flash -> Google API directly, no GPU/ollama needed
cd ~/parametric_search_tradeoff 2>/dev/null || cd /data/home/dvirla/parametric_search_tradeoff

CONDS="verbose_plain verbose_polite terse_plain verbose_natural verbose_elaborate verbose_query verbose_direct"

run_gemma(){
  model="gemma4:31b"; gpu="1"; port="11480"
  slug="gemma4_31b"
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

run_gemini(){
  model="gemini-3.5-flash"; slug="gemini-3.5-flash"
  echo "[$slug] started $(date -Is) (no GPU/ollama)"

  SEARCH_BACKEND=brave SCALE=full NUM_WORKERS=2 NO_GRADER=1 PARALLEL=0 \
    RESULTS_ROOT=results/frames_cues_full_brave CONDITIONS="$CONDS" \
    bash scripts/run_frames_grid_experiment.sh "$model"

  SEARCH_BACKEND=brave SCALE=full NUM_WORKERS=2 NO_GRADER=1 PARALLEL=0 \
    RESULTS_ROOT=results/frames_cues_rerun_brave CONDITIONS="verbose_plain" \
    bash scripts/run_frames_grid_experiment.sh "$model"

  echo "=== SRV3_BRAVE_${slug}_DONE $(date -Is) ==="
}

run_gemma  > /tmp/brave_gemma4_31b.log 2>&1 &
run_gemini > /tmp/brave_gemini-3.5-flash.log 2>&1 &
wait
echo "ALL_SRV3_FRAMES_BRAVE_DONE $(date -Is)"
