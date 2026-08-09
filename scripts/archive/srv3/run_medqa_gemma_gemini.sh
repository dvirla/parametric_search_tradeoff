#!/bin/bash
cd /data/home/dvirla/parametric_search_tradeoff
mkdir -p results/cue_smoke_medqa
DRV=results/cue_smoke_medqa/multi_driver.log
: > "$DRV"
run() {  # model provider tmpl
  local model="$1" provider="$2" tmpl="$3"
  local slug=$(echo "$model" | tr ":/" "__")
  echo "START $model $tmpl $(date -Is)" >> "$DRV"
  uv run python scripts/run_qa_eval_experiment.py \
    --agent_type baseline --dataset medqa \
    --model_name "$model" --provider_name "$provider" \
    --query_template "$tmpl" --num_examples 50 --seed 0 \
    --run_name "$tmpl" --output_dir results/cue_smoke_medqa \
    --grader_model gemini-3-flash-preview --grader_provider Google \
    --num_workers 1 > "results/cue_smoke_medqa/${slug}_${tmpl}.log" 2>&1
  echo "DONE  $model $tmpl rc=$? $(date -Is)" >> "$DRV"
}
run "gemma4:31b"              "ollama" plain
run "gemma4:31b"              "ollama" elaborate
run "gemini-3.1-pro-preview"  "Google" plain
run "gemini-3.1-pro-preview"  "Google" elaborate
echo "ALL DONE $(date -Is)" >> "$DRV"
