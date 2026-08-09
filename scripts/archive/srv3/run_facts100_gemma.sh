#!/bin/bash
cd /data/home/dvirla/parametric_search_tradeoff
mkdir -p results/facts_gemma100
DRV=results/facts_gemma100/driver.log; : > "$DRV"
CMN="--agent_type baseline --dataset facts-open --dataset_path data/facts/facts_open_100.csv --model_name gemma4:31b --provider_name ollama --output_dir results/facts_gemma100 --grader_model gemini-3-flash-preview --grader_provider Google --num_workers 1"
echo "START plain $(date -Is)" >> "$DRV"
uv run python scripts/run_qa_eval_experiment.py $CMN --query_template plain --run_name plain > results/facts_gemma100/plain.log 2>&1
echo "DONE plain rc=$? $(date -Is)" >> "$DRV"
echo "START elaborate $(date -Is)" >> "$DRV"
uv run python scripts/run_qa_eval_experiment.py $CMN --query_template elaborate --run_name elaborate > results/facts_gemma100/elaborate.log 2>&1
echo "DONE elaborate rc=$? $(date -Is)" >> "$DRV"
echo "ALL DONE $(date -Is)" >> "$DRV"
