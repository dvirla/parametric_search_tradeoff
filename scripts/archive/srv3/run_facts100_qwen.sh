#!/bin/bash
cd /data/home/dvirla/parametric_search_tradeoff
mkdir -p results/facts_qwen100
CMN="--agent_type baseline --dataset facts-open --dataset_path data/facts/facts_open_100.csv --model_name qwen3.5:122b --provider_name ollama --output_dir results/facts_qwen100 --grader_model gemini-3-flash-preview --grader_provider Google --num_workers 1"
uv run python scripts/run_qa_eval_experiment.py $CMN --query_template plain --run_name plain > results/facts_qwen100/plain.log 2>&1
echo "PLAIN done rc=$?" >> results/facts_qwen100/driver.log
uv run python scripts/run_qa_eval_experiment.py $CMN --query_template elaborate --run_name elaborate > results/facts_qwen100/elaborate.log 2>&1
echo "ELABORATE done rc=$?" >> results/facts_qwen100/driver.log
