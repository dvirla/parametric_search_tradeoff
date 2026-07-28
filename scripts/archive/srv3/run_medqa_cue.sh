#!/bin/bash
cd /data/home/dvirla/parametric_search_tradeoff
mkdir -p results/cue_smoke_medqa
COMMON="--agent_type baseline --dataset medqa --model_name qwen3.5:122b --provider_name ollama --num_examples 50 --seed 0 --output_dir results/cue_smoke_medqa --grader_model gemini-3-flash-preview --grader_provider Google --num_workers 1"
uv run python scripts/run_qa_eval_experiment.py $COMMON --query_template plain --run_name plain > results/cue_smoke_medqa/plain.log 2>&1
echo "PLAIN done rc=$?" >> results/cue_smoke_medqa/driver.log
uv run python scripts/run_qa_eval_experiment.py $COMMON --query_template elaborate --run_name elaborate > results/cue_smoke_medqa/elaborate.log 2>&1
echo "ELABORATE done rc=$?" >> results/cue_smoke_medqa/driver.log
