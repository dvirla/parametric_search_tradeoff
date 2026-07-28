#!/usr/bin/env bash
# Remote (nlp-srv3) driver: MedQA cue grid for Qwen + Gemma, 200 Qs, local BM25 index.
# Models run SEQUENTIALLY to avoid GPU contention. NO Brave: --search-backend local everywhere.
# Answerers are free (local Ollama); only cost = Gemini flash grader (GOOGLE_API_KEY).
set -u
export PATH="/data/home/dvirla/.local/bin:$PATH"
cd /data/home/dvirla/parametric_search_tradeoff
mkdir -p results/medqa_cue_200

COMMON=(--dataset medqa --agent_type baseline
  --search-backend local --index-dir data/medqa_index --local-backend bm25
  --grader_model gemini-3-flash-preview --grader_provider Google
  --num_examples 200 --num_workers 1 --seed 0
  --output_dir results/medqa_cue_200 --resume)

for MODEL in qwen3.5:122b gemma4:31b; do
  for TPL in plain elaborate; do
    echo "=========== START $MODEL $TPL $(date -Is) ==========="
    uv run python scripts/run_qa_eval_experiment.py "${COMMON[@]}" \
      --model_name "$MODEL" --provider_name ollama \
      --query_template "$TPL" --run_name "$TPL"
    echo "=========== END   $MODEL $TPL rc=$? $(date -Is) ==========="
  done
done
echo "ALL_REMOTE_DONE $(date -Is)"
