# Paraphrse only epistemic conditions (strong_boost, strong_hedge) on top of a pre-audited neutral file (e.g. neutral_audited.jsonl) used as --base.
uv run python scripts/paraphrase_frames_cues.py --validate \
  --conditions epi_strong_boost epi_strong_hedge \
  --base data/frames_cues/neutral_audited.jsonl \
  --model gemini-3-flash-preview --provider Google \
  --validate-model gemini-3-flash-preview --validate-provider Google

uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues \
  --dataset_path data/frames_cues/orig_phrasing50.jsonl --query_template direct \
  --search-backend local --index-dir data/frames_index --local-backend bm25 \
  --agent_type baseline --model_name gemini-3.1-pro-preview --provider_name Google \
  --grader_provider Google --grader_model gemini-3-flash-preview \
  --run_name verbose_direct --output_dir results/frames_cues/gemini-3.1-pro-preview --num_workers 4 --resume

uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues \
  --dataset_path data/frames_cues/neutral_matched50.jsonl --query_template direct \
  --search-backend local --index-dir data/frames_index --local-backend bm25 \
  --agent_type baseline --model_name gemini-3.1-pro-preview --provider_name Google \
  --grader_provider Google --grader_model gemini-3-flash-preview \
  --run_name terse_direct --output_dir results/frames_cues/gemini-3.1-pro-preview --num_workers 4 --resume

uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues \
  --dataset_path data/frames_cues/orig_phrasing50.jsonl --query_template direct \
  --search-backend local --index-dir data/frames_index --local-backend bm25 \
  --agent_type baseline --model_name qwen3.5:122b --provider_name ollama \
  --grader_provider Google --grader_model gemini-3-flash-preview \
  --run_name verbose_direct --output_dir results/frames_cues/qwen3.5_122b --num_workers 4 --resume > qwen_verbose_direct.log 2>&1 &

uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues \
  --dataset_path data/frames_cues/orig_phrasing50.jsonl --query_template direct \
  --search-backend local --index-dir data/frames_index --local-backend bm25 \
  --agent_type baseline --model_name nemotron-3-nano:30b --provider_name ollama \
  --grader_provider Google --grader_model gemini-3-flash-preview \
  --run_name verbose_direct --output_dir results/frames_cues/nemotron-3-nano_30b --num_workers 4 --resume > nemotron_verbose_direct.log 2>&1 &

uv run python scripts/run_qa_eval_experiment.py --dataset frames-cues \
  --dataset_path data/frames_cues/orig_phrasing50.jsonl --query_template direct \
  --search-backend local --index-dir data/frames_index --local-backend bm25 \
  --agent_type baseline --model_name gemma4:31b --provider_name ollama \
  --grader_provider Google --grader_model gemini-3-flash-preview \
  --run_name verbose_direct --output_dir results/frames_cues/gemma4_31b --num_workers 4 --resume > gemma_verbose_direct.log 2>&1 &

# ---------------------------------------------------------------------------
# Full-scale (501-row) directive grid on the local models, resumable & dynamic.
# Reuses the saved 50-example results per condition (the _full files are strict
# supersets with identical question text), and excludes the epistemic conditions
# so this does NOT wait on the epi_strong_* paraphrases. Grader held constant at
# gemini-3-flash-preview to stay apples-to-apples with the saved 50-row rows.
#
# All three models in parallel (one bg process each; logs scratch_grid_<slug>.log).
# To run a single model / use fewer GPUs, just pass that one model name. Stopping
# (Ctrl-C / kill) and re-running the same command resumes where it left off.
export OLLAMA_MAX_LOADED_MODELS=3   # hold all three resident across the 4 GPUs
export OLLAMA_SCHED_SPREAD=1        # spread the 122B across GPUs
CONDITIONS="verbose_plain verbose_natural verbose_elaborate verbose_polite verbose_direct verbose_query \
            terse_plain terse_natural terse_elaborate terse_polite terse_direct terse_query" \
SCALE=full \
GRADER_PROVIDER=Google GRADER_MODEL=gemini-3-flash-preview \
bash scripts/run_frames_grid_experiment.sh "qwen3.5:122b" "nemotron-3-nano:30b" "gemma4:31b"
# Single-model variant (fewer GPUs):
#   SCALE=full GRADER_PROVIDER=Google GRADER_MODEL=gemini-3-flash-preview \
#     bash scripts/run_frames_grid_experiment.sh "qwen3.5:122b"
# Later, add the epistemic conditions (after the paraphrases finish) without
# re-running the grid:
#   CONDITIONS="epi_strong_boost epi_strong_hedge" SCALE=full \
#   GRADER_PROVIDER=Google GRADER_MODEL=gemini-3-flash-preview \
#     bash scripts/run_frames_grid_experiment.sh "qwen3.5:122b" "nemotron-3-nano:30b" "gemma4:31b"