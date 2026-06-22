1. scripts/label_sharechat_hops.py — Streamlit labeling app
- Sidebar with progress bar and question navigator (filterable by labeled/unlabeled)
- Main view: natural + benchmark question, collapsible reasoning hint, tabbed model responses
- Hop annotation form: configurable number of hops (1–6), sub-question + gold answer per hop, aggregate answer, notes
- "Auto-suggest" button calls a configurable LLM (default gpt-oss:20b via ollama) to pre-populate hop fields from the question's reasoning text
- Auto-saves to data/curated_sharechat_hops.json after each "Save & Next"

2. scripts/run_sharechat_parametric_uncertainty.py — Per-hop uncertainty script
- Reads annotated hops from data/curated_sharechat_hops.json
- Runs the model K=5 times per sub-question (no search) to compute semantic entropy
- Output format matches MusiQue exactly: results/sharechat_parametric/curated_sharechat_parametric_uncertainty_<model>.json
- Supports --resume

3. scripts/analyze_parametric_search_interplay.py — Patched with 4 new args:
- --trace-dir (separate dir for traces, defaults to --eval-dir)
- --eval-glob / --eval-file-prefix (configurable uncertainty file pattern)
- --trace-glob / --trace-file-prefix (configurable trace file pattern)

Run order after labeling:
# Step 1: label
uv run streamlit run scripts/label_sharechat_hops.py -- --suggest-model gpt-oss:20b

# Step 2: compute per-hop uncertainty (on GPU machine with Ollama)
uv run python scripts/run_sharechat_parametric_uncertainty.py \
    --hops-json data/curated_sharechat_hops.json \
    --model-name qwen3.5:122b --provider ollama --num-runs 5 --resume

# Step 3: interplay analysis (uses benchmark traces for matching)
uv run python scripts/analyze_parametric_search_interplay.py \
    --eval-dir results/sharechat_parametric \
    --eval-glob "curated_sharechat_parametric_uncertainty_*.json" \
    --eval-file-prefix "curated_sharechat_parametric_uncertainty_" \
    --trace-dir results/curated_sharechat \
    --trace-glob "curated-sharechat-benchmark_*_baseline_agent_run_1_traces.json" \
    --trace-file-prefix "curated-sharechat-benchmark_" \
    --output-dir results/sharechat_parametric/interplay_analysis

# The ShareChat search-volume comparison is produced by make_paper_figures.py
# (fig: sharechat_search_comparison) from the baseline result JSONs.
uv run python scripts/make_paper_figures.py --output-dir results/paper_figures