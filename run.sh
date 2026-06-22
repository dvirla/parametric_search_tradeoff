# Build all paper figures + stats from the standard interplay/commitment layout
# (replaces the old unified_interplay_analysis.py runs; tuned + sharechat handled by default).
uv run python scripts/make_paper_figures.py --output-dir results/paper_figures

uv run python scripts/bridge_sharechat_confidence.py \
  --confidence-csv gemini-3-pro-preview:results/curated_sharechat/atomic_fact_confidence_gemini.csv \
  --output-dir results/curated_sharechat/interplay_analysis
        
uv run python scripts/enrich_sft_nosearch.py \
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
    --output-dir data/sft/musique_nosearch \
    --lookback-days 90 \
    --output-traces data/sft/musique_nosearch/raw_parametric_traces.json

uv run python scripts/curate_onpolicy_sft.py \
    --input-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft.jsonl \
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \
    --natural-jsonl data/musique_train_natural.jsonl \
    --output-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft_curated.jsonl


# 1) Drop the tuned parametric uncertainty JSON into the canonical location
#    (slug-normalize the filename: ':' → '_')
cp results/musique-natural/interplay_analysis_tuned_stage/musique_parametric_uncertainty_nemotron-3-nano-musique-v3-aug:latest.json \
    results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano-musique-v3-aug_latest.json

# 2) Figures already include the tuned natural slice — see make_paper_figures.py at top.