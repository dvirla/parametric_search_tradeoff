# Paraphrse only epistemic conditions (strong_boost, strong_hedge) on top of a pre-audited neutral file (e.g. neutral_audited.jsonl) used as --base.
uv run python scripts/paraphrase_frames_cues.py --validate \
  --conditions epi_strong_boost epi_strong_hedge \
  --base data/frames_cues/neutral_audited.jsonl \
  --model gemini-3-flash-preview --provider Google \
  --validate-model gemini-3-flash-preview --validate-provider Google