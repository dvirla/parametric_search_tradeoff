---
name: project_natural2_entropy_override_bug
description: canonical-entropy override silently no-opped for natural2 (decorated model names), inflating its Missed cell; fixed in viz.canonical_model
metadata:
  type: project
---

`viz.load_interplay_summary` overrides per-hop entropy from the grader-reclustered
canonical JSONs, keyed by model slug. The natural2 interplay_summary.csv stores
DECORATED model names (e.g. `musique-natural2_gemini-3-pro-preview_baseline_agent_run_1`),
so the override key never matched the canonical JSON (`gemini-3-pro-preview`) and
**silently no-opped for natural2** — it kept stale pre-reclustering entropy, while
benchmark got the canonical values. `_normalize_natural2_model_names` strips the
prefix only AFTER the override, too late.

**Impact (was):** 12.3% of natural2 hops had the wrong uncertain-status; 574 hops
spuriously counted as uncertain (M-eligible) vs 90 the other way — net ~484 hops
inflating the natural2 Missed/uncertain pool (Gemini +293, Qwen +184, Nemotron +97).
This artificially inflated ΔM in `fig1_quadrant_taxonomy_simple`, the cell-shift
figure, calibration, and Test 1. After fix, ΔM (missed-rate increase) shrank:
Gemini +0.056->+0.016 (−71%), Nemotron +0.053->+0.031, Qwen +0.068->+0.043.

**Fix (2026-06-21):** `viz.canonical_model` now strips dataset prefixes
(`musique[-natural[2]]_`, `curated-sharechat[-benchmark]_`) and the
`_baseline_agent_run_\d+` suffix before aliasing; `load_interplay_summary` keys the
entropy override on `canonical_model(model)`. Verified entropy now identical across
phrasings per (model, example, hop).

**How to apply:** any natural2 figure/CSV generated before this fix is stale —
regenerate (`make_paper_figures.py --mode natural2`, `test_missed_hop_necessity.py`,
`inspect_missed_leakage.py`). Distinct from [[project_natural2_interplay_aggregate_bug]]
(that one is the aggregate_correct column; this one is per-hop entropy). Related:
[[project_uncertainty_calibration]], [[project_natural2_paraphrase_leakage]].
