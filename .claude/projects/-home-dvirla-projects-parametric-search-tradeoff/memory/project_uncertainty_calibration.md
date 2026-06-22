---
name: project_uncertainty_calibration
description: "Per-hop 'uncertain' is better defined by parametric accuracy than entropy=0; add_cost_cells now supports it"
metadata: 
  node_type: memory
  type: project
  originSessionId: f2facbe3-45f8-434d-b8a7-a965ecdfbe4c
---

The per-hop certainty rule in `analyze_parametric_search_interplay.py:854` is `certain = (semantic_entropy==0) AND (num_correct==num_runs)`. The `entropy==0` clause is the weak part.

**Calibration evidence** (`scripts/calibrate_uncertainty_threshold.py` → `results/natural2_paper_figures/uncertainty_calibration.png`): using commitment-locus in-trace correctness as ground truth for hops the agent SKIPPED (n=6141, benchmark+natural), parametric accuracy `num_correct/num_runs` predicts whether the skip was safe ~2× better than entropy (point-biserial r = 0.41 vs 0.19). P(skipped hop correct in-trace) rises monotonically with pacc: 16%(0)→61%(1.0). Even within entropy>0 hops it separates 16%→49%, so entropy discards usable signal. BUT even pacc=1 hops are only 61% correct in-trace — the isolated single-hop probe is OPTIMISTIC vs the multi-hop chain, so there is no clean binary "certain" line; treat uncertainty as graded (search-priority ≈ 1−pacc). pacc is discrete in 1/num_runs steps (0.2 at num_runs=5).

**Tooling (added 2026-06-07):** `viz.add_cost_cells(certain_rule="entropy"|"pacc", pacc_threshold=0.8)` — "pacc" sets `uncertain = (num_correct/num_runs) < threshold`, ignoring entropy; NaN probe → uncertain. `scripts/plot_missed_hop_outcome_breakdown.py` and `scripts/plot_missed_hop_knowledge.py` expose `--certain-rule`/`--pacc-threshold`. Recalibrated figures live in `results/natural2_paper_figures_pacc0.8/` (vs legacy `results/natural2_paper_figures/`). Analysis-layer only — no producer/Ollama re-run, since the CSVs already carry num_correct/num_runs.

**Effect of pacc<0.8 vs entropy rule:** truly-missed-hops-per-example drops (well-known hops reclassified certain), cost-of-a-miss (Δacc) rises (remaining misses genuinely unknown), and missed-in-correct hops become dominated by "never known" (52–83%). The [[project_missed_hop_paradox]] finding (natural = more misses, flat accuracy, cheaper-per-miss) is robust to the recalibration. Related: [[project_results_db]], [[project_grading_artifact_accuracy_gap]].
