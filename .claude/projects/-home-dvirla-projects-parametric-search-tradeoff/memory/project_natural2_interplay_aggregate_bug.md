---
name: project_natural2_interplay_aggregate_bug
description: "natural2 interplay_summary.csv aggregate_correct is wrong — identical to formal, not the natural2 run"
metadata: 
  node_type: memory
  type: project
  originSessionId: f2facbe3-45f8-434d-b8a7-a965ecdfbe4c
---

`results/musique-natural2/interplay_analysis/interplay_summary.csv` has an `aggregate_correct` column that is **100% identical to the formal/parametric run, example-by-example** (verified: agreement 1.0 across 1797 joined rows; both 0.376 acc). The interplay producer (`analyze_parametric_search_interplay.py`) populates `aggregate_correct` from `entry["aggregate_result"]["is_correct"]` in the parametric uncertainty JSON it consumes, and for natural2 that aggregate is the formal aggregate question — so it does NOT reflect the natural2 baseline eval. (Nat1's summary, by contrast, has its own distinct accuracy, so only natural2 is affected — or at least natural2 is the clear case.)

**Why:** any analysis joining interplay per-hop rows to "did the final answer come out correct" will be silently wrong for natural2 if it trusts the summary's `aggregate_correct`.

**How to apply:** source per-example final-answer correctness from the results DB (`src/results_db.py` `load_eval(base_dataset='musique', phrasing=..., model=..., grading='reevaluated')`) keyed on `example_id`, not from `aggregate_correct`. The DB has example_id populated for formal/natural/natural2 under both `original` and `reevaluated` grading. `scripts/plot_missed_hop_outcome_breakdown.py` does this correctly. Related: [[project_missed_hop_paradox]], [[project_grading_artifact_accuracy_gap]], [[project_results_db]].

**FIXED (2026-06-07):**
- `scripts/patch_interplay_aggregate_correct.py` — no-Ollama utility that rewrites `aggregate_correct` in the 3 `interplay_summary.csv` + `example_metrics.csv` from the DB (default `--grading reevaluated`), backing up originals to `*.preDBpatch.bak`. **Already run**: all three phrasings now hold REEVALUATED grading (Gemini 0.660/0.634/0.644, Nemotron 0.360/0.431/0.404, Qwen 0.482/0.475/0.496); natural2 ≠ formal confirmed. NOTE side-effect: `make_paper_figures.py` `fig_missed_hop_cost` now reads reevaluated outcomes for formal/nat1 too (was original).
- Producer `analyze_parametric_search_interplay.py` gained `--outcome-from-db --outcome-phrasing {formal,natural,natural2} --outcome-grading` to override `aggregate_correct` from the DB on future full runs (normalizes trace model names like `musique-natural2_<slug>_baseline_agent_run_1` before `load_eval`), plus a loud WARNING when `--natural-jsonl` is used without `--outcome-from-db`. Both verified working locally with heuristic attribution (no Ollama needed for the analysis pass).
