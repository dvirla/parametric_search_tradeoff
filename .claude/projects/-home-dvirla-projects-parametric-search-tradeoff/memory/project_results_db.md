---
name: project_results_db
description: The SQLite results DB was removed (2026-06-20); figures now read source JSON/CSV files directly via src/results_files.py + viz
metadata:
  node_type: memory
  type: project
  originSessionId: 3466705b-d778-4410-8a19-3c6214f072ae
---

As of 2026-06-20 the SQLite results store was **removed** to eliminate confusing duplicate state (multiple parametric clustering variants — original/_nli/_grader — collided in the entropy loader). Deleted: `results/results.db*`, `src/results_db.py`, `scripts/ingest_results.py`, `scripts/archive/analyze_phrasing.py`. Figures read the original files directly.

- **Aggregate accuracy/search (phrasing bars)**: `src/results_files.py: paired_eval_files(phrasings, model_canonical, grading="reevaluated")` reads the reevaluated baseline JSONs under `results/musique-<phrasing>/` and inner-joins on `example_id` per model. Natural2 records have **no `example_id`** — it's recovered from `data/musique_val_natural2.jsonl` (`text`→`example_id`, 599/600 covered) via `_load_pairing`, exactly as the old ingester did. File-based numbers verified byte-identical to the old `rdb.paired_eval`.
- **Per-hop uncertainty**: `viz.load_canonical_entropy` now reads ONLY `CANONICAL_ENTROPY_JSONS`, which points at the grader-reclustered `results/musique_parametric/musique_parametric_uncertainty_<model>_grader.json` (grader is the single canonical clustering). The DB branch is gone; the `--clustering`/`set_canonical_entropy_jsons` toggle added mid-session was reverted.
- **Identity**: `canonical_model` + `MODEL_ALIASES` moved from results_db into `src/viz.py` (folds `:`→`_`, `gemini-3.1-pro-preview`→`gemini-3-pro-preview`, nemotron size aliases). `paired_eval_files` canonicalizes the raw slug parsed from each baseline filename, so it matches the natural2 `gemini-3.1` file when asked for `gemini-3`.
- **make_paper_figures scope**: `--mode natural2` is now strictly **formal vs natural2** (natural-1 dropped from taxonomy/cell-shift/cascade/bars); default `--certain-rule` flipped to `entropy` (entropy==0 alone, no all-runs-correct gate). `--mode musique` and `--mode sharechat` kept, ported DB-free. The retired `--db` arg is gone.
- **Workflow manual**: `docs/RESULTS_WORKFLOW.md` rewritten file-based (run eval → re-eval → `make_paper_figures.py`). See [[project_grading_artifact_accuracy_gap]] and [[project_uncertainty_calibration]] for the grading/clustering context.
