# Results Workflow — how to run the scripts now

Results are **plain files** under `results/`. There is no database: the figure script reads
the source JSON/CSV files directly. (The former SQLite store `results/results.db` +
`scripts/ingest_results.py` + `src/results_db.py` were removed — the files were always the
source of truth, so the DB was an extra layer to keep in sync.)

```
run eval ──► re-evaluate ──► make_paper_figures.py
 (files)       (files)           (figures)
```

What the figures read:
- **Aggregate accuracy + search calls** (phrasing bars): the reevaluated baseline JSONs
  `results/musique-<phrasing>/musique-<phrasing>_baseline_<model>_run_1_reevaluated.json`,
  inner-joined formal ∩ natural2 on `example_id` per model
  (`src/results_files.py: paired_eval_files`).
- **Per-hop uncertainty** (taxonomy, cell-shift, calibration): the grader-reclustered
  uncertainty JSONs `results/musique_parametric/musique_parametric_uncertainty_<model>_grader.json`
  (`src/viz.py: CANONICAL_ENTROPY_JSONS` / `load_canonical_entropy`), overlaid on the
  per-hop `interplay_analysis/interplay_summary.csv` from
  `analyze_parametric_search_interplay.py`.

---

## Everyday loop

### 1. Run an evaluation (writes a file)
```bash
uv run python scripts/run_qa_eval_experiment.py \
    --agent_type baseline --model_name <model> --dataset <dataset> --run_name run_1
# → results/<dataset>/<dataset>_baseline_<model>_run_1.json
```

### 2. Re-grade with the shared judge (writes `*_reevaluated.json`)
Always grade both phrasings with the same judge so accuracy comparisons are fair.
```bash
uv run python scripts/re_evaluate_logs.py \
    "results/<dataset>/<dataset>_baseline_<model>_run_1.json" \
    --grader_provider ollama --grader_model gpt-oss:120b --template natural
# → ..._run_1_reevaluated.json
```

### 3. Make the figures
```bash
uv run python scripts/make_paper_figures.py --mode musique     # main paper set
uv run python scripts/make_paper_figures.py --mode natural2    # formal vs natural2
uv run python scripts/make_paper_figures.py --mode sharechat   # sharechat figures
```

---

## The `--mode natural2` figures (formal vs natural2)

A two-way comparison (MuSiQue **formal/benchmark** vs **natural2**) across all 3 models
(Gemini, Nemotron, Qwen):
- `phrasing_accuracy_natural2.png/pdf` — aggregate accuracy, Wilson CI, McNemar vs formal.
- `phrasing_searches_natural2.png/pdf` — mean searches/example, 95% CI, Wilcoxon vs formal.
- `phrasing_stats_natural2.csv` — the numbers behind both.
- the per-hop interplay figures (taxonomy, cell-shift, calibration, redundancy).

Defaults: output `results/natural2_paper_figures/`, grading `reevaluated`
(`--phrasing-grading original` to override), certain-rule `entropy` (semantic entropy == 0;
`--certain-rule joint` adds the all-runs-correct gate). The accuracy/search bars only include
examples present in **both** formal and natural2 (599 of 600).

---

## Model aliases

Model slugs vary across files (`qwen3.5:122b` vs `qwen3.5_122b`,
`gemini-3.1-pro-preview` vs `gemini-3-pro-preview`, `nemotron-3-nano:30b` vs
`nemotron-3-nano_30b`). `viz.canonical_model()` (+ `viz.MODEL_ALIASES`) folds punctuation
variants and the `gemini-3.1`→`gemini-3` / nemotron size aliases onto one canonical slug, so
the same model lines up across formal/natural2 in every figure. `paired_eval_files`
canonicalizes the raw slug parsed from each baseline filename, so it finds e.g. the
`gemini-3.1-pro-preview` natural2 file when asked for `gemini-3-pro-preview`.

---

## Command cheat-sheet

| Goal | Command |
|------|---------|
| Main paper figures | `uv run python scripts/make_paper_figures.py --mode musique` |
| Phrasing comparison (formal vs natural2) | `uv run python scripts/make_paper_figures.py --mode natural2` |
| ShareChat figures | `uv run python scripts/make_paper_figures.py --mode sharechat` |
| Ad-hoc accuracy query | `python -c "import sys;sys.path.insert(0,'.');from src import results_files as r,viz as v;print(r.paired_eval_files(['formal','natural2'], 'gemini-3-pro-preview').head())"` |

A one-time safety copy of the original files lives in `results.bak/` (delete it once you trust
the pipeline).
