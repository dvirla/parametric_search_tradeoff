# Results Workflow — how to run the scripts now

This project keeps results in **two layers**:

| Layer | What | Source of truth for |
|-------|------|---------------------|
| **Capture** | The JSON files under `results/` (eval logs, parametric, traces) written by the experiment runners. | The raw data. Never deleted or mutated. |
| **Analysis** | `results/results.db` (SQLite) built from those files by `scripts/ingest_results.py`. | Querying/plotting. **Rebuildable at any time** from the files. |

The golden rule: **runners write files; you ingest files into the DB; analysis/plots read the DB.**
The DB is disposable — if anything looks off, delete it and re-ingest.

```
run eval ──► re-evaluate ──► ingest_results.py ──► ingest_results.py --verify ──► make_paper_figures.py
 (files)       (files)            (DB)                  (sanity gate)                  (figures)
```

Derived "interplay" data (`results/<dataset>/interplay_analysis/`, produced by
`analyze_parametric_search_interplay.py`) is **still file-based** — the figure script reads those
CSVs directly. Only the *aggregate eval* numbers (accuracy, search calls) for the phrasing bar
charts come from the DB.

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

### 3. Ingest into the DB (idempotent, additive)
```bash
uv run python scripts/ingest_results.py            # loads new/changed files only
uv run python scripts/ingest_results.py --verify   # row-count + round-trip checks
```
- Re-running ingest only touches files whose mtime changed (keyed on `source_path`+`mtime`).
- `--dry-run` shows what *would* be ingested without writing.
- `--verify` must end with **`ALL CHECKS PASSED`**. It also prints an **identity** report
  (canonical model ← raw slugs) and a **coverage** report (files deferred to a later phase:
  traces, interplay CSVs).

### 4. Make the figures
```bash
uv run python scripts/make_paper_figures.py --mode musique     # main paper set
uv run python scripts/make_paper_figures.py --mode natural2    # 3-way phrasing comparison
uv run python scripts/make_paper_figures.py --mode sharechat   # sharechat figures
```

---

## The `--mode natural2` figures (formal vs natural vs natural2)

This mode produces, across **all 3 models** (Gemini, Nemotron, Qwen):
- `phrasing_accuracy_natural2.png/pdf` — aggregate accuracy, Wilson CI, McNemar vs formal.
- `phrasing_searches_natural2.png/pdf` — mean searches/example, 95% CI, Wilcoxon vs formal.
- `phrasing_stats_natural2.csv` — the numbers behind both.
- plus the per-hop interplay figures (taxonomy, cell-shift, calibration, redundancy).

Defaults: output `results/natural2_paper_figures/`, grading `reevaluated` (falls back to
`original` per run when no reevaluated file exists), reference phrasing `formal`. Override with
`--output-dir`, `--phrasing-grading original`, `--db <path>`.

The accuracy/search bars come from `results_db.paired_eval`, which inner-joins formal ∩ natural ∩
natural2 on `example_id` for each model — so they only include examples present in all three
phrasings. The old standalone `analyze_phrasing*.py` scripts are retired (in `scripts/archive/`);
this mode replaces them.

---

## Identity / model aliases

Model slugs vary across files (`qwen3.5:122b` vs `qwen3.5_122b`, `gemini-3.1-pro-preview` vs
`gemini-3-pro-preview`). The DB canonicalizes them. Punctuation variants (`:`↔`_`) are folded
automatically; genuinely different-looking names that are the **same model** are listed in
`_EXPLICIT_ALIASES` in `src/results_db.py`.

**If you add or change an alias, rebuild the DB** — aliasing is a global remap and ingest won't
re-key files it already loaded:
```bash
rm -f results/results.db results/results.db-wal results/results.db-shm
uv run python scripts/ingest_results.py
uv run python scripts/ingest_results.py --verify   # identity report should show the fold
```
Rebuilding is cheap (seconds) and lossless (the files are the source of truth).

When `--verify`'s identity report shows two lines that are actually one model, add the alias and
rebuild.

---

## Worked example: adding gemini-3.1 to the natural2 comparison

This is exactly the flow for "I ran gemini-3.1 on natural2, re-graded it, and ran interplay —
now compare all 3 models":

```bash
# (already done) eval + re-eval gemini-3.1 over natural2, then:
#   uv run python scripts/analyze_parametric_search_interplay.py ... (writes interplay_analysis/)

# 1. gemini-3.1 is the same model as gemini-3 → alias already in src/results_db.py:
#      "gemini-3.1-pro-preview": "gemini-3-pro-preview"
#    Because we changed identity, rebuild the DB:
rm -f results/results.db results/results.db-wal results/results.db-shm
uv run python scripts/ingest_results.py
uv run python scripts/ingest_results.py --verify
#    → identity shows:  gemini-3-pro-preview <- [gemini-3-pro-preview, gemini-3.1-pro-preview]

# 2. Make the 3-way figures:
uv run python scripts/make_paper_figures.py --mode natural2
#    → results/natural2_paper_figures/phrasing_accuracy_natural2.png
#    → results/natural2_paper_figures/phrasing_searches_natural2.png  (all 3 models)
```

---

## Command cheat-sheet

| Goal | Command |
|------|---------|
| What would ingest? | `uv run python scripts/ingest_results.py --dry-run` |
| Ingest new/changed files | `uv run python scripts/ingest_results.py` |
| Sanity-check the DB | `uv run python scripts/ingest_results.py --verify` |
| Rebuild after alias change | `rm -f results/results.db*` then ingest |
| Main paper figures | `uv run python scripts/make_paper_figures.py --mode musique` |
| Phrasing comparison | `uv run python scripts/make_paper_figures.py --mode natural2` |
| Ad-hoc query | `python -c "import sys;sys.path.insert(0,'.');from src import results_db as r;c=r.connect();print(r.load_eval(c, base_dataset='musique', phrasing='natural2').head())"` |

`results/results.db` is gitignored. A one-time safety copy of the original files lives in
`results.bak/` (delete it once you trust the pipeline).
