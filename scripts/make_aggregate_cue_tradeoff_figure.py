#!/usr/bin/env python3
"""Aggregate cue-tradeoff figures for the paper (replaces the dense
brief_combined_search_acc_{primary,secondary}.png model grids).

For each cue, computes the per-model (paired, vs that model's own PLAIN
baseline) % change in search calls and pp change in accuracy — 11
point estimates per cue, one per model. Accuracy is EM/regex-strict for
FRAMES and LLM-judge for MedQA (see GRADE_FIELD below and the "Grading"
note in the emitted brief_aggregate_tables.md -- MedQA's EM-vs-LLM-judge
gap is large enough, 26-36pp on `plain`, to make EM a misleading lower
bound specifically for that domain). Two renderings of the same
underlying per-model arrays:

  - mean:   bootstrap 95% CI of the cross-model mean; significance via a
            one-sample t-test per bar against 0.
  - median: bootstrap 95% CI of the cross-model median; significance via a
            one-sample Wilcoxon signed-rank test per bar against 0.

Each figure has its own family of 40 tests (2 metrics x 2 datasets x 10
bars incl. the PLAIN<->PLAIN noise-floor reference), corrected with
Benjamini-Hochberg FDR (q < 0.05/0.01/0.001 -> */**/***). The bootstrap
CIs plotted are the raw (uncorrected) per-comparison intervals; only the
significance stars reflect the multiple-comparison correction.

Outputs (default, no --exclude-models):
  results/cue_briefing/brief_aggregate_search_acc_mean.png
  results/cue_briefing/brief_aggregate_search_acc_median.png

Pass --exclude-models to run a leave-N-out sensitivity check; outputs get a
"_excl_<slug>" suffix instead of overwriting the full-roster figures, e.g.:
  uv run python scripts/make_aggregate_cue_tradeoff_figure.py --exclude-models nemotron-3-nano_30b

--datasets selects which panels are rendered (default "FRAMES,MedQA" -- the
original two-dataset behaviour, byte-identical). "HotpotQA" adds the HotpotQA
cue grid as a third dataset. Run it in its OWN invocation, into its own
--output-dir: the BH-FDR correction is defined over whatever panels are loaded,
so folding HotpotQA into the FRAMES/MedQA call would silently change the
FRAMES/MedQA stars in the paper figures. E.g.:
  uv run python scripts/make_aggregate_cue_tradeoff_figure.py \
      --datasets HotpotQA --output-dir results/hotpotqa_cue_briefing
"""
import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

ap = argparse.ArgumentParser()
ap.add_argument("--exclude-models", default="", help="comma-separated model slugs to drop from MODEL_ORDER")
ap.add_argument("--datasets", default="FRAMES,MedQA",
                help="comma-separated datasets to render panels for: FRAMES, MedQA, HotpotQA. "
                     "Each invocation is its own BH-FDR family -- see module docstring.")
ap.add_argument("--output-dir", default=None,
                help="where the figures/tables land (default results/cue_briefing)")
args = ap.parse_args()
EXCLUDE = [m.strip() for m in args.exclude_models.split(",") if m.strip()]

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = args.output_dir or os.path.join(ROOT, "results", "cue_briefing")
if not os.path.isabs(OUT):
    OUT = os.path.join(ROOT, OUT)
os.makedirs(OUT, exist_ok=True)

KNOWN_DATASETS = ["FRAMES", "MedQA", "HotpotQA"]
DATASETS = [d.strip() for d in args.datasets.split(",") if d.strip()]
for _d in DATASETS:
    if _d not in KNOWN_DATASETS:
        raise SystemExit(f"--datasets: unknown dataset {_d!r}, expected one of {KNOWN_DATASETS}")

MODEL_ORDER = [
    "gemini-3.1-pro-preview", "gemini-3.5-flash", "qwen3.5_122b", "qwen3.5_35b", "qwen3.5_4b",
    "gemma4_31b", "gemma4_e4b", "nemotron-3-nano_30b", "nemotron-cascade-2_30b",
    "gpt-oss_120b", "gpt-oss_20b",
]
for _m in EXCLUDE:
    if _m not in MODEL_ORDER:
        raise SystemExit(f"--exclude-models: unknown model slug {_m!r}, not in MODEL_ORDER")
MODEL_ORDER = [m for m in MODEL_ORDER if m not in EXCLUDE]
N_MODELS = len(MODEL_ORDER)
SUFFIX = ("_excl_" + "_".join(EXCLUDE)) if EXCLUDE else ""

# MedQA's multiturn/searchmulti/confident_parametric were never LLM-graded for any
# model at collection time (sampler_correct is None on 100% of rows, all 11 models
# -- confirmed directly). A targeted backfill grading pass
# (scripts/regrade_medqa_conversation_cues_llm.py + apply_medqa_conversation_cue_grades.py)
# covers only the 6 models this session's parametric-uncertainty/entropy work
# centers on -- NOT the full roster. Rather than mix metrics within one aggregate
# (some models LLM-graded, some not, for the same bar) or silently drop the other
# 5 models, MedQA renders as TWO SEPARATE panels: the 6-model subset under the LLM
# judge (all 9 conditions), and the remaining 5 under EM (unchanged from before this
# session -- these 5 never had their MedQA grading touched at all).
MEDQA_LLM_MODELS = [m for m in
                     ["gemma4_31b", "gpt-oss_20b", "gpt-oss_120b", "nemotron-3-nano_30b",
                      "nemotron-cascade-2_30b", "qwen3.5_122b"]
                     if m in MODEL_ORDER]
MEDQA_REGEX_MODELS = [m for m in MODEL_ORDER if m not in MEDQA_LLM_MODELS]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c


# AgentAsSampler.acall() counts search calls over pydantic-ai's all_messages(),
# which includes the injected message_history -- so raw sampler_search_calls for
# these history-injected cues is inflated by exactly this many FAKE search calls
# from the mocked history itself (see analyze_necessity_vs_template_search_5run.py
# for the full explanation). Subtract before any downstream stat reads search_calls,
# including the zero-search-frequency table that feeds the paper's Table 1.
MOCK_HISTORY_OFFSET = {"searchmulti": 1, "searchmulti2": 2, "searchmulti3": 3}


def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    offset = d["cue"].map(MOCK_HISTORY_OFFSET).fillna(0)
    d["search_calls"] = (d["search_calls"] - offset).clip(lower=0)
    return d[["model", "phrasing", "cue", "example_id", "search_calls"]]


# ---------------------------------------------------------------------------
# HotpotQA loaders.
#
# HotpotQA does NOT go through the joined_tokens.csv / regrade_regex path the
# other two datasets use, for two reasons that are properties of the run, not
# choices made here:
#   * No Logfire traces were downloaded for the HotpotQA grid, so there is no
#     joined_tokens.csv (that join is what supplies thinking tokens; the
#     aggregate figures only need search calls, which live in the eval rows).
#   * Every HotpotQA row was collected with --no_grader, so `sampler_correct`
#     is None everywhere and there is no LLM judge to read. Correctness is
#     decided offline by scripts/grade_hotpotqa_regex.py, which reuses
#     regrade_regex.py's exact match functions -- so HotpotQA EM is the same
#     grader as the FRAMES panel's EM, just computed in a different script.
# Both search calls and EM verdicts therefore come from that script's
# per_row.csv, which is also where the searchmulti mocked-history correction
# has already been applied (`search_calls` is post-correction; `search_calls_raw`
# is not) -- do NOT re-apply MOCK_HISTORY_OFFSET on top of it.
HOTPOTQA_PER_ROW = os.path.join(ROOT, "results/hotpotqa_cue_grid_regex/per_row.csv")

# The HotpotQA grid has a single phrasing (the dataset's own question text); it
# was never re-run under the FRAMES/MedQA "terse" rewrite. `orig` is a
# placeholder so the shared (phrasing, cue) plumbing works unchanged; the TERSE
# bar consequently has no data and renders as an empty slot on this panel.
HOTPOTQA_PHRASING = "orig"

# The FRAMES cue-robustness SFT checkpoint is present in results/hotpotqa_cue_grid/
# as the out-of-domain transfer arm. It is deliberately trained to be cue-invariant,
# so averaging it into a "do cues move search?" roster would dilute the very effect
# being measured. It is excluded here and analysed separately.
HOTPOTQA_EXCLUDE_SLUGS = {"gemma4-frames-robust-q4km_latest"}


def _load_hotpotqa_per_row():
    if not os.path.exists(HOTPOTQA_PER_ROW):
        raise SystemExit(
            f"missing {HOTPOTQA_PER_ROW}\n"
            "Build it first:  uv run python scripts/grade_hotpotqa_regex.py "
            "--results-root results/hotpotqa_cue_grid")
    d = pd.read_csv(HOTPOTQA_PER_ROW)
    d = d[~d.model.isin(HOTPOTQA_EXCLUDE_SLUGS)]
    # Same exclusion grade_hotpotqa_regex.py applies to its own aggregates: rows
    # with stop_reason set are salvaged best-effort answers from a run that hit the
    # agent loop cap. Rare (1 row in 29,700) but extreme -- that row had
    # search_calls=100 and single-handedly moved qwen3.5:4b `natural` by 0.33 calls,
    # more than most models' entire run-to-run floor.
    d = d[d.stop_reason.isna() | (d.stop_reason.astype(str).str.strip() == "")]
    d["cue"] = d.run_name.astype(str)
    d["phrasing"] = HOTPOTQA_PHRASING
    return d


def load_tokens_hotpotqa(per_row):
    # `search_calls` is already history-corrected by grade_hotpotqa_regex.py.
    t = per_row[per_row.cue != "plain_rep2"].copy()
    return t[["model", "phrasing", "cue", "example_id", "search_calls"]]


def load_graded_hotpotqa(per_row):
    # Headline accuracy is EM over NON-boolean rows only. On the ~4.7% of
    # hotpotqa-300 whose gold is literally "yes"/"no", substring matching is
    # meaningless ("no" occurs constantly in prose), so grade_hotpotqa_regex.py
    # scores them with a separate first-standalone-yes/no-token matcher and keeps
    # them out of the headline. Same convention here; dropping them symmetrically
    # from both sides of every paired comparison keeps the pairing intact.
    g = per_row[(~per_row.answer_is_boolean.astype(bool)) & (per_row.cue != "plain_rep2")].copy()
    g["regex"] = g["strict"].astype(int)
    return g[["model", "phrasing", "cue", "example_id", "regex"]]


def load_rerun_hotpotqa(per_row):
    # HotpotQA's noise floor is the `plain_rep2` replicate inside the SAME grid
    # directory (the other two datasets keep theirs in a separate _rerun tree).
    # Unlike FRAMES/MedQA, its accuracy IS like-for-like with the panel's own bars:
    # both are EM from the same grader, since HotpotQA has no LLM judge anywhere.
    r = per_row[(per_row.cue == "plain_rep2") & (~per_row.answer_is_boolean.astype(bool))].copy()
    r["regex"] = r["strict"].astype(int)
    return r[["model", "example_id", "regex", "search_calls"]]


_TOK_SOURCES = {
    "FRAMES": lambda: load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
    "MedQA": lambda: load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv")),
    "HotpotQA": lambda: load_tokens_hotpotqa(HOTPOTQA_ROWS),
}
HOTPOTQA_ROWS = _load_hotpotqa_per_row() if "HotpotQA" in DATASETS else None
HOTPOTQA_MODELS = ([m for m in MODEL_ORDER if m in set(HOTPOTQA_ROWS.model)]
                   if HOTPOTQA_ROWS is not None else [])
TOK = {ds: _TOK_SOURCES[ds]() for ds in DATASETS}

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)

# The perturbation-condition bars use EM/regex-strict for FRAMES (unchanged, the
# paper's deliberate strict-lower-bound primary metric) but the LLM judge for
# MedQA: MedQA's `correct_answer` is often full multiple-choice option text that
# models restate without reproducing verbatim, which EM cannot credit but an
# options-aware LLM judge (Gemini 3 Flash, `sampler_correct` -- already present
# in every row, no new grading run needed) correctly recognizes. A targeted audit
# found EM undercounts MedQA accuracy by 26-36pp vs. the LLM judge on `plain`,
# vs. only 7-11pp on FRAMES -- large enough to make EM a misleading lower bound
# specifically for this domain. See accuracy_revision.md caveat 8 and Appendix C
# ("Dual-Metric Robustness Check", Level Agreement: Pearson r=0.86 MedQA vs. 0.96
# FRAMES -- this looser agreement is the same gap, seen from the other side).
# PANELS: one entry per rendered aggregate panel. FRAMES is a single panel over the
# full 11-model roster (unchanged). MedQA is split into two panels -- see the
# MEDQA_LLM_MODELS/MEDQA_REGEX_MODELS comment above for why -- each with its own
# fixed model list and grading field, so every bar within one panel is on a
# consistent metric and a consistent N, rather than mixing per-condition coverage.
ALL_PANELS = [
    dict(key="FRAMES", ds="FRAMES", raw_ds="frames", label="FRAMES",
         models=MODEL_ORDER, grade_field="regex_strict", grade_label="EM"),
    dict(key="MedQA_llm6", ds="MedQA", raw_ds="medqa", label="MedQA (6-model uncertainty subset, LLM-judge)",
         models=MEDQA_LLM_MODELS, grade_field="llm_correct", grade_label="LLM-judge"),
    dict(key="MedQA_regex5", ds="MedQA", raw_ds="medqa", label="MedQA (remaining 5 models, EM)",
         models=MEDQA_REGEX_MODELS, grade_field="regex_strict", grade_label="EM"),
    # HotpotQA is a single panel: there is no LLM judge for it at all (every row was
    # collected with --no_grader), so no metric split is possible or needed. Its
    # roster is whatever subset of MODEL_ORDER was actually run -- the two Gemini
    # models were not, so this is a 9-model open-weights roster, not 11.
    dict(key="HotpotQA", ds="HotpotQA", raw_ds="hotpotqa",
         label="HotpotQA (open-weights roster, EM)",
         models=HOTPOTQA_MODELS, grade_field="regex_strict", grade_label="EM"),
]
PANELS = [p for p in ALL_PANELS if p["ds"] in DATASETS]
if not PANELS:
    raise SystemExit(f"no panels for --datasets {DATASETS}")
GRADE_LABEL = {p["key"]: p["grade_label"] for p in PANELS}


def load_graded(dataset, grid_dir, grade_field):
    # When grade_field == "llm_correct", an ungraded row has g[grade_field] is None,
    # NOT False -- coercing via bool() would silently count "never graded" as
    # "incorrect", which corrupts any condition/model not yet LLM-graded (found:
    # MedQA's multiturn/searchmulti/confident_parametric were 100% ungraded for
    # all 11 models before a targeted 6-model backfill; the other 5 models remain
    # ungraded there by design -- see PAPER_REFACTOR_PLAN.md). Drop, don't coerce.
    rows = []
    for path, slug, cond in _RG.find_grid_files(grid_dir, dataset):
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            if g[grade_field] is None:
                continue
            rows.append({"model": slug, "condition": cond, "example_id": g["example_id"],
                         "regex": int(bool(g[grade_field]))})
    df = pd.DataFrame(rows)
    df["cue"] = df["condition"].map(base_cue)

    def get_ph(c):
        if c.startswith("epi_strong_"):
            return "epi_strong"
        return c.split("_", 1)[0]

    df["phrasing"] = df["condition"].map(get_ph)
    return df


GRID_DIR_FOR = {"FRAMES": os.path.join(ROOT, "results/frames_cues_full"),
                "MedQA": os.path.join(ROOT, "results/medqa_grid")}
GRADED = {p["key"]: (load_graded_hotpotqa(HOTPOTQA_ROWS) if p["ds"] == "HotpotQA"
                     else load_graded(p["raw_ds"], GRID_DIR_FOR[p["ds"]], p["grade_field"]))
          for p in PANELS}

RERUN_MIN_N = 100


def load_rerun(dataset, rerun_dir):
    # Always regex-graded, for BOTH datasets, regardless of GRADE_FIELD above:
    # results/medqa_grid_rerun/ was never LLM-graded (sampler_correct is None on
    # all 5,489 rows, every model -- confirmed directly, not assumed). Silently
    # falling back to bool(None)==False here would fabricate a fake ~0% MedQA
    # noise-floor accuracy bar. This bar's accuracy is a secondary reference
    # (its main job is the search-call noise floor, which doesn't depend on
    # grading), so it stays regex-graded on both datasets rather than triggering
    # a new LLM-grading run for this figure alone -- flag this asymmetry in the
    # write-up rather than fix it by launching an ungated grading job.
    rows = []
    if not os.path.isdir(rerun_dir):
        return pd.DataFrame(columns=["model", "example_id", "regex", "search_calls"])
    for path, slug, cond in _RG.find_grid_files(rerun_dir, dataset):
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({"model": slug, "example_id": g["example_id"],
                         "regex": int(bool(g["regex_strict"])),
                         "search_calls": r.get("sampler_search_calls", 0)})
    return pd.DataFrame(rows)


_RERUN_SOURCES = {
    "FRAMES": lambda: load_rerun("frames", os.path.join(ROOT, "results/frames_cues_rerun")),
    "MedQA": lambda: load_rerun("medqa", os.path.join(ROOT, "results/medqa_grid_rerun")),
    "HotpotQA": lambda: load_rerun_hotpotqa(HOTPOTQA_ROWS),
}
RERUN = {ds: _RERUN_SOURCES[ds]() for ds in DATASETS}


def get_conditions(ds):
    base_ph = "verbose" if ds == "FRAMES" else "orig"
    if ds == "HotpotQA":
        base_ph = HOTPOTQA_PHRASING
    # Order follows the paper's taxonomy: Style (terse, polite), Conversation
    # State (general multiturn, search multiturn), Directives (short,
    # detailed, direct, structured, capability-framing).
    conds = [
        ("terse", "plain"),
        (base_ph, "polite"),
        (base_ph, "multiturn"),
        (base_ph, "searchmulti"),
        (base_ph, "natural"),
        (base_ph, "elaborate"),
        (base_ph, "direct"),
        (base_ph, "query"),
        (base_ph, "confident_parametric"),
    ]
    return base_ph, conds


def get_label(ph, cue):
    if cue == "natural":
        return "SHORT"
    if cue == "confident_parametric":
        return "CONFIDENT"
    if cue == "plain":
        return "TERSE"
    if cue == "searchmulti":
        return "SEARCH\nMULTITURN"
    return cue.upper()


def search_pct_change(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return np.nan
    mb = j["base"].mean()
    return 100.0 * (j["target"].mean() - mb) / mb if mb > 0 else np.nan


def acc_pp_change(gdf, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = gdf[gdf.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "regex"]].rename(columns={"regex": "c"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "regex"]].rename(columns={"regex": "p"})
    if t.empty or b.empty:
        return np.nan
    j = t.merge(b, on="example_id")
    if len(j) < 5:
        return np.nan
    return 100.0 * (j["c"].mean() - j["p"].mean())


def rerun_search_pct(ds, m, base_ph, base_cue="plain"):
    tok = TOK[ds]; rr = RERUN[ds]
    b = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    t = rr[rr.model == m][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    j = b.merge(t, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan
    mb = j["base"].mean()
    return 100.0 * (j["target"].mean() - mb) / mb if mb > 0 else np.nan


def rerun_acc_pp(ds, gdf, m, base_ph, base_cue="plain"):
    rr = RERUN[ds]
    b = gdf[(gdf.model == m) & (gdf.phrasing == base_ph) & (gdf.cue == base_cue)][["example_id", "regex"]].rename(columns={"regex": "p"})
    t = rr[rr.model == m][["example_id", "regex"]].rename(columns={"regex": "c"})
    if b.empty or t.empty:
        return np.nan
    j = t.merge(b, on="example_id")
    if len(j) < RERUN_MIN_N:
        return np.nan
    return 100.0 * (j["c"].mean() - j["p"].mean())


def zero_search_pp_change(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    """pp change in P(search_calls == 0), paired by example_id."""
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "t"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "b"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return np.nan
    return 100.0 * ((j["t"] == 0).mean() - (j["b"] == 0).mean())


def rerun_zero_search_pp(ds, m, base_ph, base_cue="plain"):
    tok = TOK[ds]; rr = RERUN[ds]
    b = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "b"})
    t = rr[rr.model == m][["example_id", "search_calls"]].rename(columns={"search_calls": "t"})
    j = b.merge(t, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan
    return 100.0 * ((j["t"] == 0).mean() - (j["b"] == 0).mean())


def example_level_spearman(tok, gdf, m, target_ph, target_cue, base_ph, base_cue="plain"):
    """Spearman r between per-example (Δ search calls, Δ regex-correct), paired vs PLAIN."""
    ts = tok[(tok.model == m) & (tok.phrasing == target_ph) & (tok.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "s_t"})
    bs = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "s_b"})
    ta = gdf[(gdf.model == m) & (gdf.phrasing == target_ph) & (gdf.cue == target_cue)][["example_id", "regex"]].rename(columns={"regex": "a_t"})
    ba = gdf[(gdf.model == m) & (gdf.phrasing == base_ph) & (gdf.cue == base_cue)][["example_id", "regex"]].rename(columns={"regex": "a_b"})
    j = ts.merge(bs, on="example_id").merge(ta, on="example_id").merge(ba, on="example_id").dropna()
    if len(j) < 10:
        return np.nan
    d_search = j["s_t"] - j["s_b"]
    d_acc = j["a_t"] - j["a_b"]
    if d_search.nunique() < 2 or d_acc.nunique() < 2:
        return np.nan
    r, _ = stats.spearmanr(d_search, d_acc)
    return float(r)


def rerun_example_level_spearman(ds, gdf, m, base_ph, base_cue="plain"):
    tok, rr = TOK[ds], RERUN[ds]
    bs = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "s_b"})
    ba = gdf[(gdf.model == m) & (gdf.phrasing == base_ph) & (gdf.cue == base_cue)][["example_id", "regex"]].rename(columns={"regex": "a_b"})
    rr_m = rr[rr.model == m][["example_id", "search_calls", "regex"]].rename(columns={"search_calls": "s_t", "regex": "a_t"})
    j = bs.merge(ba, on="example_id").merge(rr_m, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan
    d_search = j["s_t"] - j["s_b"]
    d_acc = j["a_t"] - j["a_b"]
    if d_search.nunique() < 2 or d_acc.nunique() < 2:
        return np.nan
    r, _ = stats.spearmanr(d_search, d_acc)
    return float(r)


def clean(vals):
    return np.array([v for v in vals if not np.isnan(v)])


def point_estimate(vals, stat):
    vals = clean(vals)
    if len(vals) == 0:
        return np.nan
    return float(stat(vals))


def raw_pvalue(vals, method):
    vals = clean(vals)
    n = len(vals)
    if n < 2:
        return np.nan
    if method == "ttest":
        if np.allclose(vals, vals[0]):
            return 1.0 if np.isclose(vals[0], 0) else 0.0
        return float(stats.ttest_1samp(vals, 0.0).pvalue)
    else:  # wilcoxon signed-rank
        if np.allclose(vals, 0):
            return 1.0
        try:
            return float(stats.wilcoxon(vals).pvalue)
        except ValueError:
            return 1.0


def stars(q):
    if q is None or (isinstance(q, float) and np.isnan(q)):
        return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


# ===========================================================================
# Single data-collection pass: raw per-model delta arrays for every
# (dataset, bar, metric). Both figures render from these same arrays.
# ===========================================================================
# Taxonomy matches the paper's condition write-up: Variation Measurement
# (rerun noise floor), Style (terse/polite), Conversation State (general
# multiturn / search multiturn), Directives (short/detailed/direct/
# structured/capability-framing — confident_parametric is a Directive, not
# a separate category, in the paper's framing).
GROUP_COLOR = {
    "Variation Measurement": "#9e9e9e",
    "Style": "#5aae61",
    "Conversation State": "#e08214",
    "Directives": "#0571b0",
}
CUE_GROUP = {
    "REFERENCE": "Variation Measurement",
    "polite": "Style", "plain": "Style",
    "multiturn": "Conversation State", "searchmulti": "Conversation State",
    "natural": "Directives", "elaborate": "Directives",
    "query": "Directives", "direct": "Directives", "confident_parametric": "Directives",
}
CUE_COLOR = {k: GROUP_COLOR[v] for k, v in CUE_GROUP.items()}

PANEL_KEYS = [p["key"] for p in PANELS]
N_MODELS_BY_PANEL = {p["key"]: len(p["models"]) for p in PANELS}
PANEL_LABEL = {p["key"]: p["label"] for p in PANELS}
DATA = {}  # panel key -> {"labels":[...], "colors":[...], "search":[vals_array,...], "acc":[vals_array,...]}

for panel in PANELS:
    key, ds, models = panel["key"], panel["ds"], panel["models"]
    tok, gdf = TOK[ds], GRADED[key]
    base_ph, conds = get_conditions(ds)

    labels = ["RERUN\n(noise floor)"]
    colors = [CUE_COLOR["REFERENCE"]]
    groups = [CUE_GROUP["REFERENCE"]]
    search_arrs = [np.array([rerun_search_pct(ds, m, base_ph) for m in models])]
    acc_arrs = [np.array([rerun_acc_pp(ds, gdf, m, base_ph) for m in models])]
    zero_search_arrs = [np.array([rerun_zero_search_pp(ds, m, base_ph) for m in models])]
    corr_arrs = [np.array([rerun_example_level_spearman(ds, gdf, m, base_ph) for m in models])]

    for (ph, cue) in conds:
        labels.append(get_label(ph, cue))
        colors.append(CUE_COLOR.get(cue, "#666"))
        groups.append(CUE_GROUP.get(cue, "Directives"))
        search_arrs.append(np.array([search_pct_change(tok, m, ph, cue, base_ph) for m in models]))
        acc_arrs.append(np.array([acc_pp_change(gdf, m, ph, cue, base_ph) for m in models]))
        zero_search_arrs.append(np.array([zero_search_pp_change(tok, m, ph, cue, base_ph) for m in models]))
        corr_arrs.append(np.array([example_level_spearman(tok, gdf, m, ph, cue, base_ph) for m in models]))

    DATA[key] = dict(labels=labels, colors=colors, groups=groups, base_ph=base_ph, search=search_arrs, acc=acc_arrs,
                      zero_search=zero_search_arrs, corr=corr_arrs)

N_BARS = len(DATA[PANEL_KEYS[0]]["labels"])  # 10 (noise floor + 9 cues)


def n_models_contributing(arr):
    return int(np.sum(~np.isnan(arr)))


def family_qmap(metric, test_method):
    """BH-FDR corrected q-values across the combined N_PANELS-panel x N_BARS family for one metric."""
    pval_index, pvals = [], []
    for ds in PANEL_KEYS:
        for bi, arr in enumerate(DATA[ds][metric]):
            pval_index.append((ds, bi))
            pvals.append(raw_pvalue(arr, test_method))
    pvals = np.array(pvals)
    valid = ~np.isnan(pvals)
    qvals = np.full_like(pvals, np.nan)
    if valid.sum() > 0:
        _, q_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        qvals[valid] = q_valid
    return {idx: q for idx, q in zip(pval_index, qvals)}


def render(estimator_name, stat_fn, test_method, name_stub):
    """estimator_name in {'mean','median'}; test_method in {'ttest','wilcoxon'}.
    Saves one figure per panel: {name_stub}_{panel_key}{SUFFIX}.png"""
    # --- collect raw p-values across the full family: 2 metrics x N_PANELS panels x N_BARS bars ---
    # (kept as one combined family so the correction is defined over the full analysis,
    # even though each panel is now rendered as its own image.)
    pval_index = []  # (ds, metric, bar_idx)
    pvals = []
    for ds in PANEL_KEYS:
        for metric in ["search", "acc"]:
            for bi, arr in enumerate(DATA[ds][metric]):
                pval_index.append((ds, metric, bi))
                pvals.append(raw_pvalue(arr, test_method))
    pvals = np.array(pvals)
    valid = ~np.isnan(pvals)
    qvals = np.full_like(pvals, np.nan)
    if valid.sum() > 0:
        _, q_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        qvals[valid] = q_valid
    qmap = {idx: q for idx, q in zip(pval_index, qvals)}

    for ds in PANEL_KEYS:
        d = DATA[ds]
        labels, colors = d["labels"], d["colors"]
        xb = np.arange(len(labels))

        fig, axes = plt.subplots(2, 1, figsize=(8, 8.6), constrained_layout=True, sharex=True)

        for row, metric, ylabel, fmt in [
            (0, "search", f"Δ Search calls (%)\n{estimator_name}", "{:+.0f}%"),
            (1, "acc", f"Δ {GRADE_LABEL[ds]} accuracy (pp)\n{estimator_name}", "{:+.1f}"),
        ]:
            ax = axes[row]
            pts = [point_estimate(arr, stat_fn) for arr in d[metric]]
            bars = ax.bar(xb, pts, 0.62, color=colors)
            bars[0].set_hatch("//"); bars[0].set_edgecolor("#444")
            finite = [v for v in pts if not np.isnan(v)]
            pad = 0.05 * (max(finite) - min(finite) or 1) if finite else 1
            for xi, val in zip(xb, pts):
                if np.isnan(val):
                    continue
                q = qmap.get((ds, metric, xi))
                label = fmt.format(val) + stars(q)
                # label sits just outside the bar's far tip (away from zero),
                # never between zero and the tip, so it never overlaps the fill
                y = val + pad if val >= 0 else val - pad
                va = "bottom" if val >= 0 else "top"
                ax.text(xi, y, label, ha="center", va=va, fontsize=9, fontweight="bold")
            ax.axhline(0, color="#333", lw=0.9)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#eee", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.margins(y=0.15)
            if row == 0:
                ax.set_title(PANEL_LABEL[ds], fontsize=11)
            else:
                ax.set_xticks(xb)
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
            ax.set_ylabel(ylabel)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
        fig.legend(handles, GROUP_COLOR.keys(), loc="lower center", ncol=2, fontsize=9.5,
                   bbox_to_anchor=(0.5, -0.09), frameon=False)

        out_path = os.path.join(OUT, f"{name_stub}_{ds}{SUFFIX}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")


DATASET_COLOR = {"FRAMES": "#2c7fb8", "MedQA_llm6": "#d95f02", "MedQA_regex5": "#e7a55c",
                 "HotpotQA": "#5aae61"}


def render_zero_search_dotplot(metric, stat_fn, test_method, value_fmt, out_name, xlabel):
    """Condensed single-panel Cleveland dot plot: one row per cue, one dot per
    panel, connected by a dumbbell line (spanning min-to-max across however
    many panels have a finite point). Replaces the 2x4 zero-search bar
    grid for a compact, trend-at-a-glance paper figure."""
    qmap = family_qmap(metric, test_method)
    labels = DATA[PANEL_KEYS[0]]["labels"]
    groups = DATA[PANEL_KEYS[0]]["groups"]
    n = len(labels)
    y = np.arange(n)[::-1]  # RERUN at top, CONFIDENT at bottom, reading order

    fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.5), constrained_layout=True)

    for bi in range(n):
        pts = {}
        for ds in PANEL_KEYS:
            v = point_estimate(DATA[ds][metric][bi], stat_fn)
            pts[ds] = v
        finite = [v for v in pts.values() if not np.isnan(v)]
        if len(finite) >= 2:
            ax.plot([min(finite), max(finite)], [y[bi], y[bi]],
                     color="#bbb", lw=1.5, zorder=1)
        for ds in PANEL_KEYS:
            v = pts[ds]
            if np.isnan(v):
                continue
            q = qmap.get((ds, bi))
            sig = q is not None and not np.isnan(q) and q < 0.05
            ax.scatter([v], [y[bi]], s=70, color=DATASET_COLOR[ds],
                       edgecolors=DATASET_COLOR[ds], linewidths=1.5,
                       facecolors=DATASET_COLOR[ds] if sig else "white", zorder=3)

    ax.axvline(0, color="#333", lw=0.9, zorder=0)
    group_boundaries = [i for i in range(1, n) if groups[i] != groups[i - 1]]
    for gb in group_boundaries:
        ax.axhline(y[gb] + 0.5, color="gray", linestyle="--", lw=1.0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl.replace("\n", " ") for lbl in labels], fontsize=10)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.margins(y=0.02)

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DATASET_COLOR[ds],
                           markeredgecolor=DATASET_COLOR[ds], markersize=9, label=PANEL_LABEL[ds]) for ds in PANEL_KEYS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                               markeredgecolor="#333", markersize=9, label="not significant (q≥.05)"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    out_path = os.path.join(OUT, f"{out_name}{SUFFIX}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


render("mean", np.mean, "ttest", "brief_aggregate_search_acc_mean")
render("median", np.median, "wilcoxon", "brief_aggregate_search_acc_median")
render_zero_search_dotplot("zero_search", np.mean, "ttest", "{:+.0f}pp",
                            "brief_zero_search_dotplot", "Δ zero-search rate (pp) — mean, filled = q<.05 (BH-FDR)")

# remove figures superseded by earlier versions of this script (single-estimator,
# then combined-dataset 2x2 layouts)
stale_names = ["brief_aggregate_search_acc_mean.png", "brief_aggregate_search_acc_median.png"]
if not EXCLUDE:
    stale_names.append("brief_aggregate_search_acc.png")
for stale in stale_names:
    base, ext = os.path.splitext(stale)
    p = os.path.join(OUT, f"{base}{SUFFIX}{ext}")
    if os.path.exists(p):
        os.remove(p)
        print(f"Removed superseded {p}")


# ===========================================================================
# Markdown tables (paper-friendly alternative to the zero-search and
# accuracy/search-correlation PNG grids): zero-search suppression and
# example-level Spearman correlation, aggregated across models the same way
# as the mean/median bar figures above.
# ===========================================================================
def make_condensed_dataset_table(metric, value_fmt, estimator_name, stat_fn, test_method):
    """One row per cue, one column per panel (not per estimator) — the
    condensed alternative to make_md_table for metrics where mean/median
    agree in direction and a full (panel x estimator) breakdown is
    more detail than the headline trend needs."""
    qmap = family_qmap(metric, test_method)
    all_full = all(
        n_models_contributing(DATA[ds][metric][bi]) == N_MODELS_BY_PANEL[ds]
        for ds in PANEL_KEYS for bi in range(len(DATA[ds][metric]))
    )
    header_cols = ["Cue", "Group"] + [PANEL_LABEL[ds] for ds in PANEL_KEYS] + \
        ([] if all_full else [f"{PANEL_LABEL[ds]} N" for ds in PANEL_KEYS])
    n_desc = "/".join(f"{PANEL_LABEL[ds]}={N_MODELS_BY_PANEL[ds]}" for ds in PANEL_KEYS)
    lines = [f"({estimator_name}, N models: {n_desc}" + (")" if all_full else " unless noted)"), "",
             "| " + " | ".join(header_cols) + " |",
             "|" + "---|" * len(header_cols)]
    ref_labels = DATA[PANEL_KEYS[0]]["labels"]
    ref_groups = DATA[PANEL_KEYS[0]]["groups"]
    for bi, (label, group) in enumerate(zip(ref_labels, ref_groups)):
        label_clean = label.replace("\n", " ")
        cells = [label_clean, group]
        for ds in PANEL_KEYS:
            arr = DATA[ds][metric][bi]
            v = point_estimate(arr, stat_fn)
            cells.append((value_fmt.format(v) + stars(qmap.get((ds, bi)))) if not np.isnan(v) else "—")
        if not all_full:
            for ds in PANEL_KEYS:
                cells.append(f"{n_models_contributing(DATA[ds][metric][bi])}/{N_MODELS_BY_PANEL[ds]}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def make_md_table(metric, value_fmt):
    qmap_mean = family_qmap(metric, "ttest")
    qmap_median = family_qmap(metric, "wilcoxon")
    lines = []
    for ds in PANEL_KEYS:
        d = DATA[ds]
        lines.append(f"### {PANEL_LABEL[ds]}\n")
        lines.append("| Cue | Group | Mean | Median | N models |")
        lines.append("|---|---|---|---|---|")
        for bi, (label, group) in enumerate(zip(d["labels"], d["groups"])):
            arr = d[metric][bi]
            mean_v = point_estimate(arr, np.mean)
            median_v = point_estimate(arr, np.median)
            mean_s = (value_fmt.format(mean_v) + stars(qmap_mean.get((ds, bi)))) if not np.isnan(mean_v) else "—"
            median_s = (value_fmt.format(median_v) + stars(qmap_median.get((ds, bi)))) if not np.isnan(median_v) else "—"
            label_clean = label.replace("\n", " ")
            lines.append(f"| {label_clean} | {group} | {mean_s} | {median_s} | {n_models_contributing(arr)}/{N_MODELS_BY_PANEL[ds]} |")
        lines.append("")
    return "\n".join(lines)


# The panels-and-grading note is MedQA-specific (it exists to explain why MedQA is
# split into an LLM-judge panel and an EM panel). Emit it only when a MedQA panel is
# actually being rendered, and give HotpotQA its own note instead of silently
# inheriting a paragraph about datasets that are not in the table.
def MEDQA_GRADING_NOTE():
    # f-strings inside reference PANEL_LABEL['MedQA_*'] / MEDQA_*_MODELS, which only
    # exist when a MedQA panel is loaded -- hence a function, evaluated on demand.
    return (
        "**Panels and grading:** FRAMES accuracy uses SQuAD-style EM (`regex_strict`), the paper's "
    "primary metric, across all 11 models. MedQA is split into TWO panels rather than one, because "
    "MedQA's `multiturn`/`searchmulti`/`confident_parametric` conditions were never LLM-graded for "
    "any model at collection time (`sampler_correct` was `None` on 100% of rows, all 11 models), "
    "unlike the other 6 MedQA conditions (~92% graded already): "
    f"**{PANEL_LABEL['MedQA_llm6']}** ({N_MODELS_BY_PANEL['MedQA_llm6']} models: "
    f"{', '.join(MEDQA_LLM_MODELS)}) uses the LLM judge (`sampler_correct`, Gemini 3 Flash) for all "
    "9 conditions, after a targeted backfill grading pass for just these models' 3 previously-"
    f"ungraded conditions; **{PANEL_LABEL['MedQA_regex5']}** ({N_MODELS_BY_PANEL['MedQA_regex5']} "
    f"models: {', '.join(MEDQA_REGEX_MODELS)}) stays on EM throughout, unchanged from before this "
    "session. Splitting into two model-disjoint panels (rather than one mixed-metric MedQA panel) "
    "keeps every bar within a panel on a consistent metric and a consistent N. Motivation for "
    "preferring the LLM judge at all on MedQA: EM undercounts MedQA accuracy by 26-36pp vs. the LLM "
    "judge on `plain` (vs. FRAMES's 7-11pp), since MedQA's `correct_answer` is full multiple-choice "
    "option text that models often restate without the verbatim phrasing (Appendix \"Dual-Metric "
    "Robustness Check\": Pearson r=0.86 MedQA vs. 0.96 FRAMES level agreement, the same gap seen "
    "from the other side). The RERUN (noise-floor) row is regex-graded on ALL THREE panels "
    "regardless of the panel's own grading field -- `results/medqa_grid_rerun/` was never "
    "LLM-graded at all (`sampler_correct` is `None` on every row, all 11 models) -- so treat "
    "RERUN's accuracy number as an approximate reference, not a like-for-like comparison, on either "
    "MedQA panel."
)

def HOTPOTQA_GRADING_NOTE():
    return (
    "**Panel and grading:** HotpotQA accuracy is SQuAD-style EM (`regex_strict`) throughout -- "
    "not a choice between metrics but the only one available: every HotpotQA row was collected "
    "with `--no_grader`, so `sampler_correct` is `None` on all of them and there is no LLM judge "
    "to fall back on. Verdicts come from `scripts/grade_hotpotqa_regex.py`, which imports "
    "`regrade_regex.py`'s match functions, so this is the same grader as the FRAMES panel's EM. "
    "Two HotpotQA-specific exclusions, both inherited from that script so the numbers here match "
    "its own tables: the ~4.7% of examples whose gold is literally `yes`/`no` are dropped from "
    "accuracy (substring matching is meaningless on them -- `no` occurs constantly in prose), and "
    "rows with `stop_reason` set (1 row in 29,700, a `UsageLimitExceeded` salvage with 100 search "
    "calls) are dropped from every aggregate. The roster is the "
    f"{N_MODELS_BY_PANEL['HotpotQA']} open-weights models that were run on HotpotQA -- both Gemini "
    "models are absent, so this panel is not the same 11-model roster as FRAMES. The FRAMES "
    "cue-robustness SFT checkpoint present in the grid directory is also excluded: it is trained "
    "to be cue-invariant by construction, so averaging it in would dilute the effect being "
    "measured. Unlike FRAMES/MedQA, the RERUN (noise-floor) row here IS like-for-like with the "
    "panel's other bars -- same EM grader, since there is no LLM judge anywhere in this dataset. "
    "The TERSE bar is empty: the HotpotQA grid has a single phrasing and was never re-run under "
    "the terse rewrite."
)

GRADING_NOTES = []
if any(p['ds'] == 'MedQA' for p in PANELS):
    GRADING_NOTES.append(MEDQA_GRADING_NOTE())
if any(p['ds'] == 'HotpotQA' for p in PANELS):
    GRADING_NOTES.append(HOTPOTQA_GRADING_NOTE())
GRADING_NOTE = "\n\n".join(GRADING_NOTES)

excl_note = f" (excludes: {', '.join(EXCLUDE)})" if EXCLUDE else ""
md = [
    f"# Aggregate Cue Tables{excl_note}",
    "",
    "Per-cue aggregation across the model roster, paired vs. each model's own PLAIN baseline "
    "(same underlying per-model deltas as `brief_aggregate_search_acc_{mean,median}_*.png`). "
    "**Mean** columns use a one-sample t-test of the per-model point estimates against 0; "
    "**Median** columns use a one-sample Wilcoxon signed-rank test. Stars are Benjamini-Hochberg "
    f"FDR corrected within each table's own family of tests ({len(PANEL_KEYS)} "
    f"panel{'s' if len(PANEL_KEYS) != 1 else ''} x 10 rows = "
    f"{len(PANEL_KEYS)*10} tests): `*` q<.05, `**` q<.01, `***` q<.001. \"N models\" is how many "
    "models had enough paired examples to contribute a point estimate for that row, out of that "
    "panel's own model count (see Grading note below -- panels have different N).",
    "",
    GRADING_NOTE,
    "",
    "## Zero-Search Suppression",
    "",
    "Δ percentage points in the share of examples where the model made **zero** search calls, vs. "
    "that model's own PLAIN baseline. Positive = the cue makes the model more likely to skip search "
    "entirely on a given example (a complementary view to the Δ search-calls tables: a cue can lower "
    "the *average* call count either by trimming calls on examples that still get searched, or by "
    "pushing more examples to zero searches — this table isolates the latter). Condensed to the mean "
    "estimator only — median agrees in direction and magnitude throughout; see "
    f"`brief_zero_search_dotplot{SUFFIX}.png` for the same numbers as a single figure, and "
    "`brief_aggregate_search_acc_{mean,median}_*.png` if the mean/median split matters for your use.",
    "",
    make_condensed_dataset_table("zero_search", "{:+.1f}pp", "mean", np.mean, "ttest"),
    "## Search-Accuracy Example-Level Correlation",
    "",
    "Spearman correlation between each example's Δ search calls and Δ correctness "
    f"(graded per panel: {', '.join(f'{PANEL_LABEL[k]} = {GRADE_LABEL[k]}' for k in PANEL_KEYS)} -- "
    "see grading note above), "
    "computed within each (model, cue, dataset) at the example level (every paired example in "
    "that dataset), then aggregated across models. Values near 0 indicate no consistent example-level "
    "relationship between how much search shifted on that example and whether it got more or less "
    "correct — i.e. the search-volume drop is not concentrated on the examples driving accuracy "
    "changes (or vice versa).",
    "",
    make_md_table("corr", "{:+.2f}"),
]

md_path = os.path.join(OUT, f"brief_aggregate_tables{SUFFIX}.md")
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"Saved {md_path}")
