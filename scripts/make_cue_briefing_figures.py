#!/usr/bin/env python3
"""All-model cue-briefing figures (FRAMES + MedQA).

Same figure family as scripts/make_slide_figures.py (heatmap, search-shift bars,
suppression scatter, accuracy bars, combined search+accuracy, zero-search bars),
but rendered for **every** model in the grid rather than the 4 hand-picked slide
models. The condition axis is the style/instruction split used on the slides:
a "style" group (POLITE, TERSE-PLAIN) and an "instructions" group
(SHORT/natural, ELABORATE, QUERY, DIRECT), separated by a divider, all measured
against PLAIN of the base phrasing (verbose for FRAMES, orig for MedQA).

Inputs (regenerate these first when new models land):
  results/{frames,medqa}_token_analysis/joined_tokens.csv   (analyze_thinking_tokens.py)
  results/{frames_cues_full,medqa_grid}/<model>/*.json       (graded live via regrade_regex.py)

Outputs into results/cue_briefing/:
  brief_heatmap.png, brief_search_bars.png, brief_suppression.png,
  brief_accuracy.png, brief_combined_search_acc.png, brief_zero_search.png

--datasets picks which dataset rows the 2xN grids have (default "FRAMES,MedQA",
i.e. the original behaviour unchanged). "HotpotQA" renders the HotpotQA cue
grid; run it into its own --output-dir so it does not overwrite the paper's
FRAMES/MedQA figures, e.g.:
  uv run python scripts/make_cue_briefing_figures.py \
      --datasets HotpotQA --output-dir results/hotpotqa_cue_briefing
The suppression figure is skipped for any dataset without thinking-token data
(HotpotQA has no downloaded Logfire traces, so no thinking tokens exist for it).
"""
import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

_ap = argparse.ArgumentParser()
_ap.add_argument("--datasets", default="FRAMES,MedQA",
                 help="comma-separated dataset rows: FRAMES, MedQA, HotpotQA")
_ap.add_argument("--output-dir", default=None,
                 help="where the figures land (default results/cue_briefing)")
_args = _ap.parse_args()

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = _args.output_dir or os.path.join(ROOT, "results", "cue_briefing")
if not os.path.isabs(OUT):
    OUT = os.path.join(ROOT, OUT)
os.makedirs(OUT, exist_ok=True)

KNOWN_DATASETS = ["FRAMES", "MedQA", "HotpotQA"]
DATASETS = [d.strip() for d in _args.datasets.split(",") if d.strip()]
for _d in DATASETS:
    if _d not in KNOWN_DATASETS:
        raise SystemExit(f"--datasets: unknown dataset {_d!r}, expected one of {KNOWN_DATASETS}")
# Datasets whose rows carry thinking-token counts. That data comes from the
# joined_tokens.csv Logfire-trace join, which only exists for FRAMES and MedQA;
# the suppression (search-vs-thinking) figure is silently skipped for the rest.
THINKING_DATASETS = {"FRAMES", "MedQA"}

# ALL models (not just the 4 slide models). Order: gemini, qwen ladder,
# gemma ladder, nemotron, gpt-oss ladder.
MODEL_ORDER = [
    "gemini-3.1-pro-preview", "gemini-3.5-flash", "qwen3.5_122b", "qwen3.5_35b", "qwen3.5_4b",
    "gemma4_31b", "gemma4_e4b", "nemotron-3-nano_30b", "nemotron-cascade-2_30b",
    "gpt-oss_120b", "gpt-oss_20b",
]
MODEL_LABEL = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "qwen3.5_122b": "Qwen3.5 122B",
    "qwen3.5_35b": "Qwen3.5 35B", "qwen3.5_4b": "Qwen3.5 4B",
    "gemma4_31b": "Gemma4 31B", "gemma4_e4b": "Gemma4 E4B",
    "nemotron-3-nano_30b": "Nemotron3 30B", "nemotron-cascade-2_30b": "Nemotron-Cascade2 30B",
    "gpt-oss_120b": "GPT-OSS 120B", "gpt-oss_20b": "GPT-OSS 20B",
}
MODEL_COLOR = {
    "gemini-3.1-pro-preview": "#0571b0", "gemini-3.5-flash": "#dd3497",
    "qwen3.5_122b": "#ca0020",
    "qwen3.5_35b": "#f4a582", "qwen3.5_4b": "#92c5de",
    "gemma4_31b": "#5aae61", "gemma4_e4b": "#e66101",
    "nemotron-3-nano_30b": "#9970ab", "nemotron-cascade-2_30b": "#bf812d",
    "gpt-oss_120b": "#01665e", "gpt-oss_20b": "#80cdc1",
}

N = len(MODEL_ORDER)
GRID_W = 3.9 * N          # width of a 2xN per-model grid
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls", "thinking_words"]]


# ---------------------------------------------------------------------------
# HotpotQA. Loaded from scripts/grade_hotpotqa_regex.py's per_row.csv rather than
# the joined_tokens.csv + regrade_regex path the other two use, because (a) no
# Logfire traces were downloaded for the HotpotQA grid so there is no token join,
# and (b) every HotpotQA row was collected with --no_grader so `sampler_correct`
# is None everywhere and correctness is decided offline. That script reuses
# regrade_regex.py's own match functions, so HotpotQA EM == FRAMES EM as a metric.
# Its `search_calls` column already has the searchmulti mocked-history correction
# applied -- do not subtract MOCK_HISTORY_OFFSET again.
# Kept in step with the same loaders in scripts/make_aggregate_cue_tradeoff_figure.py.
HOTPOTQA_PER_ROW = os.path.join(ROOT, "results/hotpotqa_cue_grid_regex/per_row.csv")
HOTPOTQA_PHRASING = "orig"   # single phrasing; the grid was never re-run under "terse"
HOTPOTQA_EXCLUDE_SLUGS = {"gemma4-frames-robust-q4km_latest"}  # cue-invariant SFT transfer arm


def _load_hotpotqa_per_row():
    if not os.path.exists(HOTPOTQA_PER_ROW):
        raise SystemExit(
            f"missing {HOTPOTQA_PER_ROW}\n"
            "Build it first:  uv run python scripts/grade_hotpotqa_regex.py "
            "--results-root results/hotpotqa_cue_grid")
    d = pd.read_csv(HOTPOTQA_PER_ROW)
    d = d[~d.model.isin(HOTPOTQA_EXCLUDE_SLUGS)]
    # Same exclusion grade_hotpotqa_regex.py applies to its own aggregates: rows with
    # stop_reason set are salvaged answers from a run that hit the agent loop cap
    # (1 row in 29,700, with search_calls=100 -- enough on its own to move a cell).
    d = d[d.stop_reason.isna() | (d.stop_reason.astype(str).str.strip() == "")]
    d["cue"] = d.run_name.astype(str)
    d["phrasing"] = HOTPOTQA_PHRASING
    return d


HOTPOTQA_ROWS = _load_hotpotqa_per_row() if "HotpotQA" in DATASETS else None


def load_tokens_hotpotqa(rows):
    t = rows[rows.cue != "plain_rep2"].copy()
    # No thinking tokens exist for HotpotQA; the column is present so the shared
    # plumbing does not KeyError, and the suppression figure skips this dataset.
    t["thinking_words"] = np.nan
    return t[["model", "phrasing", "cue", "example_id", "search_calls", "thinking_words"]]


def load_graded_hotpotqa(rows):
    # EM over NON-boolean rows only -- on the ~4.7% of examples whose gold is
    # literally "yes"/"no", substring matching is meaningless ("no" occurs
    # constantly in prose), so grade_hotpotqa_regex.py scores those separately and
    # keeps them out of the headline. Dropped symmetrically from both sides here.
    g = rows[(~rows.answer_is_boolean.astype(bool)) & (rows.cue != "plain_rep2")].copy()
    g["regex"] = g["strict"].astype(int)
    g["llm"] = np.nan          # no LLM judge exists for HotpotQA
    return g[["model", "phrasing", "cue", "example_id", "regex", "llm"]]


def load_rerun_hotpotqa(rows):
    # HotpotQA's noise floor is the `plain_rep2` replicate inside the SAME grid dir
    # (FRAMES/MedQA keep theirs in a separate _rerun tree). Unlike those two, its
    # accuracy IS like-for-like with the panel's other bars: same EM grader.
    r = rows[(rows.cue == "plain_rep2") & (~rows.answer_is_boolean.astype(bool))].copy()
    r["regex"] = r["strict"].astype(int)
    return r[["model", "example_id", "regex", "search_calls"]]


_TOK_SOURCES = {
    "FRAMES": lambda: load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
    "MedQA": lambda: load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv")),
    "HotpotQA": lambda: load_tokens_hotpotqa(HOTPOTQA_ROWS),
}
TOK = {ds: _TOK_SOURCES[ds]() for ds in DATASETS}

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)


def load_graded(dataset, grid_dir):
    rows = []
    for path, slug, cond in _RG.find_grid_files(grid_dir, dataset):
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({"model": slug, "condition": cond, "example_id": g["example_id"],
                         "llm": int(bool(g["llm_correct"])), "regex": int(bool(g["regex_strict"]))})
    df = pd.DataFrame(rows)
    df["cue"] = df["condition"].map(base_cue)

    def get_ph(c):
        if c.startswith("epi_strong_"):
            return "epi_strong"
        return c.split("_", 1)[0]

    df["phrasing"] = df["condition"].map(get_ph)
    return df


_GRADED_SOURCES = {
    "FRAMES": lambda: load_graded("frames", os.path.join(ROOT, "results/frames_cues_full")),
    "MedQA": lambda: load_graded("medqa", os.path.join(ROOT, "results/medqa_grid")),
    "HotpotQA": lambda: load_graded_hotpotqa(HOTPOTQA_ROWS),
}
GRADED = {ds: _GRADED_SOURCES[ds]() for ds in DATASETS}


# ---------------------------------------------------------------------------
# Second plain run (test-retest): per (model, example_id) regex + search from the
# _rerun dirs, used for the "PLAIN<->PLAIN" reference bar (plain run2 vs plain run1).
# ---------------------------------------------------------------------------
def load_rerun(dataset, rerun_dir):
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
RERUN_MIN_N = 100  # need at least this many paired examples to show the reference bar
RERUN_LABEL = "PLAIN↔PLAIN"


def rerun_search_delta(ds, m, base_ph, base_cue="plain"):
    """% change in search calls: plain run2 vs plain run1 (paired by example_id)."""
    tok = TOK[ds]; rr = RERUN[ds]
    b = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    t = rr[rr.model == m][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    j = b.merge(t, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan, np.nan
    diff = (j["target"] - j["base"]).values
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        p = wilcoxon(j["target"].values, j["base"].values, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    mb = j["base"].mean()
    return (100.0 * (j["target"].mean() - mb) / mb if mb > 0 else np.nan), p


def rerun_acc_delta(ds, m, base_ph, base_cue="plain"):
    """pp change in regex accuracy: plain run2 vs plain run1 (McNemar, paired)."""
    gdf = GRADED[ds]; rr = RERUN[ds]
    b = gdf[(gdf.model == m) & (gdf.phrasing == base_ph) & (gdf.cue == base_cue)][["example_id", "regex"]].rename(columns={"regex": "p"})
    t = rr[rr.model == m][["example_id", "regex"]].rename(columns={"regex": "c"})
    if b.empty or t.empty:
        return np.nan, np.nan, 0
    j = t.merge(b, on="example_id")
    if len(j) < RERUN_MIN_N:
        return np.nan, np.nan, 0
    cc, pp = j["c"].values, j["p"].values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    Nn = len(j)
    disc = n10 + n01
    pv = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, pv, Nn


def get_conditions(ds):
    base_ph = "verbose" if ds == "FRAMES" else "orig"
    if ds == "HotpotQA":
        base_ph = HOTPOTQA_PHRASING
    # The ("terse", "plain") slot is kept for every dataset so the bar positions and
    # the group dividers line up across figures; on HotpotQA it simply has no data
    # and renders as an empty slot (that grid has one phrasing only).
    conds = [
        (base_ph, "polite"),
        ("terse", "plain"),
        (base_ph, "multiturn"),
        (base_ph, "searchmulti"),
        (base_ph, "natural"),
        (base_ph, "elaborate"),
        (base_ph, "query"),
        (base_ph, "direct"),
        (base_ph, "confident_parametric"),
    ]
    return base_ph, conds


def get_label(ph, cue):
    if cue == "plain":
        return f"{ph.upper()} (PLAIN)"
    if cue == "natural":
        return "SHORT"
    if cue == "confident_parametric":
        return "NO SEARCH NEEDED"
    return cue.upper()


def has_model(container, m):
    return m in set(container.model)


def mcnemar_delta(gdf, m, target_ph, target_cue, base_ph, base_cue="plain", metric="regex"):
    sub = gdf[gdf.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", metric]].rename(columns={metric: "c"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", metric]].rename(columns={metric: "p"})
    if t.empty or b.empty:
        return np.nan, np.nan, 0
    j = t.merge(b, on="example_id")
    cc, pp = j["c"].values, j["p"].values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    Nn = len(j)
    if Nn == 0:
        return np.nan, np.nan, 0
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, p_val, Nn


def abs_change_ci(tok, m, target_ph, target_cue, base_ph, base_cue="plain", nboot=2000, seed=0):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls", "thinking_words"]].rename(columns={"search_calls": "sc_t", "thinking_words": "tc_t"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls", "thinking_words"]].rename(columns={"search_calls": "sc_b", "thinking_words": "tc_b"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return None
    sc_t, sc_b = j["sc_t"].values, j["sc_b"].values
    tc_t, tc_b = j["tc_t"].values, j["tc_b"].values
    dS, dT = (sc_t - sc_b), (tc_t - tc_b)
    x, y = dS.mean(), dT.mean()
    rng = np.random.default_rng(seed)
    n = len(j)
    bx = np.array([dS[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    by = np.array([dT[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    xlo, xhi = np.percentile(bx, [2.5, 97.5])
    ylo, yhi = np.percentile(by, [2.5, 97.5])
    return dict(ph=target_ph, cue=target_cue, x=x, y=y, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                xsig=(xlo > 0 or xhi < 0), ysig=(ylo > 0 or yhi < 0))


def mean_search_ci(tok, m, ph, cue, nboot=2000, seed=0):
    """Bootstrap mean + 95% CI of RAW (not delta) search_calls for one (model, phrasing, cue)."""
    vals = tok[(tok.model == m) & (tok.phrasing == ph) & (tok.cue == cue)]["search_calls"].dropna().values
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=vals.mean(), lo=lo, hi=hi, n=n)


def wilcoxon_search(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return np.nan, np.nan
    diff = (j["target"] - j["base"]).values
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        p = wilcoxon(j["target"].values, j["base"].values, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    mean_base = j["base"].mean()
    mean_target = j["target"].mean()
    pct_change = 100.0 * (mean_target - mean_base) / mean_base if mean_base > 0 else np.nan
    return float(pct_change), p


def zero_search_delta(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "t_search"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "b_search"})
    if t.empty or b.empty:
        return np.nan, np.nan, 0
    j = t.merge(b, on="example_id").dropna()
    cc = (j["t_search"] == 0).astype(int).values
    pp = (j["b_search"] == 0).astype(int).values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    Nn = len(j)
    if Nn == 0:
        return np.nan, np.nan, 0
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, p_val, Nn


def rerun_zero_search_delta(ds, m, base_ph, base_cue="plain"):
    """pp change in zero-search frequency: plain run2 vs plain run1 (McNemar, paired)."""
    tok = TOK[ds]; rr = RERUN[ds]
    b = tok[(tok.model == m) & (tok.phrasing == base_ph) & (tok.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "b_search"})
    t = rr[rr.model == m][["example_id", "search_calls"]].rename(columns={"search_calls": "t_search"})
    if b.empty or t.empty:
        return np.nan, np.nan, 0
    j = t.merge(b, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan, np.nan, 0
    cc = (j["t_search"] == 0).astype(int).values
    pp = (j["b_search"] == 0).astype(int).values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    Nn = len(j)
    disc = n10 + n01
    pv = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, pv, Nn


def blank_panel(ax, m, ds, r):
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{MODEL_LABEL[m]}\n({ds})\nno data", ha="center", va="center",
            fontsize=9, color="#999")


# ===========================================================================
# Drop models that no loaded dataset has any rows for, so a single-dataset run
# does not spend half its panel grid on "no data" placeholders. With the default
# FRAMES+MedQA pair this is a no-op (all 11 models appear in both); on a
# HotpotQA-only run it drops the two Gemini models, which were never run there.
_PRESENT = set().union(*[set(TOK[ds].model) | set(GRADED[ds].model) for ds in DATASETS])
_MISSING = [m for m in MODEL_ORDER if m not in _PRESENT]
if _MISSING:
    print(f"  [skip] no rows in {DATASETS} for: {', '.join(_MISSING)}")
MODEL_ORDER = [m for m in MODEL_ORDER if m in _PRESENT]

MODEL_GROUP_1 = [m for m in ['gemini-3.1-pro-preview', 'qwen3.5_122b', 'gemma4_31b', 'gpt-oss_120b']
                 if m in MODEL_ORDER]
MODEL_GROUP_2 = [m for m in MODEL_ORDER if m not in MODEL_GROUP_1]
MODEL_GROUPS = [('primary', MODEL_GROUP_1), ('secondary', MODEL_GROUP_2)]

def _plot_suppression(group_name, MODEL_ORDER, supp_datasets):
    """FIG 3: suppression-vs-substitution scatter. Split out of plot_all() because
    it is the one figure that needs thinking tokens, so it renders only the subset
    of datasets that have them (see THINKING_DATASETS)."""
    N = len(MODEL_ORDER)
    fig, axes = plt.subplots(len(supp_datasets), N, figsize=(3.9 * N, 4 * len(supp_datasets)),
                             constrained_layout=True, squeeze=False)
    for r, ds in enumerate(supp_datasets):
        tok = TOK[ds]
        base_ph, conds = get_conditions(ds)
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            pts = []
            for (ph, cue) in conds:
                p = abs_change_ci(tok, m, ph, cue, base_ph)
                if p:
                    pts.append(p)
            if not pts:
                blank_panel(ax, m, ds, r)
                continue
            xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
            xmarg = (max(xs) - min(xs) or 1) * 0.25 + max(abs(p["xhi"] - p["x"]) for p in pts)
            ymarg = (max(ys) - min(ys) or 1) * 0.25 + max(abs(p["yhi"] - p["y"]) for p in pts)
            xlo, xhi = min(0, min(xs)) - xmarg, max(0, max(xs)) + xmarg
            ylo, yhi = min(0, min(ys)) - ymarg, max(0, max(ys)) + ymarg
            fx = (0 - xlo) / (xhi - xlo)
            ax.axhspan(0, yhi, xmin=0, xmax=fx, color="#0571b0", alpha=0.06)
            ax.axhspan(ylo, 0, xmin=0, xmax=fx, color="#ca0020", alpha=0.06)
            ax.axhline(0, color="#888", lw=0.8)
            ax.axvline(0, color="#888", lw=0.8)
            for p in pts:
                sig_both = p["xsig"] and p["ysig"]
                col = MODEL_COLOR[m]
                ax.errorbar(p["x"], p["y"], xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                            yerr=[[p["y"] - p["ylo"]], [p["yhi"] - p["y"]]], fmt="none",
                            ecolor=col, elinewidth=1.1, alpha=0.6)
                ax.scatter([p["x"]], [p["y"]], s=95, color=col if sig_both else "white",
                           edgecolors=col if not sig_both else "black", zorder=4)
                ax.annotate(get_label(p["ph"], p["cue"]), (p["x"], p["y"]), fontsize=8,
                            color="#333", xytext=(5, 3), textcoords="offset points")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            if r == 0 and cidx == 0:
                ax.text(0.03, 0.97, "SUBSTITUTION\nsearch↓ think↑", transform=ax.transAxes,
                        fontsize=8, color="#0571b0", fontweight="bold", va="top")
                ax.text(0.03, 0.03, "SUPPRESSION\nsearch↓ think↓", transform=ax.transAxes,
                        fontsize=8, color="#ca0020", fontweight="bold", va="bottom")
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ thinking tokens vs PLAIN {base_ph.upper()}")
            if r == len(supp_datasets) - 1:
                ax.set_xlabel(f"Δ search calls vs PLAIN {base_ph.upper()}")
    fig.savefig(os.path.join(OUT, f"brief_suppression_{group_name}.png"))
    plt.close(fig)


def plot_all(group_name, MODEL_ORDER):
    N = len(MODEL_ORDER)


    # ===========================================================================
    # FIG 2: Search-shift bars (2 x N)
    # ===========================================================================
    fig, axes = plt.subplots(len(DATASETS), N, figsize=(3.9 * N, 4 * len(DATASETS)),
                             constrained_layout=True, squeeze=False)
    for r, ds in enumerate(DATASETS):
        tok = TOK[ds]
        base_ph, conds = get_conditions(ds)
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            if not has_model(tok, m):
                blank_panel(ax, m, ds, r)
                continue
            rv, rp = rerun_search_delta(ds, m, base_ph)   # plain run2 vs plain run1 (noise floor)
            labels = [RERUN_LABEL]
            vals, pvals, is_rr = [rv], [rp], [True]
            for (ph, cue) in conds:
                labels.append(get_label(ph, cue))
                v, p = wilcoxon_search(tok, m, ph, cue, base_ph)
                vals.append(v); pvals.append(p); is_rr.append(False)
            
            xb = np.arange(len(labels))
            bars = ax.bar(xb, vals, 0.6, color=["#9e9e9e" if rr else MODEL_COLOR[m] for rr in is_rr])
            for bar, rr in zip(bars, is_rr):
                if rr:
                    bar.set_hatch("//"); bar.set_edgecolor("#444")
            vnn = [v for v in vals if not np.isnan(v)]
            span = (max(vnn) - min(vnn)) if vnn else 10
            for xi, val, p in zip(xb, vals, pvals):
                if np.isnan(val):
                    continue
                off = (span or 10) * 0.05
                off = off if val >= 0 else -off
                ax.text(xi, val + off, f"{val:+.0f}%\n{stars(p)}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(2.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(4.5, color="gray", linestyle="--", lw=1.2)
            ax.set_xticks(xb)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ search vs PLAIN {base_ph.upper()} (%)\n{ds}")
    fig.savefig(os.path.join(OUT, f"brief_search_bars_{group_name}.png"))
    plt.close(fig)

    # ===========================================================================
    # FIG 2b: Raw average search-call count per condition, with bootstrap 95% CI (2 x N)
    # ===========================================================================
    fig, axes = plt.subplots(len(DATASETS), N, figsize=(3.9 * N, 4 * len(DATASETS)),
                             constrained_layout=True, squeeze=False)
    for r, ds in enumerate(DATASETS):
        tok = TOK[ds]
        base_ph, conds = get_conditions(ds)
        all_conds = [(base_ph, "plain")] + conds
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            if not has_model(tok, m):
                blank_panel(ax, m, ds, r)
                continue
            labels, means, err_lo, err_hi = [], [], [], []
            for (ph, cue) in all_conds:
                labels.append(get_label(ph, cue))
                res = mean_search_ci(tok, m, ph, cue)
                if res is None:
                    means.append(np.nan)
                    err_lo.append(np.nan)
                    err_hi.append(np.nan)
                else:
                    means.append(res["mean"])
                    err_lo.append(max(0.0, res["mean"] - res["lo"]))
                    err_hi.append(max(0.0, res["hi"] - res["mean"]))
            if np.all(np.isnan(means)):
                blank_panel(ax, m, ds, r)
                continue
            xb = np.arange(len(labels))
            
            for xi, val, elo, ehi in zip(xb, means, err_lo, err_hi):
                if not np.isnan(val):
                    ax.bar(xi, val, 0.6, color=MODEL_COLOR[m],
                           yerr=[[elo], [ehi]], capsize=3, ecolor="#333", error_kw={"elinewidth": 1.1})
                           
            m_means = np.array(means, dtype=float)
            m_err_hi = np.array(err_hi, dtype=float)
            top = np.nanmax(m_means + m_err_hi)
            
            for xi, val in zip(xb, means):
                if not np.isnan(val):
                    ax.text(xi, val + top * 0.03, f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="#123")
            
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(2.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(4.5, color="gray", linestyle="--", lw=1.2)
            ax.set_xticks(xb)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Mean search calls (95% CI)\n{ds}")
    fig.suptitle("Average Search-Call Count by Cue (absolute, bootstrap 95% CI)", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT, f"brief_search_counts_{group_name}.png"))
    plt.close(fig)

    # ===========================================================================
    # FIG 3: Suppression vs substitution scatter (2 x N)
    # ===========================================================================
    supp_datasets = [ds for ds in DATASETS if ds in THINKING_DATASETS]
    if not supp_datasets:
        # Nothing to plot: the suppression figure is search-vs-THINKING, and no
        # loaded dataset has thinking tokens (they come from the Logfire trace join).
        print(f"  [skip] brief_suppression_{group_name}.png — no thinking-token data for {DATASETS}")
    else:
        _plot_suppression(group_name, MODEL_ORDER, supp_datasets)

    # ===========================================================================
    # FIG 4: Accuracy bars (regex, McNemar vs PLAIN) (2 x N)
    # ===========================================================================
    fig, axes = plt.subplots(len(DATASETS), N, figsize=(3.9 * N, 4 * len(DATASETS)),
                             constrained_layout=True, sharey="row", squeeze=False)
    for r, ds in enumerate(DATASETS):
        gdf = GRADED[ds]
        base_ph, conds = get_conditions(ds)
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            if not has_model(gdf, m):
                blank_panel(ax, m, ds, r)
                continue
            rv, rp, _ = rerun_acc_delta(ds, m, base_ph)   # plain run2 vs plain run1 (noise floor)
            labels = [RERUN_LABEL]
            vals, pvals, is_rr = [rv], [rp], [True]
            for (ph, cue) in conds:
                labels.append(get_label(ph, cue))
                v, p, _ = mcnemar_delta(gdf, m, ph, cue, base_ph)
                vals.append(v); pvals.append(p); is_rr.append(False)
            
            xb = np.arange(len(labels))
            bars = ax.bar(xb, vals, 0.6, color=["#9e9e9e" if rr else MODEL_COLOR[m] for rr in is_rr])
            for bar, rr in zip(bars, is_rr):
                if rr:
                    bar.set_hatch("//"); bar.set_edgecolor("#444")
            for xi, val, p in zip(xb, vals, pvals):
                if np.isnan(val):
                    continue
                off = 0.8 if val >= 0 else -0.8
                ax.text(xi, val + off, f"{val:+.0f}{stars(p)}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(2.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(4.5, color="gray", linestyle="--", lw=1.2)
            ax.set_xticks(xb)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ regex acc vs PLAIN {base_ph.upper()} (pp)\n{ds}")
    fig.savefig(os.path.join(OUT, f"brief_accuracy_{group_name}.png"))
    plt.close(fig)

    # ===========================================================================
    # FIG 5: Combined search + accuracy bars (2 x N)
    # ===========================================================================
    fig, axes = plt.subplots(len(DATASETS), N, figsize=(3.9 * N, 4 * len(DATASETS)),
                             constrained_layout=True, squeeze=False)
    for r, ds in enumerate(DATASETS):
        gdf = GRADED[ds]
        tok = TOK[ds]
        base_ph, conds = get_conditions(ds)
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            if not has_model(gdf, m) and not has_model(tok, m):
                blank_panel(ax, m, ds, r)
                continue
            w = 0.35
            rs, rsp = rerun_search_delta(ds, m, base_ph)   # plain run2 vs plain run1 (noise floor)
            ra, rap, _ = rerun_acc_delta(ds, m, base_ph)
            
            labels = [RERUN_LABEL]
            search_vals, search_ps, acc_vals, acc_ps = [rs], [rsp], [ra], [rap]
            for (ph, cue) in conds:
                labels.append(get_label(ph, cue))
                a_d, a_p, _ = mcnemar_delta(gdf, m, ph, cue, base_ph)
                s_d, s_p = wilcoxon_search(tok, m, ph, cue, base_ph)
                search_vals.append(s_d); search_ps.append(s_p); acc_vals.append(a_d); acc_ps.append(a_p)
            
            xb = np.arange(len(labels))
            ax.axvspan(-0.5, 0.5, color="#9e9e9e", alpha=0.15)
            ax.bar(xb - w / 2, search_vals, w, color="#4daf4a", label="Δ Search (%)")
            ax.bar(xb + w / 2, acc_vals, w, color="#377eb8", label="Δ Regex Acc (pp)")
            for xi, s_val, a_val, s_p, a_p in zip(xb, search_vals, acc_vals, search_ps, acc_ps):
                if not np.isnan(s_val):
                    ax.text(xi - w / 2, s_val + (2 if s_val >= 0 else -2), f"{s_val:+.0f}{stars(s_p)}",
                            ha="center", va="bottom" if s_val >= 0 else "top", fontsize=7.5, color="#123")
                if not np.isnan(a_val):
                    ax.text(xi + w / 2, a_val + (2 if a_val >= 0 else -2), f"{a_val:+.0f}{stars(a_p)}",
                            ha="center", va="bottom" if a_val >= 0 else "top", fontsize=7.5, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(2.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(4.5, color="gray", linestyle="--", lw=1.2)
            ax.set_xticks(xb)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ vs PLAIN {base_ph.upper()}\n{ds}")
    fig.savefig(os.path.join(OUT, f"brief_combined_search_acc_{group_name}.png"))
    plt.close(fig)

    # ===========================================================================
    # FIG 6: Zero-search change (2 x N)
    # ===========================================================================
    fig, axes = plt.subplots(len(DATASETS), N, figsize=(3.9 * N, 4 * len(DATASETS)),
                             constrained_layout=True, sharey="row", squeeze=False)
    for r, ds in enumerate(DATASETS):
        tok = TOK[ds]
        base_ph, conds = get_conditions(ds)
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            if not has_model(tok, m):
                blank_panel(ax, m, ds, r)
                continue
            rv, rp, _ = rerun_zero_search_delta(ds, m, base_ph)
            labels = [RERUN_LABEL]
            vals, pvals, is_rr = [rv], [rp], [True]
            for (ph, cue) in conds:
                labels.append(get_label(ph, cue))
                v, p, _ = zero_search_delta(tok, m, ph, cue, base_ph)
                vals.append(v); pvals.append(p); is_rr.append(False)
            
            xb = np.arange(len(labels))
            bars = ax.bar(xb, vals, 0.6, color=["#9e9e9e" if rr else MODEL_COLOR[m] for rr in is_rr])
            for bar, rr in zip(bars, is_rr):
                if rr:
                    bar.set_hatch("//"); bar.set_edgecolor("#444")
            for xi, val, p in zip(xb, vals, pvals):
                if np.isnan(val):
                    continue
                off = 1.0 if val >= 0 else -1.0
                ax.text(xi, val + off, f"{val:+.0f}{stars(p)}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(2.5, color="gray", linestyle="--", lw=1.2)
            ax.axvline(4.5, color="gray", linestyle="--", lw=1.2)
            ax.set_xticks(xb)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ zero-search vs PLAIN {base_ph.upper()} (pp)\n{ds}")
    fig.savefig(os.path.join(OUT, f"brief_zero_search_{group_name}.png"))
    plt.close(fig)


for name, grp in MODEL_GROUPS:
    plot_all(name, grp)
print(f"Saved all-model briefing figures to {OUT}")
