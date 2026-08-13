#!/usr/bin/env python3
"""Aggregate cue-tradeoff figures for the paper (replaces the dense
brief_combined_search_acc_{primary,secondary}.png model grids).

For each cue, computes the per-model (paired, vs that model's own PLAIN
baseline) % change in search calls and pp change in regex accuracy — 11
point estimates per cue, one per model. Two renderings of the same
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
args = ap.parse_args()
EXCLUDE = [m.strip() for m in args.exclude_models.split(",") if m.strip()]

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "cue_briefing")
os.makedirs(OUT, exist_ok=True)

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

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c


def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls"]]


TOK = {"FRAMES": load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
       "MedQA": load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))}

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)


def load_graded(dataset, grid_dir):
    rows = []
    for path, slug, cond in _RG.find_grid_files(grid_dir, dataset):
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({"model": slug, "condition": cond, "example_id": g["example_id"],
                         "regex": int(bool(g["regex_strict"]))})
    df = pd.DataFrame(rows)
    df["cue"] = df["condition"].map(base_cue)

    def get_ph(c):
        if c.startswith("epi_strong_"):
            return "epi_strong"
        return c.split("_", 1)[0]

    df["phrasing"] = df["condition"].map(get_ph)
    return df


GRADED = {"FRAMES": load_graded("frames", os.path.join(ROOT, "results/frames_cues_full")),
          "MedQA": load_graded("medqa", os.path.join(ROOT, "results/medqa_grid"))}

RERUN_MIN_N = 100


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


RERUN = {"FRAMES": load_rerun("frames", os.path.join(ROOT, "results/frames_cues_rerun")),
         "MedQA": load_rerun("medqa", os.path.join(ROOT, "results/medqa_grid_rerun"))}


def get_conditions(ds):
    base_ph = "verbose" if ds == "FRAMES" else "orig"
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


def rerun_acc_pp(ds, m, base_ph, base_cue="plain"):
    gdf = GRADED[ds]; rr = RERUN[ds]
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


def rerun_example_level_spearman(ds, m, base_ph, base_cue="plain"):
    tok, gdf, rr = TOK[ds], GRADED[ds], RERUN[ds]
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

DATASETS = ["FRAMES", "MedQA"]
DATA = {}  # ds -> {"labels":[...], "colors":[...], "search":[vals_array,...], "acc":[vals_array,...]}

for ds in DATASETS:
    tok, gdf = TOK[ds], GRADED[ds]
    base_ph, conds = get_conditions(ds)

    labels = ["RERUN\n(noise floor)"]
    colors = [CUE_COLOR["REFERENCE"]]
    groups = [CUE_GROUP["REFERENCE"]]
    search_arrs = [np.array([rerun_search_pct(ds, m, base_ph) for m in MODEL_ORDER])]
    acc_arrs = [np.array([rerun_acc_pp(ds, m, base_ph) for m in MODEL_ORDER])]
    zero_search_arrs = [np.array([rerun_zero_search_pp(ds, m, base_ph) for m in MODEL_ORDER])]
    corr_arrs = [np.array([rerun_example_level_spearman(ds, m, base_ph) for m in MODEL_ORDER])]

    for (ph, cue) in conds:
        labels.append(get_label(ph, cue))
        colors.append(CUE_COLOR.get(cue, "#666"))
        groups.append(CUE_GROUP.get(cue, "Directives"))
        search_arrs.append(np.array([search_pct_change(tok, m, ph, cue, base_ph) for m in MODEL_ORDER]))
        acc_arrs.append(np.array([acc_pp_change(gdf, m, ph, cue, base_ph) for m in MODEL_ORDER]))
        zero_search_arrs.append(np.array([zero_search_pp_change(tok, m, ph, cue, base_ph) for m in MODEL_ORDER]))
        corr_arrs.append(np.array([example_level_spearman(tok, gdf, m, ph, cue, base_ph) for m in MODEL_ORDER]))

    DATA[ds] = dict(labels=labels, colors=colors, groups=groups, base_ph=base_ph, search=search_arrs, acc=acc_arrs,
                     zero_search=zero_search_arrs, corr=corr_arrs)

N_BARS = len(DATA[DATASETS[0]]["labels"])  # 10 (noise floor + 9 cues)


def n_models_contributing(arr):
    return int(np.sum(~np.isnan(arr)))


def family_qmap(metric, test_method):
    """BH-FDR corrected q-values across the combined 2-dataset x N_BARS family for one metric."""
    pval_index, pvals = [], []
    for ds in DATASETS:
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
    Saves one figure per dataset: {name_stub}_{ds}{SUFFIX}.png"""
    # --- collect raw p-values across the full family: 2 metrics x 2 datasets x N_BARS bars ---
    # (kept as one combined family so the correction is defined over the full analysis,
    # even though each dataset is now rendered as its own image.)
    pval_index = []  # (ds, metric, bar_idx)
    pvals = []
    for ds in DATASETS:
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

    for ds in DATASETS:
        d = DATA[ds]
        labels, colors = d["labels"], d["colors"]
        xb = np.arange(len(labels))

        fig, axes = plt.subplots(2, 1, figsize=(8, 8.6), constrained_layout=True, sharex=True)

        for row, metric, ylabel, fmt in [
            (0, "search", f"Δ Search calls (%)\n{estimator_name}", "{:+.0f}%"),
            (1, "acc", f"Δ Regex accuracy (pp)\n{estimator_name}", "{:+.1f}"),
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
                ax.set_title(ds)
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


DATASET_COLOR = {"FRAMES": "#2c7fb8", "MedQA": "#d95f02"}


def render_zero_search_dotplot(metric, stat_fn, test_method, value_fmt, out_name, xlabel):
    """Condensed single-panel Cleveland dot plot: one row per cue, one dot per
    dataset, connected by a dumbbell line. Replaces the 2x4 zero-search bar
    grid for a compact, trend-at-a-glance paper figure."""
    qmap = family_qmap(metric, test_method)
    labels = DATA[DATASETS[0]]["labels"]
    groups = DATA[DATASETS[0]]["groups"]
    n = len(labels)
    y = np.arange(n)[::-1]  # RERUN at top, CONFIDENT at bottom, reading order

    fig, ax = plt.subplots(figsize=(8, 0.55 * n + 1.5), constrained_layout=True)

    for bi in range(n):
        pts = {}
        for ds in DATASETS:
            v = point_estimate(DATA[ds][metric][bi], stat_fn)
            pts[ds] = v
        finite = [v for v in pts.values() if not np.isnan(v)]
        if len(finite) == 2:
            ax.plot([pts[DATASETS[0]], pts[DATASETS[1]]], [y[bi], y[bi]],
                     color="#bbb", lw=1.5, zorder=1)
        for ds in DATASETS:
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
                           markeredgecolor=DATASET_COLOR[ds], markersize=9, label=ds) for ds in DATASETS]
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
    """One row per cue, one column per dataset (not per estimator) — the
    condensed alternative to make_md_table for metrics where mean/median
    agree in direction and a full 2x2 (dataset x estimator) breakdown is
    more detail than the headline trend needs."""
    qmap = family_qmap(metric, test_method)
    all_full = all(
        n_models_contributing(DATA[ds][metric][bi]) == N_MODELS
        for ds in DATASETS for bi in range(len(DATA[ds][metric]))
    )
    header_cols = ["Cue", "Group"] + DATASETS + ([] if all_full else [f"{ds} N" for ds in DATASETS])
    lines = [f"({estimator_name}, N={N_MODELS} models" + (")" if all_full else " unless noted)"), "",
             "| " + " | ".join(header_cols) + " |",
             "|" + "---|" * len(header_cols)]
    ref_labels = DATA[DATASETS[0]]["labels"]
    ref_groups = DATA[DATASETS[0]]["groups"]
    for bi, (label, group) in enumerate(zip(ref_labels, ref_groups)):
        label_clean = label.replace("\n", " ")
        cells = [label_clean, group]
        for ds in DATASETS:
            arr = DATA[ds][metric][bi]
            v = point_estimate(arr, stat_fn)
            cells.append((value_fmt.format(v) + stars(qmap.get((ds, bi)))) if not np.isnan(v) else "—")
        if not all_full:
            for ds in DATASETS:
                cells.append(f"{n_models_contributing(DATA[ds][metric][bi])}/{N_MODELS}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def make_md_table(metric, value_fmt):
    qmap_mean = family_qmap(metric, "ttest")
    qmap_median = family_qmap(metric, "wilcoxon")
    lines = []
    for ds in DATASETS:
        d = DATA[ds]
        lines.append(f"### {ds}\n")
        lines.append("| Cue | Group | Mean | Median | N models |")
        lines.append("|---|---|---|---|---|")
        for bi, (label, group) in enumerate(zip(d["labels"], d["groups"])):
            arr = d[metric][bi]
            mean_v = point_estimate(arr, np.mean)
            median_v = point_estimate(arr, np.median)
            mean_s = (value_fmt.format(mean_v) + stars(qmap_mean.get((ds, bi)))) if not np.isnan(mean_v) else "—"
            median_s = (value_fmt.format(median_v) + stars(qmap_median.get((ds, bi)))) if not np.isnan(median_v) else "—"
            label_clean = label.replace("\n", " ")
            lines.append(f"| {label_clean} | {group} | {mean_s} | {median_s} | {n_models_contributing(arr)}/{N_MODELS} |")
        lines.append("")
    return "\n".join(lines)


excl_note = f" (excludes: {', '.join(EXCLUDE)})" if EXCLUDE else ""
md = [
    f"# Aggregate Cue Tables{excl_note}",
    "",
    "Per-cue aggregation across the model roster, paired vs. each model's own PLAIN baseline "
    "(same underlying per-model deltas as `brief_aggregate_search_acc_{mean,median}_*.png`). "
    "**Mean** columns use a one-sample t-test of the per-model point estimates against 0; "
    "**Median** columns use a one-sample Wilcoxon signed-rank test. Stars are Benjamini-Hochberg "
    "FDR corrected within each table's own family of tests (2 datasets × 10 rows = 20 tests): "
    "`*` q<.05, `**` q<.01, `***` q<.001. \"N models\" is how many of the "
    f"{N_MODELS} models had enough paired examples to contribute a point estimate for that row.",
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
    "Spearman correlation between each example's Δ search calls and Δ correctness (regex-graded), "
    "computed within each (model, cue, dataset) at the example level (~500 paired examples per "
    "model), then aggregated across models. Values near 0 indicate no consistent example-level "
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
