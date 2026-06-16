"""
make_paper_figures.py — single entry point for every figure + number in the
"Benchmark-to-User Generalization Gap" paper (paper/draft_inside_out.tex).

Replaces the scattered analysis/plotting scripts. Reads only the precomputed
outputs of the two LLM data-producer stages
(analyze_parametric_search_interplay.py -> interplay_summary.csv / matched_examples_*.json,
 probe_commitment_locus.py -> commitment_locus_*.csv) and writes all artifacts,
flat, to a single output directory (default: results/paper_figures/).

All palette / stats / cell logic lives in src/viz.py.

Usage (no args reproduces the paper from the standard result layout):
    uv run python scripts/make_paper_figures.py
    uv run python scripts/make_paper_figures.py --output-dir results/paper_figures
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency

import src.viz as viz
import src.results_db as rdb

# ── Identity / layout used by the taxonomy figure ───────────────────────────────
_MODEL_RANK = {
    "gemini-3-pro-preview": 0,
    "nemotron-3-nano_30b": 1,
    "nemotron-3-nano": 1,
    "nemotron-3-nano-musique-v3-aug_latest": 1.5,
    "qwen3.5_122b": 2,
}
_MODEL_SHORT = {
    "gemini-3-pro-preview": "Gemini",
    "nemotron-3-nano_30b": "Nemotron",
    "nemotron-3-nano-musique-v3-aug_latest": "Nemotron (SFT)",
    "qwen3.5_122b": "Qwen",
}

# All model slugs we know how to plot, in display order. Includes the LoRA SFT model,
# so per-model figures pick it up automatically wherever its data is present (e.g. the
# ShareChat run with a 4th model), without forcing it into MuSiQue's 3-model figures.
_ALL_MODEL_SLUGS = [
    "gemini-3-pro-preview",
    "nemotron-3-nano_30b",
    "nemotron-3-nano-musique-v3-aug_latest",
    "qwen3.5_122b",
]


def _models_present(df: "pd.DataFrame") -> list[str]:
    """Known model slugs present in df, in display order."""
    present = set(df["model"].unique())
    return [s for s in _ALL_MODEL_SLUGS if s in present]


NEUTRAL_GREY = "#bdc3c7"


# A "pair" is the two conditions a dataset-agnostic figure compares: a benchmark-style
# condition (blue) and a user-like condition (red). The dataset-agnostic figures
# (taxonomy, cell-shift, calibration, redundancy, cascade) are driven by one of these.
class Pair:
    def __init__(self, bench_ds, comp_ds, bench_label, comp_label,
                 bench_short, comp_short, name, gap_desc, comp_phrasing):
        self.bench_ds = bench_ds          # dataset label of the benchmark-style condition
        self.comp_ds = comp_ds            # dataset label of the user-like condition
        self.bench_label = bench_label    # full legend label, e.g. "Benchmark (MuSiQue)"
        self.comp_label = comp_label
        self.bench_short = bench_short     # short x-tick / panel label
        self.comp_short = comp_short
        self.name = name                   # family name for titles, e.g. "MuSiQue"
        self.gap_desc = gap_desc           # phrase for titles
        self.comp_phrasing = comp_phrasing # adjective for the comparison condition, e.g. "User-like"
        self.ds_rank = {bench_ds: 0, comp_ds: 1}
        self.ds_short = {bench_ds: bench_short, comp_ds: comp_short}


MUSIQUE_PAIR = Pair(
    "musique", "musique-natural",
    "Benchmark (MuSiQue)", "User-like (MuSiQue-Natural)",
    "MuSiQue", "MuSiQue-Nat", "MuSiQue", "benchmark vs. user-like phrasing", "User-like")

NATURAL2_PAIR = Pair(
    "musique-natural", "musique-natural2",
    "MuSiQue-Natural (phrasing 1)", "MuSiQue-Natural2 (phrasing 2)",
    "Nat1", "Nat2", "MuSiQue-Natural", "phrasing 1 vs. phrasing 2", "Natural2")

MUSIQUE_NAT2_PAIR = Pair(
    "musique", "musique-natural2",
    "Benchmark (MuSiQue)", "User-like (MuSiQue-Natural2)",
    "MuSiQue", "Nat2", "MuSiQue vs Natural2", "benchmark vs. natural2 phrasing", "Natural2")

SHARECHAT_PAIR = Pair(
    "curated-sharechat-benchmark", "curated-sharechat",
    "Benchmark (paraphrase)", "Real user query",
    "SC-Bench", "SC-Real", "ShareChat", "formal paraphrase vs. real user query", "Real-user")


def _normalize_natural2_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the dataset prefix + run suffix that the interplay script bakes into
    natural2 model names (e.g. 'musique-natural2_nemotron-3-nano_30b_baseline_agent_run_1'
    → 'nemotron-3-nano_30b')."""
    df = df.copy()
    df["model"] = (df["model"]
                   .str.replace(r"^musique-natural2_", "", regex=True)
                   .str.replace(r"_baseline_agent_run_\d+$", "", regex=True))
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Shared: decompose the Missed cell into cross-hop / cascade / genuine
# ════════════════════════════════════════════════════════════════════════════════
def missed_subtype(df: pd.DataFrame) -> pd.Series:
    """For each uncertain & unsearched hop, classify the Missed cell into:
        'cross_hop' — answer surfaced via another hop's search (not truly missed),
        'cascade'   — a prior hop was answered wrong (miss may be downstream collateral),
        'genuine'   — all prior hops correct and not cross-hop-covered (the real error).
    Other hops get NaN. Uses parametric_accuracy of prior hops for the cascade test.
    """
    acc = (df.groupby(["dataset", "model", "example_id", "hop_index"])["parametric_accuracy"]
             .first().reset_index())
    have_xh = "missed_search_answer_found_cross_hop" in df.columns

    def classify(row):
        if have_xh and bool(row.get("missed_search_answer_found_cross_hop")):
            return "cross_hop"
        if row["hop_index"] == 0:
            return "genuine"
        prior = acc[(acc.dataset == row["dataset"]) & (acc.model == row["model"]) &
                    (acc.example_id == row["example_id"]) & (acc.hop_index < row["hop_index"])]
        if len(prior) == 0:
            return "genuine"
        return "cascade" if (prior["parametric_accuracy"] < 1.0).any() else "genuine"

    out = pd.Series(index=df.index, dtype=object)
    miss = (~df["certain"]) & (~df["search_attributed"])
    for idx, row in df[miss].iterrows():
        out.loc[idx] = classify(row)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Figure 1a — corrected quadrant taxonomy  (fig1_quadrant_taxonomy)
# ════════════════════════════════════════════════════════════════════════════════
# Six bands, ordered good -> bad, each with its own colour from the 6-class scale.
# The Missed cell is split per the missed-confounder analysis: cross-hop-covered
# (resolved via a sibling hop) and cascade-possible (downstream of an upstream wrong
# hop) are carved out from the genuine Missed error. (key, fill, text colour, label)
_TAX_BANDS = [
    ("E",  viz.EFFECTIVE, "white", "Effective (E)"),
    ("CP", viz.CP,        "#333",  "Correct Parametric (CP)"),
    ("XH", viz.CROSS_HOP, "#333",  "Cross-hop covered (resolved via sibling hop)"),
    ("PR", viz.PR,        "#333",  "Param. Redundant (PR)"),
    ("Mc", viz.CASCADE,   "#333",  "Missed — cascade-possible"),
    ("Mg", viz.MISSED,    "white", "Missed — genuine"),
]


def fig_quadrant_taxonomy(df: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    """Stacked six-band cell bar per (dataset, model), grouped model-first.

    Unlike the raw E/CP/PR/M split, the Missed cell here is corrected: cross-hop-
    covered and cascade-possible hops are separated from the genuine Missed error.
    """
    df = df.copy()
    df["_subtype"] = missed_subtype(df)

    def cell6(row):
        if row["certain"]:
            return "PR" if row["search_attributed"] else "CP"
        if row["search_attributed"]:
            return "E"
        return {"cross_hop": "XH", "cascade": "Mc", "genuine": "Mg"}.get(row["_subtype"], "Mg")

    df["cell6"] = df.apply(cell6, axis=1)
    fracs = {}
    for (dataset, model), sub in df.groupby(["dataset", "model"]):
        n = len(sub)
        fracs[(dataset, model)] = {b[0]: (sub["cell6"] == b[0]).sum() / n if n else 0.0 for b in _TAX_BANDS}

    combos = sorted(fracs.keys(), key=lambda c: (_MODEL_RANK.get(c[1], 99), pair.ds_rank.get(c[0], 99)))
    labels = [f"{pair.ds_short.get(d, d)}\n{_MODEL_SHORT.get(m, viz.display_name(m))}" for d, m in combos]

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.15), 5.2))
    x = np.arange(len(combos))
    bottoms = np.zeros(len(combos))
    for key, color, txtcol, _ in _TAX_BANDS:
        vals = np.array([fracs[c][key] for c in combos])
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.6,
               edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.035:
                ax.text(i, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8.5, color=txtcol, fontweight="bold")
        bottoms += vals

    for i, (_, m) in enumerate(combos):
        if i > 0 and _MODEL_RANK.get(m, 99) != _MODEL_RANK.get(combos[i - 1][1], 99):
            ax.axvline(i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=lbl)
               for _, c, _, lbl in _TAX_BANDS]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of per-hop decisions")
    ax.set_title(f"Per-hop cell decomposition — {pair.name} ({pair.gap_desc})\n"
                 "(Missed cell corrected: cross-hop-covered & cascade-possible separated from genuine)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig1_quadrant_taxonomy")


# ════════════════════════════════════════════════════════════════════════════════
# Figure — simple 4-cell taxonomy  (fig1_quadrant_taxonomy_simple)
# ════════════════════════════════════════════════════════════════════════════════
_SIMPLE_BANDS = [
    ("E",  viz.EFFECTIVE, "white", "Effective (E)"),
    ("CP", viz.CP,        "#333",  "Correct Parametric (CP)"),
    ("PR", viz.PR,        "#333",  "Param. Redundant (PR)"),
    ("M",  viz.MISSED,    "white", "Missed (M)"),
]


def fig_quadrant_taxonomy_simple(df: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    """Stacked 4-cell bar per (dataset, model) using the raw E/CP/PR/M quadrant column."""
    fracs = {}
    for (dataset, model), sub in df.groupby(["dataset", "model"]):
        n = len(sub)
        fracs[(dataset, model)] = {b[0]: (sub["quadrant"] == b[0]).sum() / n if n else 0.0
                                   for b in _SIMPLE_BANDS}

    combos = sorted(fracs.keys(), key=lambda c: (
        _MODEL_RANK.get(c[1], 99),
        pair.ds_rank.get(c[0], 99),
    ))
    labels = [f"{pair.ds_short.get(d, d)}\n{_MODEL_SHORT.get(m, viz.display_name(m))}"
              for d, m in combos]

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.15), 5.2))
    x = np.arange(len(combos))
    bottoms = np.zeros(len(combos))
    for key, color, txtcol, _ in _SIMPLE_BANDS:
        vals = np.array([fracs[c][key] for c in combos])
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.6,
               edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.035:
                ax.text(i, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8.5, color=txtcol, fontweight="bold")
        bottoms += vals

    for i, (_, m) in enumerate(combos):
        if i > 0 and _MODEL_RANK.get(m, 99) != _MODEL_RANK.get(combos[i - 1][1], 99):
            ax.axvline(i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    handles = [mpatches.Patch(facecolor=c, edgecolor="white", label=lbl)
               for _, c, _, lbl in _SIMPLE_BANDS]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Fraction of per-hop decisions", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig1_quadrant_taxonomy_simple")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 1c — cell-shift CI  (fig1c_cell_shift_ci)
# ════════════════════════════════════════════════════════════════════════════════
def fig_cell_shift_ci(df: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    """Effective + Missed cells, benchmark vs user-like, Wilson CI + chi-square shift."""
    models = [(s, viz.display_name(s)) for s in _models_present(df)]
    cells = [("E", "Effective (searched uncertain hop)"),
             ("M", "Missed (skipped uncertain hop)")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    x = np.arange(len(models))
    w = 0.34

    for ax, (cell, cell_title) in zip(axes, cells):
        for offset, ds, color, vlabel in [
            (-w / 2, pair.bench_ds, viz.BENCHMARK, pair.bench_label),
            (w / 2, pair.comp_ds, viz.NATURAL, pair.comp_label),
        ]:
            heights, los, his = [], [], []
            for slug, _ in models:
                sub = df[(df["dataset"] == ds) & (df["model"] == slug)]
                n = len(sub)
                k = int((sub["quadrant"] == cell).sum())
                p = k / n if n else 0.0
                lo, hi = viz.wilson_ci(k, n)
                heights.append(p); los.append(p - lo); his.append(hi - p)
            bars = ax.bar(x + offset, heights, w, color=color, label=vlabel, zorder=3)
            ax.errorbar(x + offset, heights, yerr=[los, his], fmt="none",
                        color="black", capsize=4, capthick=1.5, elinewidth=1.5, zorder=4)
            for bar, h, hi_err in zip(bars, heights, his):
                ax.text(bar.get_x() + bar.get_width() / 2, h + hi_err + 0.012,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        for i, (slug, _) in enumerate(models):
            b = df[(df["dataset"] == pair.bench_ds) & (df["model"] == slug)]
            nat = df[(df["dataset"] == pair.comp_ds) & (df["model"] == slug)]
            k_b, n_b = int((b["quadrant"] == cell).sum()), len(b)
            k_n, n_n = int((nat["quadrant"] == cell).sum()), len(nat)
            if min(n_b, n_n) == 0:
                continue
            _, p, _, _ = chi2_contingency([[k_b, n_b - k_b], [k_n, n_n - k_n]])
            delta = k_n / n_n - k_b / n_b
            top = max(k_b / n_b, k_n / n_n)
            ax.annotate(f"{'+' if delta >= 0 else ''}{delta:.0%} {viz.sig_stars(p)}",
                        xy=(x[i], top + 0.085), ha="center", fontsize=9,
                        color=viz.MISSED if cell == "M" else viz.BLUE, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in models], fontsize=9.5, fontweight="bold")
        ax.set_title(cell_title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.75)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of hop decisions", fontsize=10)
            ax.legend(fontsize=9, frameon=False, loc="upper left")

    fig.suptitle(f"{pair.comp_phrasing} phrasing contracts Effective search and expands Missed hops\n"
                 f"(Wilson 95% CI; chi-square on {pair.bench_short}->{pair.comp_short} shift)",
                 fontsize=13, fontweight="bold")
    fig.text(0.99, 0.01, "* p<0.05  ** p<0.01  *** p<0.001 (chi-square)",
             ha="right", va="bottom", fontsize=8, color=viz.ERR_MUTE)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    viz.savefig(fig, outdir, "fig1c_cell_shift_ci")


# ════════════════════════════════════════════════════════════════════════════════
# Commitment locus  (fig1_early_commit_rate, fig2_fate_breakdown)
# ════════════════════════════════════════════════════════════════════════════════
CAT_COLORS = {
    "commit_after_search": viz.EFFECTIVE,  # blue  — committed after searching
    "commit_no_search":    viz.MISSED,     # red   — committed without searching (the gap)
    "no_commit":           NEUTRAL_GREY,
}
CAT_LABELS = {
    "commit_after_search": "Committed after search",
    "commit_no_search":    "Committed without searching",
    "no_commit":           "Never committed",
}
CAT_ORDER = ["commit_after_search", "commit_no_search", "no_commit"]


def load_commitment(commitment_dir: str, override_entropy: bool = True) -> pd.DataFrame:
    canon = viz.load_canonical_entropy() if override_entropy else {}
    frames = []
    present_slugs = []
    for slug in _ALL_MODEL_SLUGS:
        files = glob.glob(os.path.join(commitment_dir, f"commitment_locus_{slug}_shard_*.csv"))
        if not files:
            continue
        present_slugs.append(slug)
        for f in files:
            sub = pd.read_csv(f)
            sub["model"] = viz.display_name(slug)
            sub["_slug"] = slug
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if canon:
        df["semantic_entropy"] = df.apply(
            lambda r: canon.get((r["_slug"], r["example_id"], int(r["hop_index"])),
                                r["semantic_entropy"]), axis=1)
    df.drop(columns=["_slug"], inplace=True)

    df["committed"] = df["first_committed_turn_index"] >= 0
    df["searched_for_hop"] = df["first_search_turn_for_hop"] >= 0

    def categorize(row):
        if not row["committed"]:
            return "no_commit"
        if not row["searched_for_hop"] or row["committed_before_search"] is True:
            return "commit_no_search"
        return "commit_after_search"

    df["category"] = df.apply(categorize, axis=1)
    df["early_commit"] = df["category"] == "commit_no_search"
    order = [viz.display_name(s) for s in present_slugs]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    return df[df["semantic_entropy"] > 0].copy()


def fig_early_commit_rate(unc: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    MODEL_ORDER = list(unc["model"].cat.categories)
    grp = unc.groupby(["model", "variant"], observed=True)["early_commit"]
    rates = grp.mean().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)
    counts = grp.sum().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)
    ns = unc.groupby(["model", "variant"], observed=True).size().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)

    def ci_err(model, variant):
        k, n = int(counts.loc[model, variant]), int(ns.loc[model, variant])
        lo, hi = viz.wilson_ci(k, n)
        p = rates.loc[model, variant]
        return p - lo, hi - p

    x = np.arange(len(MODEL_ORDER))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, variant, color, label in [
        (-w / 2, "benchmark", viz.BENCHMARK, pair.bench_label),
        (w / 2, "natural", viz.NATURAL, pair.comp_label),
    ]:
        heights = rates[variant].values
        errs_lo = np.array([ci_err(m, variant)[0] for m in MODEL_ORDER])
        errs_hi = np.array([ci_err(m, variant)[1] for m in MODEL_ORDER])
        bars = ax.bar(x + offset, heights, w, color=color, label=label, zorder=3)
        ax.errorbar(x + offset, heights, yerr=[errs_lo, errs_hi], fmt="none",
                    color="black", capsize=4, capthick=1.5, elinewidth=1.5, zorder=4)
        for bar, h, model in zip(bars, heights, MODEL_ORDER):
            bx = bar.get_x() + bar.get_width() / 2
            ax.text(bx, h + 0.022, f"{h:.0%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    for i, model in enumerate(MODEL_ORDER):
        k_b, n_b = int(counts.loc[model, "benchmark"]), int(ns.loc[model, "benchmark"])
        k_n, n_n = int(counts.loc[model, "natural"]), int(ns.loc[model, "natural"])
        _, p, _, _ = chi2_contingency([[k_b, n_b - k_b], [k_n, n_n - k_n]])
        b_rate, n_rate = rates.loc[model, "benchmark"], rates.loc[model, "natural"]
        ax.annotate(f"+{n_rate - b_rate:.0%} {viz.sig_stars(p)}",
                    xy=(x[i] + w / 2, n_rate + 0.105), ha="center", fontsize=10,
                    color=viz.NATURAL, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=12, fontweight="bold")
    ax.set_ylabel("Uncertain hops committed\nwithout searching", fontsize=11)
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(f"{pair.comp_phrasing} phrasing triggers more unsearched commits",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig1_early_commit_rate")


def fig_fate_breakdown(unc: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    MODEL_ORDER = list(unc["model"].cat.categories)
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(max(13, 4 * len(MODEL_ORDER)), 5.5), sharey=True)
    fig.suptitle("Commitment fate on uncertain hops — benchmark vs natural",
                 fontsize=13, fontweight="bold")
    variants = ["benchmark", "natural"]
    x = np.arange(len(variants))

    for ax, model in zip(axes, MODEL_ORDER):
        sub = unc[unc["model"] == model]
        n_total = sub.groupby("variant", observed=True).size().reindex(variants).values
        counts = (sub.groupby(["variant", "category"], observed=True).size()
                  .unstack(fill_value=0).reindex(index=variants, columns=CAT_ORDER, fill_value=0))
        fracs = counts.div(n_total, axis=0)
        bottoms = np.zeros(len(variants))
        orange_tops = np.zeros(len(variants))
        for cat in CAT_ORDER:
            vals = fracs[cat].values; raw = counts[cat].values
            ax.bar(x, vals, bottom=bottoms, color=CAT_COLORS[cat],
                   edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
            for xi, (v, b, k) in enumerate(zip(vals, bottoms, raw)):
                if v > 0.05:
                    ax.text(xi, b + v / 2, f"{v:.0%}\n({k})", ha="center", va="center",
                            fontsize=9, color="white" if cat != "no_commit" else viz.ERR_DARK,
                            fontweight="bold", linespacing=1.3)
            if cat == "commit_no_search":
                orange_tops = bottoms + vals
            bottoms += vals

        errs_lo, errs_hi = [], []
        for vi, variant in enumerate(variants):
            k = int(counts.loc[variant, "commit_no_search"]); n = int(n_total[vi])
            lo, hi = viz.wilson_ci(k, n); p = fracs.loc[variant, "commit_no_search"]
            errs_lo.append(p - lo); errs_hi.append(hi - p)
        ax.errorbar(x, orange_tops, yerr=[errs_lo, errs_hi], fmt="none",
                    color=viz.MISSED, capsize=5, capthick=1.8, elinewidth=1.8, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Benchmark\n(N={n_total[0]})", f"{pair.comp_phrasing}\n(N={n_total[1]})"], fontsize=9.5)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of uncertain hops", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.2, zorder=0)

    handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c]) for c in CAT_ORDER]
    handles.append(plt.Line2D([0], [0], color=viz.MISSED, linewidth=2, label="95% CI (commit w/o search)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig2_fate_breakdown", exts=("png",))


# ════════════════════════════════════════════════════════════════════════════════
# Missed-cell cascade decomposition  (fig_missed_cascade_bar)
# ════════════════════════════════════════════════════════════════════════════════
def fig_missed_cascade(df: pd.DataFrame, outdir: Path, pair: Pair = MUSIQUE_PAIR) -> None:
    """Decompose M-class hops into genuine-missed vs cascade-possible."""
    acc_lookup = (df.groupby(["dataset", "model", "example_id", "hop_index"])["parametric_accuracy"]
                  .first().reset_index())
    m_rows = df[df["quadrant"] == "M"].copy()

    def _label(row):
        if row["hop_index"] == 0:
            return "genuine_missed"
        prior = acc_lookup[(acc_lookup["dataset"] == row["dataset"]) &
                           (acc_lookup["model"] == row["model"]) &
                           (acc_lookup["example_id"] == row["example_id"]) &
                           (acc_lookup["hop_index"] < row["hop_index"])]
        if len(prior) == 0:
            return "genuine_missed"
        return "cascade_possible" if (prior["parametric_accuracy"] < 1.0).any() else "genuine_missed"

    m_rows["cascade_label"] = m_rows.apply(_label, axis=1)
    pivot = (m_rows.groupby(["dataset", "model", "cascade_label"]).size()
             .reset_index(name="count")
             .pivot_table(index=["dataset", "model"], columns="cascade_label",
                          values="count", fill_value=0).reset_index())
    for col in ["genuine_missed", "cascade_possible"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["genuine_missed"] + pivot["cascade_possible"]
    pivot["frac_genuine"] = pivot["genuine_missed"] / pivot["total"].replace(0, np.nan)
    pivot["frac_cascade"] = pivot["cascade_possible"] / pivot["total"].replace(0, np.nan)

    datasets = [d for d in [pair.bench_ds, pair.comp_ds] if d in pivot["dataset"].unique()]
    title_map = {pair.bench_ds: f"{pair.name} ({pair.bench_short})",
                 pair.comp_ds: f"{pair.name} ({pair.comp_short})"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets) + 1, 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = pivot[pivot["dataset"] == dataset].copy()
        sub["rank"] = sub["model"].map(_MODEL_RANK).fillna(99)
        sub = sub.sort_values("rank")
        x = np.arange(len(sub))
        ax.bar(x, sub["frac_genuine"], color=viz.GENUINE_MISSED, label="Genuine missed")
        ax.bar(x, sub["frac_cascade"], bottom=sub["frac_genuine"], color=viz.CASCADE, label="Cascade possible")
        for i, (_, row) in enumerate(sub.iterrows()):
            if (row["frac_cascade"] or 0) >= 0.05:
                ax.text(x[i], row["frac_genuine"] + row["frac_cascade"] / 2,
                        f"{row['frac_cascade'] * 100:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            if (row["frac_genuine"] or 0) >= 0.05:
                ax.text(x[i], row["frac_genuine"] / 2, f"{row['frac_genuine'] * 100:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            ax.text(x[i], 1.01, f"n={int(row['total'])}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([viz.display_name(m) for m in sub["model"]], rotation=15, ha="right")
        ax.set_title(title_map.get(dataset, dataset), fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Fraction of M-class hops")

    handles = [
        mpatches.Patch(color=viz.GENUINE_MISSED, label="Genuine missed\n(all prior hops correct)"),
        mpatches.Patch(color=viz.CASCADE, label="Cascade possible\n(>=1 prior hop incorrect)"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("M-class decomposition: genuine vs. cascade-confounded", fontsize=13, fontweight="bold")
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig_missed_cascade_bar", exts=("png",))

    pivot.to_csv(outdir / "missed_confounder_summary.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# Cost of a Missed hop  (missed_hop_cost_intrace + missed_hop_cost_outcome)
# ════════════════════════════════════════════════════════════════════════════════
def _join_cost_frame(df_by_variant: dict[str, pd.DataFrame], commitment_dir: str) -> pd.DataFrame:
    """Join interplay cost-cells with the commitment-locus in-trace correctness."""
    ip_frames = []
    for variant, d in df_by_variant.items():
        d = viz.add_cost_cells(d)
        d = d[d["model"].isin(_ALL_MODEL_SLUGS)].copy()
        d["variant"] = variant
        ip_frames.append(d)
    ip = pd.concat(ip_frames, ignore_index=True)

    cl_frames = []
    for slug in _ALL_MODEL_SLUGS:
        for f in glob.glob(os.path.join(commitment_dir, f"commitment_locus_{slug}_shard_*.csv")):
            c = pd.read_csv(f)
            c["model"] = slug
            cl_frames.append(c[["model", "variant", "example_id", "hop_index",
                                "matches_gold", "first_search_turn_for_hop"]])
    cl = pd.concat(cl_frames, ignore_index=True).rename(columns={"matches_gold": "intrace_correct"})
    cl["cl_searched"] = cl["first_search_turn_for_hop"] >= 0

    keys = ["model", "variant", "example_id", "hop_index"]
    return ip.merge(cl[keys + ["intrace_correct", "cl_searched"]], on=keys, how="inner")


def fig_missed_hop_cost(df_by_variant: dict[str, pd.DataFrame], commitment_dir: str, outdir: Path,
                        pair: Pair = MUSIQUE_PAIR) -> None:
    df = _join_cost_frame(df_by_variant, commitment_dir)
    models = [(s, viz.display_name(s)) for s in _models_present(df)]
    VARIANTS = [("benchmark", pair.bench_label), ("natural", pair.comp_label)]

    # ── Rung 1: in-trace per-hop correctness ────────────────────────────────────
    rows = []
    for slug, name in models:
        for variant, _ in VARIANTS:
            sub = df[(df["model"] == slug) & (df["variant"] == variant)]
            for cell, mask in [("Truly missed", sub["truly_missed"]),
                               ("Effective (searched)", sub["effective"])]:
                s = sub[mask]
                k, n = int(s["intrace_correct"].sum()), len(s)
                lo, hi = viz.wilson_ci(k, n)
                rows.append({"model": name, "variant": variant, "cell": cell,
                             "intrace_correct": k / n if n else np.nan,
                             "ci_lo": lo, "ci_hi": hi, "n": n, "k": k})
    r1 = pd.DataFrame(rows)
    r1.to_csv(outdir / "missed_hop_cost_intrace.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (variant, vlabel) in zip(axes, VARIANTS):
        x = np.arange(len(models)); w = 0.36
        for off, cell, color in [(-w / 2, "Truly missed", viz.MISSED),
                                 (w / 2, "Effective (searched)", viz.EFFECTIVE)]:
            hh, los, his, ns = [], [], [], []
            for _, name in models:
                r = r1[(r1.model == name) & (r1.variant == variant) & (r1.cell == cell)].iloc[0]
                hh.append(r.intrace_correct); los.append(r.intrace_correct - r.ci_lo)
                his.append(r.ci_hi - r.intrace_correct); ns.append(int(r.n))
            bars = ax.bar(x + off, hh, w, color=color, label=cell, zorder=3)
            ax.errorbar(x + off, hh, yerr=[los, his], fmt="none", color="black",
                        capsize=4, capthick=1.3, elinewidth=1.3, zorder=4)
            for b, h, hi_e, n in zip(bars, hh, his, ns):
                ax.text(b.get_x() + b.get_width() / 2, h + hi_e + 0.015, f"{h:.0%}\nn={n}",
                        ha="center", va="bottom", fontsize=8)
        for i, (_, name) in enumerate(models):
            tm = r1[(r1.model == name) & (r1.variant == variant) & (r1.cell == "Truly missed")].iloc[0]
            ef = r1[(r1.model == name) & (r1.variant == variant) & (r1.cell == "Effective (searched)")].iloc[0]
            p = viz.two_prop_p(int(tm.k), int(tm.n), int(ef.k), int(ef.n))
            ax.annotate(viz.sig_stars(p), xy=(x[i], max(tm.intrace_correct, ef.intrace_correct) + 0.10),
                        ha="center", fontsize=11, color=viz.MISSED, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in models], fontsize=9.5, fontweight="bold")
        ax.set_title(vlabel, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("In-trace per-hop correctness", fontsize=10)
            ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("Cost of a Missed hop (Rung 1): uncertain hops the model skips are answered\n"
                 "less correctly in-trace than the ones it searches (Wilson 95% CI; two-proportion test)",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.99, 0.01, "* p<0.05  ** p<0.01  *** p<0.001", ha="right", va="bottom",
             fontsize=8, color=viz.ERR_MUTE)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    viz.savefig(fig, outdir, "missed_hop_cost_intrace")

    # ── Rung 2: final-answer outcome, difficulty-stratified ─────────────────────
    q = (df.groupby(["model", "variant", "example_id"])
           .agg(n_truly_missed=("truly_missed", "sum"),
                num_hops=("hop_index", "nunique"),
                correct=("aggregate_correct", "max"))
           .reset_index().dropna(subset=["correct"]))
    q["exposed"] = q["n_truly_missed"] >= 1

    # Rung 2 needs final-answer correctness (aggregate_correct). ShareChat is
    # open-ended and has none, so skip the outcome figure when it is unavailable.
    has_outcome = not q.empty
    r2 = mh = None
    if has_outcome:
        out_rows, mh_rows = [], []
        for variant, _ in VARIANTS:
            sub = q[q["variant"] == variant]
            groups = [("all", sub)] + [(f"{h}hop", sub[sub.num_hops == h]) for h in sorted(sub.num_hops.unique())]
            for grp_label, gsub in groups:
                for exp, mask in [("0 missed", ~gsub.exposed), (">=1 missed", gsub.exposed)]:
                    s = gsub[mask]
                    k, n = int(s["correct"].sum()), len(s)
                    lo, hi = viz.wilson_ci(k, n)
                    out_rows.append({"variant": variant, "stratum": grp_label, "group": exp,
                                     "p_correct": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi, "n": n})
            strata = []
            for h in sorted(sub.num_hops.unique()):
                g = sub[sub.num_hops == h]
                a = int(g[g.exposed].correct.sum()); b = int(g.exposed.sum()) - a
                c = int(g[~g.exposed].correct.sum()); d = int((~g.exposed).sum()) - c
                strata.append((a, b, c, d))
            or_mh, p_mh = viz.mantel_haenszel(strata)
            mh_rows.append({"variant": variant, "MH_odds_ratio": or_mh, "p": p_mh})

        r2 = pd.DataFrame(out_rows); mh = pd.DataFrame(mh_rows)
        r2.to_csv(outdir / "missed_hop_cost_outcome.csv", index=False)
        mh.to_csv(outdir / "missed_hop_cost_mh.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, (variant, vlabel) in zip(axes, VARIANTS):
            strata_labels = [s for s in r2[r2.variant == variant].stratum.unique() if s != "all"]
            x = np.arange(len(strata_labels)); w = 0.36
            for off, grp, color in [(-w / 2, "0 missed", viz.EFFECTIVE), (w / 2, ">=1 missed", viz.MISSED)]:
                hh, los, his, ns = [], [], [], []
                for st in strata_labels:
                    r = r2[(r2.variant == variant) & (r2.stratum == st) & (r2.group == grp)].iloc[0]
                    hh.append(r.p_correct); los.append(r.p_correct - r.ci_lo)
                    his.append(r.ci_hi - r.p_correct); ns.append(int(r.n))
                bars = ax.bar(x + off, hh, w, color=color, label=grp, zorder=3)
                ax.errorbar(x + off, hh, yerr=[los, his], fmt="none", color="black",
                            capsize=4, capthick=1.3, elinewidth=1.3, zorder=4)
                for b, h, hi_e, n in zip(bars, hh, his, ns):
                    ax.text(b.get_x() + b.get_width() / 2, h + hi_e + 0.015, f"{h:.0%}\nn={n}",
                            ha="center", va="bottom", fontsize=8)
            mh_r = mh[mh.variant == variant].iloc[0]
            ax.set_xticks(x); ax.set_xticklabels(strata_labels, fontsize=10, fontweight="bold")
            ax.set_title(f"{vlabel}\nMantel-Haenszel OR={mh_r.MH_odds_ratio:.2f} "
                         f"({viz.sig_stars(mh_r.p)}, controls for hop count)", fontsize=10.5, fontweight="bold")
            ax.set_ylim(0, 0.9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if ax is axes[0]:
                ax.set_ylabel("P(final answer correct)", fontsize=10)
                ax.legend(fontsize=9, frameon=False, loc="upper right")
        fig.suptitle("Cost of a Missed hop (Rung 2): questions with >=1 Truly-missed hop are answered\n"
                     "correctly less often, within every difficulty stratum (Wilson 95% CI)",
                     fontsize=12.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        viz.savefig(fig, outdir, "missed_hop_cost_outcome")
    else:
        print("  (missed-hop-cost Rung 2 skipped: no final-answer correctness in this dataset)")

    agree = (df["search_attributed"] == df["cl_searched"]).mean()
    with open(outdir / "missed_hop_cost_report.md", "w") as f:
        f.write("# Cost of a Missed Hop — Report\n\n")
        f.write(f"Joined per-hop rows (base models x 2 variants): **{len(df)}**. "
                f"Interplay `searched` and commitment-locus `first_search_turn_for_hop>=0` "
                f"agree on **{agree:.1%}** of joined rows.\n\n")
        f.write("## Rung 1 — in-trace per-hop correctness (uncertain hops)\n\n")
        f.write(r1.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        if has_outcome:
            f.write("## Rung 2 — final-answer outcome by #Truly-missed hops\n\n")
            f.write(r2.to_markdown(index=False, floatfmt=".3f") + "\n\n")
            f.write("### Mantel-Haenszel (controls for hop count); OR<1 => missing lowers P(correct)\n\n")
            f.write(mh.to_markdown(index=False, floatfmt=".4f") + "\n")
        else:
            f.write("## Rung 2 — not available\n\n"
                    "This dataset has no final-answer correctness (`aggregate_correct`), so the "
                    "difficulty-stratified outcome analysis is omitted.\n")


# ════════════════════════════════════════════════════════════════════════════════
# SFT conversion targeting  (sft_conversion_targeting)
# ════════════════════════════════════════════════════════════════════════════════
def fig_sft_conversion(base_nat: pd.DataFrame, sft_nat: pd.DataFrame,
                       commitment_dir: str, outdir: Path) -> None:
    base_slug = "nemotron-3-nano_30b"
    base = viz.add_cost_cells(base_nat)
    base = base[base.model == base_slug].copy()
    sft = viz.add_cost_cells(sft_nat).copy()

    cl_files = glob.glob(os.path.join(commitment_dir, f"commitment_locus_{base_slug}_shard_*.csv"))
    cl = pd.concat([pd.read_csv(f) for f in cl_files], ignore_index=True)
    cl = cl[cl.variant == "natural"][["example_id", "hop_index", "matches_gold"]]
    cl = cl.rename(columns={"matches_gold": "base_intrace_correct"})

    keys = ["example_id", "hop_index"]
    m = (base[keys + ["truly_missed", "aggregate_correct"]]
         .merge(sft[keys + ["search_attributed", "uncertain"]].rename(
             columns={"search_attributed": "sft_searched", "uncertain": "sft_uncertain"}), on=keys)
         .merge(cl, on=keys, how="left"))

    miss = m[m.truly_missed & m.base_intrace_correct.notna()].copy()
    miss["cost_class"] = np.where(miss.base_intrace_correct.astype(bool),
                                  "low-cost\n(was right)", "high-cost\n(was wrong)")
    miss["converted"] = miss.sft_searched.astype(bool)

    rows = []
    for cls in ["high-cost\n(was wrong)", "low-cost\n(was right)"]:
        s = miss[miss.cost_class == cls]
        k_c, n_c = int(s.converted.sum()), len(s)
        lo, hi = viz.wilson_ci(k_c, n_c)
        rows.append({"cost_class": cls.replace("\n", " "), "n_base_missed": n_c,
                     "n_converted": k_c, "conversion_rate": k_c / n_c if n_c else np.nan,
                     "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(rows)
    summ.to_csv(outdir / "sft_conversion_targeting.csv", index=False)

    hi = miss[miss.cost_class.str.startswith("high")]
    lo = miss[miss.cost_class.str.startswith("low")]
    p_diff = viz.two_prop_p(int(hi.converted.sum()), len(hi), int(lo.converted.sum()), len(lo))
    conv = miss[miss.converted]
    comp_high = (conv.cost_class.str.startswith("high")).mean() if len(conv) else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    classes = ["high-cost\n(was wrong)", "low-cost\n(was right)"]
    colors = [viz.HIGH_COST, viz.LOW_COST]
    for i, (cls, col) in enumerate(zip(classes, colors)):
        r = summ[summ.cost_class == cls.replace("\n", " ")].iloc[0]
        ax.bar(i, r.conversion_rate, 0.6, color=col, zorder=3)
        ax.errorbar(i, r.conversion_rate,
                    yerr=[[r.conversion_rate - r.ci_lo], [r.ci_hi - r.conversion_rate]],
                    fmt="none", color="black", capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        ax.text(i, r.ci_hi + 0.02, f"{r.conversion_rate:.0%}\n({r.n_converted}/{r.n_base_missed})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.plot([0, 1], [summ.iloc[0].ci_hi + 0.13] * 2, color="black", lw=1.2)
    ax.text(0.5, summ.iloc[0].ci_hi + 0.135, viz.sig_stars(p_diff), ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, max(summ.ci_hi) + 0.25)
    ax.set_ylabel("SFT conversion rate\n(fraction of base-Missed hops the SFT now searches)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Does the SFT fix the Missed hops that matter?\n"
                 "Conversion of base-model Truly-missed hops, by in-trace cost",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    viz.savefig(fig, outdir, "sft_conversion_targeting")

    high_frac = (miss.cost_class.str.startswith("high")).mean()
    with open(outdir / "sft_conversion_targeting_report.md", "w") as f:
        f.write("# Does the SFT fix the Missed hops that matter?\n\n")
        f.write(f"Base `{base_slug}`, MuSiQue-Natural. Base Truly-missed hops with an in-trace label: "
                f"**{len(miss)}** ({high_frac:.0%} high-cost / {1 - high_frac:.0%} low-cost).\n\n")
        f.write(summ.to_markdown(index=False, floatfmt=".3f") + "\n\n")
        f.write(f"- Conversion high-cost vs low-cost: two-proportion p = {p_diff:.3g} ({viz.sig_stars(p_diff)}).\n")
        f.write(f"- Of all converted base-Missed hops, **{comp_high:.0%}** were high-cost.\n")


# ════════════════════════════════════════════════════════════════════════════════
# Search volume  (sharechat_search_comparison + search_volume_stats.csv) — F1
# ════════════════════════════════════════════════════════════════════════════════
def _load_sampler_calls(path: str) -> dict[str, int]:
    import json
    with open(path) as f:
        data = json.load(f)
    return {item["problem"]: item["sampler_search_calls"] for item in data}


def fig_search_volume(musique_matched_dir: str, natural_matched_dir: str,
                      sharechat_dir: str, sharechat_pairs: str, outdir: Path) -> None:
    """Per-model mean search calls across four conditions; the F1 numbers.

    MuSiQue (benchmark, blue) vs MuSiQue-Natural (user-like, red), and
    ShareChat-Benchmark (paraphrased, blue) vs ShareChat (real user, red).
    """
    import json
    import re

    # MuSiQue pair: count query_records per example, paired by example_id.
    def matched_map(d):
        out = {}
        for p in glob.glob(os.path.join(d, "matched_examples_*.json")):
            slug, counts = viz.load_search_counts_by_example(p)
            out[slug] = counts
        return out

    mus = matched_map(musique_matched_dir)
    nat = matched_map(natural_matched_dir)

    # ShareChat pair: sampler_search_calls, paired via the benchmark CSV.
    sc_real, sc_bench = {}, {}
    if os.path.isdir(sharechat_dir) and os.path.exists(sharechat_pairs):
        pairs_df = pd.read_csv(sharechat_pairs)
        run_re = re.compile(r"_run_\d+\.json$")
        real_files = {run_re.sub("", os.path.basename(f)[len("curated-sharechat_baseline_"):]): f
                      for f in glob.glob(os.path.join(sharechat_dir, "curated-sharechat_baseline_*.json"))}
        bench_files = {run_re.sub("", os.path.basename(f)[len("curated-sharechat-benchmark_baseline_"):]): f
                       for f in glob.glob(os.path.join(sharechat_dir, "curated-sharechat-benchmark_baseline_*.json"))}
        for slug in sorted(set(real_files) & set(bench_files)):
            slug_norm = slug.replace(":", "_")
            real_map = _load_sampler_calls(real_files[slug])
            bench_map = _load_sampler_calls(bench_files[slug])
            r, b = [], []
            for _, row in pairs_df.iterrows():
                if row["text"] in real_map and row["benchmark_question"] in bench_map:
                    r.append(real_map[row["text"]]); b.append(bench_map[row["benchmark_question"]])
            if r:
                sc_real[slug_norm] = np.array(r); sc_bench[slug_norm] = np.array(b)

    # Assemble per-model condition arrays.
    per_model = {}  # slug -> {cond: array}
    for slug in viz.BASE_MODEL_SLUGS:
        arrs = {}
        if slug in mus and slug in nat:
            a, b, _ = viz.paired_search_counts(mus[slug], nat[slug])
            if len(a):
                arrs["MuSiQue"], arrs["MuSiQue-Natural"] = a, b
        if slug in sc_bench and slug in sc_real:
            arrs["ShareChat-Benchmark"], arrs["ShareChat"] = sc_bench[slug], sc_real[slug]
        if arrs:
            per_model[slug] = arrs

    if not per_model:
        print("  WARNING: no search-volume data found; skipping search-volume figure")
        return

    # One panel per dataset family so a colour only ever means benchmark vs. user-like
    # phrasing (never a dataset). Blue = benchmark-style, red = user-like.
    panels = [
        ("MuSiQue", "MuSiQue", "MuSiQue-Natural"),
        ("ShareChat", "ShareChat-Benchmark", "ShareChat"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.5), sharey=True)
    w = 0.36
    stat_rows = []
    for ax, (title, c_bench, c_user) in zip(axes, panels):
        models = [s for s in viz.BASE_MODEL_SLUGS
                  if s in per_model and c_bench in per_model[s] and c_user in per_model[s]]
        x = np.arange(len(models))
        for off, cond, color, lbl in [(-w / 2, c_bench, viz.BENCHMARK, "Benchmark phrasing"),
                                      (w / 2, c_user, viz.NATURAL, "User-like phrasing")]:
            means, cis = [], []
            for slug in models:
                mean, ci95 = viz.mean_ci95(per_model[slug][cond])
                means.append(mean); cis.append(ci95 if np.isfinite(ci95) else 0.0)
            bars = ax.bar(x + off, means, w, yerr=cis, capsize=4, color=color, label=lbl,
                          error_kw={"linewidth": 1.3}, zorder=3)
            for bar, mn, ci in zip(bars, means, cis):
                ax.text(bar.get_x() + bar.get_width() / 2, mn + ci + 0.12, f"{mn:.1f}",
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
        # paired Wilcoxon + ratio annotation above each model
        for i, slug in enumerate(models):
            a = per_model[slug]
            wp = sp_stats.wilcoxon(a[c_bench], a[c_user], alternative="two-sided").pvalue
            m1, ci1 = viz.mean_ci95(a[c_bench]); m2, ci2 = viz.mean_ci95(a[c_user])
            ratio = m1 / m2 if m2 else np.nan
            top = max(m1 + ci1, m2 + ci2)
            ax.annotate(f"{ratio:.1f}x {viz.sig_stars(wp)}", xy=(x[i], top + 0.7),
                        ha="center", fontsize=11, color=viz.MISSED, fontweight="bold")
            stat_rows.append({"model": viz.display_name(slug), "pair": f"{c_bench}->{c_user}",
                              "n": len(a[c_bench]), "mean_from": m1, "ci95_from": ci1,
                              "mean_to": m2, "ci95_to": ci2, "ratio": ratio,
                              "wilcoxon_p": float(wp)})
        ax.set_xticks(x)
        ax.set_xticklabels([viz.display_name(s) for s in models], fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Mean search calls per question (95% CI)", fontsize=11)
            ax.legend(fontsize=11, frameon=False, loc="upper right")

    fig.tight_layout()
    viz.savefig(fig, outdir, "sharechat_search_comparison")

    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(outdir / "search_volume_stats.csv", index=False)
        print("\nSearch-volume (F1) — mean searches per question:")
        print(pd.DataFrame(stat_rows)[["model", "pair", "n", "mean_from", "mean_to", "ratio", "wilcoxon_p"]]
              .to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════════
# Search calibration (extended) — 6-category taxonomy per model
# ════════════════════════════════════════════════════════════════════════════════
def _as_bool(s: pd.Series) -> pd.Series:
    return s.map({True: True, False: False, "True": True, "False": False,
                  1: True, 0: False}).fillna(False).astype(bool)


# (label, parametric_certain, searched, seq_redundant, cross_hop) — None = don't filter.
# Masks are independent (categories may overlap / not sum to 100%), matching the
# original calibration definition. Left of the divider = correct decisions.
_CALIB_CATS = [
    ("Necessary\nsearch",  False, True,  None,  None),
    ("Correct\nskip",      True,  False, None,  None),
    ("Cross-hop\ncovered", False, False, None,  True),
    ("Truly\nmissed",      False, False, None,  False),
    ("Seq.\nredundant",    False, True,  True,  None),
    ("Par.\nredundant",    True,  True,  None,  None),
]


def fig_search_calibration(df_by_variant: dict[str, pd.DataFrame], outdir: Path,
                           pair: Pair = MUSIQUE_PAIR) -> None:
    """Per-model 6-category calibration taxonomy: benchmark vs user-like phrasing.

    `df_by_variant` is keyed "benchmark" (the benchmark-style condition) and "natural"
    (the user-like condition). Bars are coloured by condition (blue = benchmark,
    red = user-like) to match the rest of the paper; the x-axis carries the category
    and the dashed divider separates correct decisions (left) from search errors.
    """
    # Normalise the bool columns we slice on.
    norm = {}
    for variant, d in df_by_variant.items():
        d = d.copy()
        for col in ("parametric_certain", "seq_redundant", "missed_search_answer_found_cross_hop"):
            if col in d.columns:
                d[col] = _as_bool(d[col])
        d["searched_b"] = _as_bool(d["searched"]) if "searched" in d.columns else d["search_attributed"]
        norm[variant] = d

    x = np.arange(len(_CALIB_CATS))
    w = 0.36
    slugs = _models_present(pd.concat(list(norm.values()), ignore_index=True)) if norm else []
    for slug in slugs:
        fig, ax = plt.subplots(figsize=(11, 5))
        y_max = 0.0
        for off, variant, color, vlabel in [(-w / 2, "benchmark", viz.BENCHMARK, pair.bench_label),
                                            (w / 2, "natural", viz.NATURAL, pair.comp_label)]:
            d = norm.get(variant)
            sub = d[d["model"] == slug] if d is not None else None
            total = len(sub) if sub is not None else 0
            if not total:
                continue
            props, cnts = [], []
            for _, cert, searched, seq, cross in _CALIB_CATS:
                mask = (sub["parametric_certain"] == cert) & (sub["searched_b"] == searched)
                if seq is not None:
                    mask &= (sub["seq_redundant"] == seq)
                if cross is not None:
                    mask &= (sub["missed_search_answer_found_cross_hop"] == cross)
                c = int(mask.sum())
                props.append(100 * c / total); cnts.append(c)
            bars = ax.bar(x + off, props, w, color=color, label=f"{vlabel}  (n={total} hops)", zorder=3)
            for b, pct, c in zip(bars, props, cnts):
                if pct >= 0.5:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                            f"{pct:.0f}%", ha="center", va="bottom", fontsize=10)
            y_max = max(y_max, max(props))
        ax.axvline(2.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in _CALIB_CATS], fontsize=11)
        ax.set_ylim(0, y_max * 1.25 + 5)
        ax.set_ylabel("% of total hops", fontsize=11)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10.5, loc="upper right")
        ax.set_title(viz.display_name(slug), fontsize=13, fontweight="bold")
        fig.tight_layout()
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_")
        viz.savefig(fig, outdir, f"search_calibration_extended_{safe}")


# ════════════════════════════════════════════════════════════════════════════════
# Search redundancy  (redundancy_overlap + sequential_redundancy_distribution)
# ════════════════════════════════════════════════════════════════════════════════
def _load_example_metrics(path: str, dataset_label: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["model"] = d["model"].str.replace(":", "_", regex=False)
    d["dataset"] = dataset_label
    return d


def fig_redundancy(example_dfs: dict[str, pd.DataFrame], outdir: Path,
                   pair: Pair = MUSIQUE_PAIR) -> None:
    """Two search-redundancy figures from example-level metrics.

    `example_dfs` is keyed "benchmark" / "natural" (benchmark-style / user-like).
    """
    datasets = [d for d in ["benchmark", "natural"] if d in example_dfs]
    if not datasets:
        print("Skipping redundancy figures (no example_metrics)")
        return
    title_map = {"benchmark": pair.bench_label, "natural": pair.comp_label}

    # ── redundancy_overlap: mean queries per example, split necessary vs redundant ──
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    # good -> bad, one colour each: necessary (blue), param-only (yellow),
    # seq-only (orange), both-redundant (red)
    segs = [("necessary_count", "Necessary", viz.EFFECTIVE),
            ("parametric_only_count", "Param. redundant", viz.PR),
            ("sequential_only_count", "Seq. redundant", viz.CASCADE),
            ("both_redundant_count", "Both redundant", viz.MISSED)]
    for ax, ds in zip(axes, datasets):
        d = example_dfs[ds]
        models = _models_present(d)
        x = np.arange(len(models)); bottoms = np.zeros(len(models))
        for col, lbl, color in segs:
            vals = np.array([d[d["model"] == m][col].mean() for m in models])
            ax.bar(x, vals, 0.6, bottom=bottoms, color=color, label=lbl,
                   edgecolor="white", linewidth=0.5, zorder=3)
            bottoms += vals
        ax.set_xticks(x); ax.set_xticklabels([viz.display_name(m) for m in models], rotation=15, ha="right")
        ax.set_title(title_map.get(ds, ds), fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Mean queries per example")
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Search budget composition: necessary vs. redundant queries", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    viz.savefig(fig, outdir, "redundancy_overlap", exts=("png",))

    # ── sequential_redundancy_distribution: tail of redundant-search counts ─────────
    model_colors = {"gemini-3-pro-preview": viz.BLUE, "nemotron-3-nano_30b": viz.ORANGE,
                    "nemotron-3-nano-musique-v3-aug_latest": viz.LIGHT_BLUE,
                    "qwen3.5_122b": viz.MISSED}
    explicit = list(range(3, 10))
    bin_labels = ["1-2"] + [str(v) for v in explicit] + ["10+"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.5 * len(datasets), 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        d = example_dfs[ds].copy()
        d = d[(d["total_search_calls"] > 0) & (d["sequential_redundant_count"] >= 1)]
        models = _models_present(d)
        x = np.arange(len(bin_labels)); bw = 0.8 / max(len(models), 1)
        offs = np.linspace(-(len(models) - 1) * bw / 2, (len(models) - 1) * bw / 2, len(models))
        for i, m in enumerate(models):
            s = d[d["model"] == m]["sequential_redundant_count"].dropna()
            total = len(s)
            if not total:
                continue
            fracs = [(s <= 2).sum() / total] + [(s == v).sum() / total for v in explicit] + [(s >= 10).sum() / total]
            # Stats live in the caption/text, not the legend — legend is model name only.
            bars = ax.bar(x + offs[i], fracs, bw * 0.9, color=model_colors.get(m, viz.LIGHT_BLUE),
                          label=viz.display_name(m), zorder=3)
            for b, fr in zip(bars, fracs):
                if fr >= 0.08:
                    ax.text(b.get_x() + b.get_width() / 2, fr + 0.01, f"{fr:.0%}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=11)
        ax.set_xlabel("Sequential redundant searches per example", fontsize=11)
        ax.set_title(title_map.get(ds, ds), fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of examples", fontsize=11)
        ax.legend(fontsize=10.5)
    fig.tight_layout()
    viz.savefig(fig, outdir, "sequential_redundancy_distribution")


# ════════════════════════════════════════════════════════════════════════════════
# Knowledge tiers of Truly-missed hops  (missed_hop_knowledge_tiers)
# ════════════════════════════════════════════════════════════════════════════════
# Rebuilt from the precomputed missed_hop_knowledge.csv (the original producer script
# is absent from the repo, and the CSV is a frozen artifact). One panel per model;
# bars stack the composition of Truly-missed hops that land in CORRECT answers, split
# by each hop's closed-book knowledge tier (5/5 reliably known, 1-4/5 sometimes, 0/5
# never). The red line marks the 'known' share among misses in WRONG answers (contrast).
_KT_GREEN_DARK = "#1a9850"   # reliably known (5/5)
_KT_GREEN_LITE = "#a6d96a"   # sometimes known (1-4/5)
_KT_GREY = "#bdbdbd"         # never known (0/5)
_KT_PHRASINGS = ["Benchmark", "Natural1", "Natural2"]


def fig_knowledge_tiers(csv_path: str, outdir: Path) -> None:
    if not os.path.exists(csv_path):
        print(f"  ! knowledge-tiers CSV not found at {csv_path}; skipping fig_knowledge_tiers")
        return
    kt = pd.read_csv(csv_path)
    # The CSV is precomputed and carries display names verbatim; apply the Gemini rename.
    kt["model"] = kt["model"].replace({"Gemini 3 Pro": "Gemini 3.1 Pro"})
    models = [m for m in viz.MODEL_ORDER if m in set(kt["model"].unique())]
    if not models:
        print("  ! no known models in knowledge-tiers CSV; skipping fig_knowledge_tiers")
        return

    segs = [("f_always", "Reliably known (5/5)", _KT_GREEN_DARK, "white"),
            ("f_partial", "Sometimes known (1-4/5)", _KT_GREEN_LITE, "#333"),
            ("f_never", "Never known (0/5)", _KT_GREY, "#333")]
    phrasings = [p for p in _KT_PHRASINGS if p in set(kt["phrasing"].unique())]
    x = np.arange(len(phrasings))

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 5.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = kt[kt["model"] == model]
        correct = {p: sub[(sub.phrasing == p) & (sub.outcome == "correct")] for p in phrasings}
        wrong = {p: sub[(sub.phrasing == p) & (sub.outcome == "wrong")] for p in phrasings}
        bottoms = np.zeros(len(phrasings))
        for col, _, color, txt in segs:
            vals = np.array([float(correct[p][col].iloc[0]) if len(correct[p]) else 0.0 for p in phrasings])
            ax.bar(x, vals, 0.62, bottom=bottoms, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= 0.05:
                    ax.text(xi, b + v / 2, f"{v:.0%}", ha="center", va="center",
                            fontsize=10, color=txt, fontweight="bold")
            bottoms += vals
        # red contrast line: 'known' share among misses in WRONG answers
        for xi, p in enumerate(phrasings):
            if len(wrong[p]):
                fk = float(wrong[p]["f_known"].iloc[0])
                ax.plot([xi - 0.31, xi + 0.31], [fk, fk], color=viz.RED, linewidth=2.5, zorder=5)
        # n above each bar (correct-answer misses)
        for xi, p in enumerate(phrasings):
            n = int(correct[p]["n"].iloc[0]) if len(correct[p]) else 0
            ax.text(xi, 1.02, f"n={n}", ha="center", va="bottom", fontsize=9, color=viz.ERR_DARK)
        ax.set_xticks(x); ax.set_xticklabels(phrasings, fontsize=11, fontweight="bold")
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Composition of Truly-missed hops\nin correct answers", fontsize=11)

    handles = [mpatches.Patch(color=c, label=lbl) for _, lbl, c, _ in segs]
    handles.append(plt.Line2D([0], [0], color=viz.RED, linewidth=2.5,
                              label="'Known' share among misses in wrong answers"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    viz.savefig(fig, outdir, "missed_hop_knowledge_tiers")


# ════════════════════════════════════════════════════════════════════════════════
def run_musique(args, outdir: Path) -> None:
    """Full MuSiQue paper figure set (benchmark vs. user-like), incl. commitment-based
    figures and the cross-dataset search-volume comparison."""
    def have(path):
        return os.path.exists(path)

    mus = nat = tuned = None
    if have(args.musique_summary):
        mus = viz.load_interplay_summary(args.musique_summary, "musique", certain_rule=args.certain_rule)
    if have(args.natural_summary):
        nat = viz.load_interplay_summary(args.natural_summary, "musique-natural", certain_rule=args.certain_rule)
    if have(args.tuned_summary):
        tuned = viz.load_interplay_summary(args.tuned_summary, "musique-natural", certain_rule=args.certain_rule)

    # Taxonomy + cell-shift (need musique + musique-natural)
    if mus is not None and nat is not None:
        tax_frames = [mus, nat] + ([tuned] if tuned is not None else [])
        fig_quadrant_taxonomy(pd.concat(tax_frames, ignore_index=True), outdir, MUSIQUE_PAIR)
        fig_cell_shift_ci(pd.concat([mus, nat], ignore_index=True), outdir, MUSIQUE_PAIR)
        fig_missed_cascade(pd.concat([mus, nat], ignore_index=True), outdir, MUSIQUE_PAIR)
        fig_search_calibration({"benchmark": mus, "natural": nat}, outdir, MUSIQUE_PAIR)
    else:
        print("Skipping taxonomy / cell-shift / cascade / calibration (need musique + natural summaries)")

    # Commitment locus
    unc = load_commitment(args.commitment_dir)
    if not unc.empty:
        fig_early_commit_rate(unc, outdir)
        fig_fate_breakdown(unc, outdir)
    else:
        print("Skipping commitment figures (no commitment_locus CSVs)")

    # Search redundancy (example-level)
    em = {}
    if have(args.musique_example_metrics):
        em["benchmark"] = _load_example_metrics(args.musique_example_metrics, "musique")
    if have(args.natural_example_metrics):
        em["natural"] = _load_example_metrics(args.natural_example_metrics, "musique-natural")
    fig_redundancy(em, outdir, MUSIQUE_PAIR)

    # Missed-hop cost (needs commitment join)
    if mus is not None and nat is not None and not unc.empty:
        fig_missed_hop_cost({"benchmark": mus, "natural": nat}, args.commitment_dir, outdir)

    # SFT conversion targeting
    if nat is not None and tuned is not None and not unc.empty:
        fig_sft_conversion(nat, tuned, args.commitment_dir, outdir)
    else:
        print("Skipping SFT conversion (need natural + tuned summaries + commitment CSVs)")

    # Search volume (F1)
    fig_search_volume(args.musique_matched_dir, args.natural_matched_dir,
                      args.sharechat_dir, args.sharechat_pairs, outdir)


# ─── Phrasing accuracy/search bars (folded in from the former analyze_phrasing.py) ─
# Aggregate (not per-hop) accuracy and mean search-calls per (model, phrasing),
# sourced from the SQLite results store via results_db.paired_eval — replaces the
# old standalone analyze_phrasing_{corrected,natural2}.py scripts.
_PHRASING_COLOR = {"formal": viz.BENCHMARK, "natural": viz.NATURAL, "natural2": "#2ca02c"}
_PHRASING_FALLBACK = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]
_PHRASING_LABEL = {"formal": "Formal (benchmark)", "natural": "Natural (paraphrase 1)",
                   "natural2": "Natural2 (paraphrase 2)"}


def _phrasing_color(ph, i):
    return _PHRASING_COLOR.get(ph, _PHRASING_FALLBACK[i % len(_PHRASING_FALLBACK)])


def _phrasing_label(ph):
    return _PHRASING_LABEL.get(ph, ph.capitalize())


def _ph_correct(pe, ph):
    return pe[f"{ph}_correct"].apply(
        lambda v: 1 if (v is True or v == 1 or str(v) == "True") else 0).to_numpy()


def _ph_search(pe, ph):
    return pd.to_numeric(pe[f"{ph}_searches"], errors="coerce").fillna(0).to_numpy(float)


def _ph_mcnemar(ref_c, oth_c):
    from scipy.stats import binomtest
    b = int(((ref_c == 1) & (oth_c == 0)).sum())
    c = int(((ref_c == 0) & (oth_c == 1)).sum())
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def _ph_wilcoxon(ref_s, oth_s):
    if np.any(ref_s != oth_s):
        try:
            return sp_stats.wilcoxon(ref_s, oth_s).pvalue
        except ValueError:
            return 1.0
    return 1.0


def _phrasing_stats(conn, base_dataset, phrasings, models, grading, reference):
    rows = []
    for slug in models:
        canon = rdb.canonical_model(slug)
        pe = rdb.paired_eval(conn, base_dataset, phrasings, canon, grading=grading)
        if pe.empty:
            print(f"  ! no paired phrasing data for {slug} across {phrasings}")
            continue
        n = len(pe)
        ref_c = _ph_correct(pe, reference) if reference in phrasings else None
        ref_s = _ph_search(pe, reference) if reference in phrasings else None
        for ph in phrasings:
            c, s = _ph_correct(pe, ph), _ph_search(pe, ph)
            k = int(c.sum())
            acc, (alo, ahi) = k / n, viz.wilson_ci(k, n)
            m, hw = viz.mean_ci95(s)
            is_ref = (ph == reference)
            rows.append(dict(model=canon, phrasing=ph, metric="accuracy", n=n,
                             val=acc, lo=alo, hi=ahi,
                             p=1.0 if (is_ref or ref_c is None) else _ph_mcnemar(ref_c, c)))
            rows.append(dict(model=canon, phrasing=ph, metric="searches", n=n,
                             val=m, lo=m - hw, hi=m + hw,
                             p=1.0 if (is_ref or ref_s is None) else _ph_wilcoxon(ref_s, s)))
    return pd.DataFrame(rows)


def fig_phrasing_bars(stats_df, metric, ylabel, phrasings, models, reference, outdir, name):
    sub = stats_df[stats_df.metric == metric]
    if sub.empty:
        return
    x = np.arange(len(models))
    w = min(0.8 / len(phrasings), 0.38)
    offs = np.linspace(-(len(phrasings) - 1) * w / 2, (len(phrasings) - 1) * w / 2, len(phrasings))
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 3, 4.5))
    for i, ph in enumerate(phrasings):
        vals, los, his, ps = [], [], [], []
        for slug in models:
            canon = rdb.canonical_model(slug)
            row = sub[(sub.model == canon) & (sub.phrasing == ph)]
            if row.empty:
                vals.append(np.nan); los.append(0); his.append(0); ps.append(1.0)
            else:
                r = row.iloc[0]
                vals.append(r.val); los.append(r.val - r.lo); his.append(r.hi - r.val); ps.append(r.p)
        bars = ax.bar(x + offs[i], vals, w, yerr=[los, his], capsize=3,
                      color=_phrasing_color(ph, i), label=_phrasing_label(ph),
                      error_kw=dict(ecolor=viz.ERR_DARK, lw=1), zorder=3)
        if ph != reference:
            for bar, v, hi, p in zip(bars, vals, his, ps):
                star = viz.sig_stars(p)
                if star and np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + hi, star,
                            ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([viz.display_name(rdb.canonical_model(s)) for s in models],
                       rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(f"{ylabel} — {', '.join(phrasings)} (vs {reference})", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    viz.savefig(fig, outdir, name)


def run_phrasing_bars(outdir, phrasings, models, grading, reference, suffix, db_path):
    """Emit phrasing_accuracy_<suffix> + phrasing_searches_<suffix> from the DB."""
    if not os.path.exists(db_path):
        print(f"  ! results.db not found at {db_path}; skipping phrasing bars "
              f"(run scripts/ingest_results.py first)")
        return
    conn = rdb.connect(db_path)
    stats = _phrasing_stats(conn, "musique", phrasings, models, grading, reference)
    conn.close()
    if stats.empty:
        return
    stats.to_csv(outdir / f"phrasing_stats_{suffix}.csv", index=False)
    fig_phrasing_bars(stats, "accuracy", "Aggregate accuracy", phrasings, models,
                      reference, outdir, f"phrasing_accuracy_{suffix}")
    fig_phrasing_bars(stats, "searches", "Mean searches / example", phrasings, models,
                      reference, outdir, f"phrasing_searches_{suffix}")
    print(f"  -> phrasing_accuracy_{suffix}.png/pdf, phrasing_searches_{suffix}.png/pdf, "
          f"phrasing_stats_{suffix}.csv")


def run_natural2(args, outdir: Path) -> None:
    """Compare MuSiQue benchmark / Natural1 / Natural2 phrasings.

    Produces:
    - 3-way taxonomy (all three conditions side-by-side)
    - Cell-shift CI: benchmark vs natural2  (MUSIQUE_NAT2_PAIR)
    - Cell-shift CI: natural1  vs natural2  (NATURAL2_PAIR)
    - Missed-cascade, calibration, redundancy for both pairs
    Commitment-locus figures skipped (no probe data for natural2 yet).
    """
    have = os.path.exists

    mus = viz.load_interplay_summary(args.musique_summary, "musique", certain_rule=args.certain_rule) if have(args.musique_summary) else None
    nat = viz.load_interplay_summary(args.natural_summary, "musique-natural", certain_rule=args.certain_rule) if have(args.natural_summary) else None
    nat2_raw = viz.load_interplay_summary(args.natural2_summary, "musique-natural2", certain_rule=args.certain_rule) if have(args.natural2_summary) else None
    nat2 = _normalize_natural2_model_names(nat2_raw) if nat2_raw is not None else None
    # SFT/tuned Nemotron, evaluated on the natural (Nat1) phrasing — shown alongside Nat1.
    tuned = viz.load_interplay_summary(args.tuned_summary, "musique-natural", certain_rule=args.certain_rule) if have(args.tuned_summary) else None

    if nat2 is None:
        print(f"ERROR: natural2 summary not found at {args.natural2_summary}")
        return

    common_bench = set(mus["model"].unique()) & set(nat2["model"].unique()) if mus is not None else set()
    common_nat   = set(nat["model"].unique())  & set(nat2["model"].unique()) if nat  is not None else set()
    print(f"  Models benchmark∩nat2: {sorted(common_bench)}")
    print(f"  Models natural1∩nat2 : {sorted(common_nat)}")

    # ── 3-way taxonomy (benchmark + natural1 + natural2) ────────────────────────
    all_frames = [f for f in [mus, nat, nat2] if f is not None]
    if len(all_frames) > 1:
        three_pair_full = Pair(
            "musique", "musique-natural2",
            "Benchmark (MuSiQue)", "MuSiQue-Natural2",
            "MuSiQue", "Nat2", "MuSiQue vs Natural2", "benchmark vs. natural2 phrasing", "Natural2")
        # Override ds_rank to give nat1 a rank between benchmark and nat2.
        three_pair_full.ds_rank = {"musique": 0, "musique-natural": 1, "musique-natural2": 2}
        three_pair_full.ds_short = {"musique": "MuSiQue", "musique-natural": "Nat1", "musique-natural2": "Nat2"}
        combined_all = pd.concat(all_frames, ignore_index=True)
        fig_quadrant_taxonomy(combined_all, outdir, three_pair_full)
        # Simple taxonomy also shows the SFT Nemotron over Nat1 when available.
        simple_frames = all_frames + ([tuned] if tuned is not None else [])
        fig_quadrant_taxonomy_simple(pd.concat(simple_frames, ignore_index=True), outdir, three_pair_full)

    # ── Benchmark vs Natural2 ────────────────────────────────────────────────────
    if mus is not None and common_bench:
        combined_bn2 = pd.concat([mus, nat2], ignore_index=True)
        viz.savefig  # ensure module loaded
        fig_cell_shift_ci(combined_bn2, outdir, MUSIQUE_NAT2_PAIR)
        # rename output so it doesn't overwrite the nat1-vs-nat2 version
        for ext in ("png", "pdf"):
            src = outdir / f"fig1c_cell_shift_ci.{ext}"
            dst = outdir / f"fig1c_cell_shift_ci_benchmark_vs_nat2.{ext}"
            if src.exists():
                src.rename(dst)
        fig_missed_cascade(combined_bn2, outdir, MUSIQUE_NAT2_PAIR)
        for ext in ("png",):
            src = outdir / f"fig_missed_cascade_bar.{ext}"
            dst = outdir / f"fig_missed_cascade_bar_benchmark_vs_nat2.{ext}"
            if src.exists():
                src.rename(dst)
        fig_search_calibration({"benchmark": mus, "natural": nat2}, outdir, MUSIQUE_NAT2_PAIR)

    # ── Natural1 vs Natural2 ─────────────────────────────────────────────────────
    if nat is not None and common_nat:
        combined_n1n2 = pd.concat([nat, nat2], ignore_index=True)
        fig_cell_shift_ci(combined_n1n2, outdir, NATURAL2_PAIR)
        for ext in ("png", "pdf"):
            src = outdir / f"fig1c_cell_shift_ci.{ext}"
            dst = outdir / f"fig1c_cell_shift_ci_nat1_vs_nat2.{ext}"
            if src.exists():
                src.rename(dst)
        fig_missed_cascade(combined_n1n2, outdir, NATURAL2_PAIR)
        for ext in ("png",):
            src = outdir / f"fig_missed_cascade_bar.{ext}"
            dst = outdir / f"fig_missed_cascade_bar_nat1_vs_nat2.{ext}"
            if src.exists():
                src.rename(dst)

    # ── Search redundancy ────────────────────────────────────────────────────────
    em = {}
    if have(args.musique_example_metrics):
        em["benchmark"] = _load_example_metrics(args.musique_example_metrics, "musique")
    if have(args.natural2_example_metrics):
        em["natural"] = _load_example_metrics(args.natural2_example_metrics, "musique-natural2")
        if "natural" in em:
            em["natural"] = _normalize_natural2_model_names(em["natural"])
    fig_redundancy(em, outdir, MUSIQUE_NAT2_PAIR)

    # ── Commitment locus: benchmark (formal MuSiQue) vs natural2 ─────────────────
    # The probe's `benchmark` variant is formal MuSiQue and `natural` is natural2,
    # so this is the MUSIQUE_NAT2_PAIR comparison. Outputs are renamed with a
    # _benchmark_vs_nat2 suffix so they don't collide with the --mode musique
    # (benchmark vs Nat1) commitment figures.
    unc2 = load_commitment(args.natural2_commitment_dir)
    if not unc2.empty:
        fig_early_commit_rate(unc2, outdir, MUSIQUE_NAT2_PAIR)
        fig_fate_breakdown(unc2, outdir, MUSIQUE_NAT2_PAIR)
        for base in ("fig1_early_commit_rate", "fig2_fate_breakdown"):
            for ext in ("png", "pdf"):
                src = outdir / f"{base}.{ext}"
                dst = outdir / f"{base}_benchmark_vs_nat2.{ext}"
                if src.exists():
                    src.rename(dst)
        print("  -> fig1_early_commit_rate_benchmark_vs_nat2.png, "
              "fig2_fate_breakdown_benchmark_vs_nat2.png")
    else:
        print(f"  ! no commitment CSVs in {args.natural2_commitment_dir}; "
              "skipping natural2 commitment figures")

    # ── Phrasing accuracy / search bars: formal vs natural vs natural2 ───────────
    # Aggregate accuracy + search calls across all 3 models, from the results DB.
    run_phrasing_bars(outdir, ["formal", "natural", "natural2"], viz.BASE_MODEL_SLUGS,
                      args.phrasing_grading, "formal", "natural2", args.db)

    # ── Knowledge tiers of Truly-missed hops (appendix) ──────────────────────────
    # Rebuilt from the frozen missed_hop_knowledge.csv that lives in this output dir.
    kt_csv = args.knowledge_tiers_csv or str(outdir / "missed_hop_knowledge.csv")
    fig_knowledge_tiers(kt_csv, outdir)

    print(f"\nnatural2 artifacts written to {outdir}/")


def run_sharechat(args, outdir: Path) -> None:
    """Dataset-agnostic figures on the curated-sharechat pair
    (curated-sharechat-benchmark -> curated-sharechat).

    ShareChat has no commitment-locus probe and no gold hops, so the commitment,
    missed-hop-cost and SFT figures are not produced. The search-volume comparison
    is produced by the default `--mode musique` run.
    """
    def have(path):
        return os.path.exists(path)

    pair = SHARECHAT_PAIR
    bench = comp = None
    if have(args.sc_benchmark_summary):
        bench = viz.load_interplay_summary(args.sc_benchmark_summary, pair.bench_ds, override_entropy=False, certain_rule=args.certain_rule)
    if have(args.sc_summary):
        comp = viz.load_interplay_summary(args.sc_summary, pair.comp_ds, override_entropy=False, certain_rule=args.certain_rule)
    if bench is None or comp is None:
        print("Skipping ShareChat figures: need both "
              f"{args.sc_benchmark_summary} and {args.sc_summary}")
        return

    combined = pd.concat([bench, comp], ignore_index=True)
    fig_quadrant_taxonomy(combined, outdir, pair)
    fig_cell_shift_ci(combined, outdir, pair)
    fig_missed_cascade(combined, outdir, pair)
    fig_search_calibration({"benchmark": bench, "natural": comp}, outdir, pair)

    em = {}
    if have(args.sc_benchmark_example_metrics):
        em["benchmark"] = _load_example_metrics(args.sc_benchmark_example_metrics, pair.bench_ds)
    if have(args.sc_example_metrics):
        em["natural"] = _load_example_metrics(args.sc_example_metrics, pair.comp_ds)
    fig_redundancy(em, outdir, pair)

    # Commitment-locus probe exists for ShareChat too (benchmark/natural = paraphrase/real).
    unc = load_commitment(args.sc_commitment_dir, override_entropy=False)
    if not unc.empty:
        fig_early_commit_rate(unc, outdir, pair)
        fig_fate_breakdown(unc, outdir, pair)
        fig_missed_hop_cost({"benchmark": bench, "natural": comp}, args.sc_commitment_dir, outdir, pair)
    else:
        print(f"Skipping ShareChat commitment / missed-cost figures (no CSVs in {args.sc_commitment_dir})")


# ════════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["musique", "sharechat", "natural2"], default="musique",
                   help="musique: full paper figure set; "
                        "sharechat: dataset-agnostic figures on the curated-sharechat pair; "
                        "natural2: compare MuSiQue-Natural vs MuSiQue-Natural2 phrasings")
    p.add_argument("--output-dir", default=None,
                   help="default: results/paper_figures (musique) or "
                        "results/curated_sharechat/paper_figures (sharechat)")
    p.add_argument("--musique-summary",
                   default="results/musique_parametric/interplay_analysis/interplay_summary.csv")
    p.add_argument("--natural-summary",
                   default="results/musique-natural/interplay_analysis/interplay_summary.csv")
    p.add_argument("--tuned-summary",
                   default="results/musique-natural/interplay_analysis_tuned/interplay_summary.csv")
    p.add_argument("--commitment-dir", default="results/commitment_locus")
    p.add_argument("--musique-matched-dir",
                   default="results/musique_parametric/interplay_analysis")
    p.add_argument("--natural-matched-dir",
                   default="results/musique-natural/interplay_analysis")
    p.add_argument("--sharechat-dir", default="results/curated_sharechat")
    p.add_argument("--sharechat-pairs", default="data/curated_sharechat_wildchat_benchmark.csv")
    p.add_argument("--musique-example-metrics",
                   default="results/musique_parametric/interplay_analysis/example_metrics.csv")
    p.add_argument("--natural-example-metrics",
                   default="results/musique-natural/interplay_analysis/example_metrics.csv")
    # natural2 (--mode natural2) inputs
    p.add_argument("--natural2-summary",
                   default="results/musique-natural2/interplay_analysis/interplay_summary.csv")
    p.add_argument("--natural2-example-metrics",
                   default="results/musique-natural2/interplay_analysis/example_metrics.csv")
    p.add_argument("--natural2-commitment-dir",
                   default="results/musique-natural2/commitment_locus",
                   help="Commitment-locus CSVs for the benchmark-vs-natural2 probe.")
    p.add_argument("--knowledge-tiers-csv", default=None,
                   help="Precomputed missed_hop_knowledge.csv for the appendix knowledge-tiers "
                        "figure (default: <output-dir>/missed_hop_knowledge.csv).")
    # ShareChat (--mode sharechat) inputs
    p.add_argument("--sc-benchmark-summary",
                   default="results/curated_sharechat/benchmark_interplay_analysis/interplay_summary.csv")
    p.add_argument("--sc-summary",
                   default="results/curated_sharechat/interplay_analysis/interplay_summary.csv")
    p.add_argument("--sc-benchmark-example-metrics",
                   default="results/curated_sharechat/benchmark_interplay_analysis/example_metrics.csv")
    p.add_argument("--sc-example-metrics",
                   default="results/curated_sharechat/interplay_analysis/example_metrics.csv")
    p.add_argument("--sc-commitment-dir", default="results/curated_sharechat/commitment_locus")
    # Phrasing accuracy/search bars (natural2 mode) read aggregate eval results
    # from the SQLite results store.
    p.add_argument("--db", default=rdb.DEFAULT_DB_PATH,
                   help="SQLite results store for phrasing accuracy/search bars.")
    p.add_argument("--phrasing-grading", default="reevaluated",
                   choices=["reevaluated", "original"],
                   help="Which grading version to use for the phrasing bars.")
    p.add_argument("--certain-rule", default="joint", choices=["joint", "entropy"],
                   help="Parametric-knowledge rule: 'entropy' = semantic entropy alone "
                        "(entropy == 0); 'joint' = entropy AND parametric probe correct "
                        "(default).")
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = {
            "sharechat": "results/curated_sharechat/paper_figures",
            "natural2":  "results/natural2_paper_figures",
        }.get(args.mode, "results/paper_figures")

    viz.apply_theme()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sharechat":
        run_sharechat(args, outdir)
    elif args.mode == "natural2":
        run_natural2(args, outdir)
    else:
        run_musique(args, outdir)

    print(f"\nAll {args.mode} artifacts written to {outdir}/")


if __name__ == "__main__":
    main()
