"""
Analyze commitment-locus probe results across all models.

Core question: do natural-phrased questions trigger earlier/unsearched commits
compared to benchmark-phrased questions on the same underlying hops?

Produces 3 focused figures:
  fig1 — early-commit rate (direct answer to the core question)
  fig2 — commitment fate breakdown per model (3 categories)
  fig3 — first-commitment turn-index distribution
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion. Returns (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── Aesthetics ──────────────────────────────────────────────────────────────
BM_COLOR  = "#3498db"   # blue  = benchmark
NAT_COLOR = "#9b59b6"   # purple = natural

CAT_COLORS = {
    "commit_after_search": "#27ae60",   # green  = correct
    "commit_no_search":    "#e67e22",   # orange = early / without search
    "no_commit":           "#bdc3c7",   # gray   = never committed
}
CAT_LABELS = {
    "commit_after_search": "Committed after search",
    "commit_no_search":    "Committed without searching",
    "no_commit":           "Never committed",
}
CAT_ORDER = ["commit_after_search", "commit_no_search", "no_commit"]

MODEL_ORDER = ["Gemini 3 Pro", "Nemotron Nano 30B", "Qwen 3.5 122B"]
SLUG_TO_LABEL = {
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "nemotron-3-nano_30b":  "Nemotron Nano 30B",
    "qwen3.5_122b":         "Qwen 3.5 122B",
}


SLUG_TO_UNCERTAINTY_JSON = {
    "gemini-3-pro-preview": "results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json",
    "nemotron-3-nano_30b":  "results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json",
    "qwen3.5_122b":         "results/musique_parametric/musique_parametric_uncertainty_qwen3.5_122b.json",
}


def setup_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="results/commitment_locus")
    p.add_argument("--output-dir", default="results/consolidated_analysis/commitment_locus")
    return p.parse_args()


VAL_SET_SIZE = 600  # first N records in each uncertainty JSON are the validation set


def load_canonical_entropy() -> dict[tuple[str, int], float]:
    """Load semantic_entropy from the authoritative parametric uncertainty JSONs.
    Returns {(example_id, hop_index): entropy}.
    Only the first VAL_SET_SIZE records are used — JSONs may contain additional
    training-set entries beyond that position.
    """
    canon = {}
    for slug, path in SLUG_TO_UNCERTAINTY_JSON.items():
        if not os.path.exists(path):
            print(f"  WARNING: uncertainty JSON not found: {path}")
            continue
        import json
        data = json.load(open(path))
        for rec in data[:VAL_SET_SIZE]:
            for sq in rec.get("sub_questions_results", []):
                canon[(rec["example_id"], sq["hop_index"])] = sq["semantic_entropy"]
    return canon


def load(input_dir: str) -> pd.DataFrame:
    frames = []
    for slug, label in SLUG_TO_LABEL.items():
        pattern = os.path.join(input_dir, f"commitment_locus_{slug}_shard_*.csv")
        files = glob.glob(pattern)
        if not files:
            print(f"  WARNING: no files for {slug}")
            continue
        for f in files:
            sub = pd.read_csv(f)
            sub["model"] = label
            frames.append(sub)
    df = pd.concat(frames, ignore_index=True)

    # Overwrite semantic_entropy from the single authoritative source so both
    # variants always use identical values for the same (example_id, hop_index).
    canon = load_canonical_entropy()
    df["semantic_entropy"] = df.apply(
        lambda r: canon.get((r["example_id"], r["hop_index"]), r["semantic_entropy"]), axis=1
    )

    df["committed"]      = df["first_committed_turn_index"] >= 0
    df["searched_for_hop"] = df["first_search_turn_for_hop"] >= 0

    def categorize(row):
        if not row["committed"]:
            return "no_commit"
        # Collapse commit_before_search into commit_no_search for simplicity
        # (it's always tiny; merging avoids confusing readers)
        if not row["searched_for_hop"] or row["committed_before_search"] == True:
            return "commit_no_search"
        return "commit_after_search"

    df["category"]     = df.apply(categorize, axis=1)
    df["early_commit"] = df["category"] == "commit_no_search"
    df["model"]        = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


# ── Fig 1: core answer — early-commit rate ──────────────────────────────────
def fig1_early_commit_rate(unc: pd.DataFrame, outdir: str):
    """Grouped bar per model. Each bar shows the fraction of uncertain hops
    where the model committed without searching. Wilson 95% CI error bars.
    Raw counts (k / N hops) shown below each bar. Chi-square significance shown
    as stars above the natural bar delta annotation."""
    grp    = unc.groupby(["model", "variant"])["early_commit"]
    rates  = grp.mean().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)
    counts = grp.sum().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)
    ns     = unc.groupby(["model", "variant"]).size().unstack()[["benchmark", "natural"]].reindex(MODEL_ORDER)

    def ci_err(model, variant):
        k, n = int(counts.loc[model, variant]), int(ns.loc[model, variant])
        lo, hi = wilson_ci(k, n)
        p = rates.loc[model, variant]
        return p - lo, hi - p

    n_models = len(MODEL_ORDER)
    x = np.arange(n_models)
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 6))

    for offset, variant, color, label in [
        (-w/2, "benchmark", BM_COLOR,  "Benchmark (formal)"),
        ( w/2, "natural",   NAT_COLOR, "Natural (paraphrase)"),
    ]:
        heights = rates[variant].values
        errs_lo = np.array([ci_err(m, variant)[0] for m in MODEL_ORDER])
        errs_hi = np.array([ci_err(m, variant)[1] for m in MODEL_ORDER])
        bars = ax.bar(x + offset, heights, w, color=color, label=label, zorder=3)
        ax.errorbar(x + offset, heights, yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=4, capthick=1.5,
                    elinewidth=1.5, zorder=4)
        for i, (bar, h, model) in enumerate(zip(bars, heights, MODEL_ORDER)):
            k = int(counts.loc[model, variant])
            n = int(ns.loc[model, variant])
            bx = bar.get_x() + bar.get_width() / 2
            # percentage above bar
            ax.text(bx, h + 0.025, f"{h:.0%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
            # raw count below bar
            ax.text(bx, -0.055, f"{k}/{n}", ha="center", va="top",
                    fontsize=8, color="#555")

    # Significance stars + delta above each natural bar
    for i, model in enumerate(MODEL_ORDER):
        k_b = int(counts.loc[model, "benchmark"]); n_b = int(ns.loc[model, "benchmark"])
        k_n = int(counts.loc[model, "natural"]);   n_n = int(ns.loc[model, "natural"])
        _, p, _, _ = chi2_contingency([[k_b, n_b - k_b], [k_n, n_n - k_n]])
        stars = sig_stars(p)
        b_rate, n_rate = rates.loc[model, "benchmark"], rates.loc[model, "natural"]
        ax.annotate(f"+{n_rate-b_rate:.0%} {stars}",
                    xy=(x[i] + w/2, n_rate + 0.062),
                    ha="center", fontsize=9, color=NAT_COLOR, fontweight="bold")

    # Secondary x-axis labels: "N=XXX uncertain hops" per model
    ax2 = ax.secondary_xaxis("bottom")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{MODEL_ORDER[i]}\n(bm: {int(ns.loc[m,'benchmark'])} | nat: {int(ns.loc[m,'natural'])} uncertain hops)"
         for i, m in enumerate(MODEL_ORDER)],
        fontsize=8.5, color="#444"
    )
    ax2.tick_params(length=0, pad=28)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", pad=3)
    ax.set_ylabel("Fraction of uncertain hops\nwhere model committed without searching", fontsize=10)
    ax.set_ylim(-0.02, 0.65)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}" if y >= 0 else ""))
    ax.set_title("Natural questions trigger more unsearched commits\n(parametrically uncertain hops only)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.text(0.99, 0.02, "* p<0.05  ** p<0.01  *** p<0.001 (χ²)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#777")

    fig.tight_layout()
    path = os.path.join(outdir, "fig1_early_commit_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 2: full fate breakdown ───────────────────────────────────────────────
def fig2_fate_breakdown(unc: pd.DataFrame, outdir: str):
    """3-category stacked bar. Each bar is labelled with its raw count (k hops).
    N total uncertain hops shown below each x-tick. Wilson 95% CI error bars
    on the orange segment."""
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Commitment fate on uncertain hops — benchmark vs natural",
                 fontsize=13, fontweight="bold")

    variants = ["benchmark", "natural"]
    x = np.arange(len(variants))

    for ax, model in zip(axes, MODEL_ORDER):
        sub = unc[unc["model"] == model]
        n_total = sub.groupby("variant").size().reindex(variants).values  # [n_bm, n_nat]

        counts = (
            sub.groupby(["variant", "category"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=variants, columns=CAT_ORDER, fill_value=0)
        )
        fracs = counts.div(n_total, axis=0)
        bottoms = np.zeros(len(variants))
        orange_tops = np.zeros(len(variants))

        for cat in CAT_ORDER:
            vals = fracs[cat].values
            raw  = counts[cat].values
            ax.bar(x, vals, bottom=bottoms, color=CAT_COLORS[cat],
                   edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
            for xi, (v, b, k) in enumerate(zip(vals, bottoms, raw)):
                if v > 0.05:
                    ax.text(xi, b + v/2, f"{v:.0%}\n({k})",
                            ha="center", va="center", fontsize=9,
                            color="white" if cat != "no_commit" else "#444",
                            fontweight="bold", linespacing=1.3)
            if cat == "commit_no_search":
                orange_tops = bottoms + vals
            bottoms += vals

        # 95% Wilson CI on the orange segment
        errs_lo, errs_hi = [], []
        for vi, variant in enumerate(variants):
            k = int(counts.loc[variant, "commit_no_search"])
            n = int(n_total[vi])
            lo, hi = wilson_ci(k, n)
            p = fracs.loc[variant, "commit_no_search"]
            errs_lo.append(p - lo)
            errs_hi.append(hi - p)
        ax.errorbar(x, orange_tops, yerr=[errs_lo, errs_hi],
                    fmt="none", color="#c0392b", capsize=5, capthick=1.8,
                    elinewidth=1.8, zorder=5)

        # x-tick labels with N
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"Benchmark\n(N={n_total[0]})", f"Natural\n(N={n_total[1]})"],
            fontsize=9.5
        )
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Fraction of uncertain hops", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(axis="y", alpha=0.2, zorder=0)

    handles = [mpatches.Patch(color=CAT_COLORS[c], label=CAT_LABELS[c]) for c in CAT_ORDER]
    handles.append(plt.Line2D([0], [0], color="#c0392b", linewidth=2, label="95% CI (commit w/o search)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_fate_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 3: commitment-turn distribution ─────────────────────────────────────
def fig3_turn_distribution(unc: pd.DataFrame, outdir: str):
    """Standard boxplot (no notches) of first_committed_turn_index for committed
    uncertain hops. Mean shown as a white diamond. n_committed and median
    annotated per box."""
    committed = unc[unc["committed"]].copy()

    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(13, 5.5), sharey=True)
    fig.suptitle("When does the model first commit to a hop answer?\n"
                 "(uncertain hops that received a commitment; mean shown as ◆)",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, MODEL_ORDER):
        sub = committed[committed["model"] == model]
        data_b = sub[sub["variant"] == "benchmark"]["first_committed_turn_index"].values
        data_n = sub[sub["variant"] == "natural"]["first_committed_turn_index"].values

        bp = ax.boxplot(
            [data_b, data_n],
            positions=[0, 1],
            widths=0.45,
            notch=False,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.35),
        )
        for patch, color in zip(bp["boxes"], [BM_COLOR, NAT_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for i, (data, color) in enumerate([(data_b, BM_COLOR), (data_n, NAT_COLOR)]):
            med  = np.median(data)
            mean = np.mean(data)
            n    = len(data)
            # mean diamond
            ax.scatter([i], [mean], marker="D", s=55, color="white",
                       edgecolors="black", linewidths=0.8, zorder=5)
            # median label inside box
            ax.text(i, med - 1.1, f"med={med:.0f}", ha="center", fontsize=8.5,
                    color="white", fontweight="bold", va="top")
            # n_committed + mean below x-axis
            ax.text(i, -3.2, f"n={n}\nmean={mean:.1f}", ha="center", fontsize=8,
                    color="#444", va="top", linespacing=1.3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benchmark", "Natural"], fontsize=10)
        ax.tick_params(axis="x", pad=30)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Trace turn index of first commitment\n(0 = first message)", fontsize=10)
        ax.set_ylim(-5, 25)
        ax.set_yticks(range(0, 26, 5))
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = os.path.join(outdir, "fig3_turn_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_summary(unc: pd.DataFrame):
    print("\n" + "=" * 65)
    print("CORE RESULT: Natural phrasing → more unsearched commits")
    print("=" * 65)
    rates = (
        unc.groupby(["model", "variant"])["early_commit"]
        .mean()
        .unstack()[["benchmark", "natural"]]
        .reindex(MODEL_ORDER)
    )
    rates["delta"] = rates["natural"] - rates["benchmark"]
    rates["ratio"] = rates["natural"] / rates["benchmark"]
    print(rates.applymap(lambda x: f"{x:.1%}" if x < 5 else f"{x:.1f}×").to_string())

    print()
    print("Correct (commit after search) rate on uncertain hops:")
    correct = (
        unc.groupby(["model", "variant"])
        .apply(lambda x: (x["category"] == "commit_after_search").mean(), include_groups=False)
        .unstack()[["benchmark", "natural"]]
        .reindex(MODEL_ORDER)
    )
    print(correct.applymap(lambda x: f"{x:.1%}").to_string())
    print("=" * 65)


def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading from {args.input_dir}...")
    df = load(args.input_dir)
    print(f"  {len(df)} rows across {df['model'].nunique()} models.")

    unc = df[df["semantic_entropy"] > 0].copy()
    print(f"  {len(unc)} rows with parametric entropy > 0 (uncertain hops).")

    print_summary(unc)
    fig1_early_commit_rate(unc, args.output_dir)
    fig2_fate_breakdown(unc, args.output_dir)
    fig3_turn_distribution(unc, args.output_dir)
    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
