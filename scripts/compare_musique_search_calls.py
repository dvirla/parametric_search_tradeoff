"""
Compare search call counts and quadrant behavior between musique (formal) and
musique-natural (paraphrased) datasets, matched by example_id.

Analogous to compare_sharechat_search_calls.py but for the two MusiQue variants.

Outputs:
  musique_search_comparison.png  — violin + summary bar chart per model
  musique_quadrant_shift.png     — quadrant transition heatmaps (one panel per model)
  musique_search_comparison_stats.csv — paired stats table

Usage:
    uv run python scripts/compare_musique_search_calls.py \\
      --parametric-examples \\
        results/musique_parametric/interplay_analysis/matched_examples_gemini-3-pro-preview.json \\
        results/musique_parametric/interplay_analysis/matched_examples_qwen3.5_122b.json \\
        results/musique_parametric/interplay_analysis/matched_examples_nemotron-3-nano_30b.json \\
      --natural-examples \\
        results/musique-natural/interplay_analysis/matched_examples_gemini-3-pro-preview.json \\
        results/musique-natural/interplay_analysis/matched_examples_qwen3.5_122b.json \\
        results/musique-natural/interplay_analysis/matched_examples_nemotron-3-nano_30b.json \\
      --parametric-summary results/musique_parametric/interplay_analysis/interplay_summary.csv \\
      --natural-summary results/musique-natural/interplay_analysis/interplay_summary.csv \\
      --output results/consolidated_analysis/musique_search_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.1)

MODEL_DISPLAY = {
    "gemini-3-pro-preview": "Gemini 3.1 Pro",
    "nemotron-3-nano_30b": "Nemotron Nano",
    "nemotron-3-nano": "Nemotron Nano",
    "qwen3.5_122b": "Qwen 3.5",
}

QUADRANT_COLORS = {
    "E":  "#4CAF50",
    "CP": "#2196F3",
    "PR": "#FF9800",
    "M":  "#F44336",
}

COLORS = ["#2196F3", "#1565C0"]  # formal = lighter blue, natural = dark blue


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def load_matched_examples(path: str) -> dict[str, int]:
    """Return {example_id: total_search_calls} from a matched_examples JSON."""
    with open(path) as f:
        data = json.load(f)
    model = data[0]["model"] if data else "unknown"
    model = model.replace(":", "_")
    result = {}
    for ex in data:
        eid = ex["example_id"]
        result[eid] = len(ex.get("query_records", []))
    return model, result


def build_paired_arrays(
    param_map: dict[str, int], nat_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common = sorted(set(param_map) & set(nat_map))
    param_arr = np.array([param_map[eid] for eid in common])
    nat_arr = np.array([nat_map[eid] for eid in common])
    return param_arr, nat_arr, common


def mean_ci95(arr: np.ndarray) -> tuple[float, float]:
    n = len(arr)
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return arr.mean(), t_crit * sem


def run_tests(param: np.ndarray, nat: np.ndarray) -> dict:
    wilcox = stats.wilcoxon(param, nat, alternative="two-sided")
    ttest = stats.ttest_rel(param, nat)
    param_mean, param_ci = mean_ci95(param)
    nat_mean, nat_ci = mean_ci95(nat)
    return {
        "n": len(param),
        "formal_mean": param_mean,
        "formal_ci95": param_ci,
        "formal_median": float(np.median(param)),
        "natural_mean": nat_mean,
        "natural_ci95": nat_ci,
        "natural_median": float(np.median(nat)),
        "wilcox_stat": wilcox.statistic,
        "wilcox_p": wilcox.pvalue,
        "ttest_t": ttest.statistic,
        "ttest_p": ttest.pvalue,
    }


def pval_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_violin(ax, param: np.ndarray, nat: np.ndarray, stats_: dict, model: str):
    vp = ax.violinplot([param, nat], positions=[0, 1], showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.7)

    for x, arr, color in zip([0, 1], [param, nat], COLORS):
        ax.scatter([x], [arr.mean()], marker="D", color=color, zorder=5, s=45)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Formal\n(MusiQue)", "Natural\n(MusiQue-Nat)"])
    ax.set_ylabel("Search calls")
    ax.set_title(display_name(model), fontsize=10, fontweight="bold")

    w_p, t_p = stats_["wilcox_p"], stats_["ttest_p"]
    ann = (
        f"Wilcoxon: p={w_p:.3f} {pval_stars(w_p)}\n"
        f"Paired t:  p={t_p:.3f} {pval_stars(t_p)}\n"
        f"n={stats_['n']}"
    )
    ax.text(0.03, 0.97, ann, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def plot_summary(ax, all_stats: dict[str, dict]):
    models = list(all_stats.keys())
    labels = [display_name(m) for m in models]
    x = np.arange(len(models))
    w = 0.35

    formal_means = np.array([all_stats[m]["formal_mean"] for m in models])
    nat_means = np.array([all_stats[m]["natural_mean"] for m in models])
    formal_cis = np.array([all_stats[m]["formal_ci95"] for m in models])
    nat_cis = np.array([all_stats[m]["natural_ci95"] for m in models])

    ax.bar(x - w / 2, formal_means, w, yerr=formal_cis, capsize=5,
           color=COLORS[0], alpha=0.8, label="Formal (MusiQue)", error_kw={"linewidth": 1.5})
    ax.bar(x + w / 2, nat_means, w, yerr=nat_cis, capsize=5,
           color=COLORS[1], alpha=0.8, label="Natural (MusiQue-Nat)", error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean search calls (95% CI)")
    ax.set_title("Mean search calls across models", fontweight="bold")
    ax.legend()

    for i, m in enumerate(models):
        p = all_stats[m]["wilcox_p"]
        y = nat_means[i] + nat_cis[i] + 0.3
        ax.text(x[i] + w / 2, y, pval_stars(p), ha="center", va="bottom", fontsize=11)

    ax.text(0.01, 0.99, "* p<0.05  ** p<0.01  *** p<0.001  ns: not sig.\n(Wilcoxon signed-rank, error bars = 95% CI)",
            transform=ax.transAxes, va="top", fontsize=7, color="gray")


def load_summary_for_quadrants(path: str, dataset_label: str) -> pd.DataFrame:
    """Load interplay_summary.csv and assign quadrants."""
    df = pd.read_csv(path)
    df["model"] = df["model"].str.replace(":", "_", regex=False)
    df = df.rename(columns={"semantic_entropy": "entropy", "searched": "search_attributed"})
    df["search_attributed"] = df["search_attributed"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)
    df["certain"] = (df["entropy"] == 0.0) & (df["parametric_certain"].astype(str) == "True")
    conditions = [
        (~df["certain"]) & df["search_attributed"],
        (~df["certain"]) & (~df["search_attributed"]),
        df["certain"] & df["search_attributed"],
        df["certain"] & (~df["search_attributed"]),
    ]
    df["quadrant"] = np.select(conditions, ["E", "M", "PR", "CP"], default="?")
    df["dataset"] = dataset_label
    return df[["dataset", "model", "example_id", "hop_index", "quadrant"]]


def plot_quadrant_shift(param_df: pd.DataFrame, nat_df: pd.DataFrame,
                        out_path: Path, stats_path: Path) -> None:
    """Quadrant transition heatmaps (formal → natural) per model."""
    param_df = param_df.rename(columns={"quadrant": "q_formal"})
    nat_df = nat_df.rename(columns={"quadrant": "q_natural"})

    merged = param_df.merge(
        nat_df[["model", "example_id", "hop_index", "q_natural"]],
        on=["model", "example_id", "hop_index"],
        how="inner",
    )
    if merged.empty:
        print("Warning: no matched hops for quadrant shift plot; skipping")
        return

    models = sorted(merged["model"].unique())
    n_models = len(models)
    quadrants = ["E", "CP", "PR", "M"]

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    shift_rows = []
    for ax, model in zip(axes, models):
        sub = merged[merged["model"] == model]
        counts = pd.crosstab(sub["q_formal"], sub["q_natural"])
        counts = counts.reindex(index=quadrants, columns=quadrants, fill_value=0)
        # Normalize rows (fraction of formal-Q going to each natural-Q)
        row_sums = counts.sum(axis=1).replace(0, np.nan)
        norm = counts.div(row_sums, axis=0).fillna(0)

        sns.heatmap(norm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=quadrants, yticklabels=quadrants,
                    vmin=0, vmax=1, cbar=ax == axes[-1])
        ax.set_title(display_name(model), fontweight="bold")
        ax.set_xlabel("Natural quadrant")
        ax.set_ylabel("Formal quadrant")

        for qf in quadrants:
            for qn in quadrants:
                raw = counts.loc[qf, qn] if qf in counts.index and qn in counts.columns else 0
                shift_rows.append({"model": model, "formal_q": qf, "natural_q": qn,
                                   "count": int(raw),
                                   "fraction": float(norm.loc[qf, qn]) if qf in norm.index else 0.0})

    fig.suptitle("Quadrant transition: Formal → Natural (row-normalised)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    pd.DataFrame(shift_rows).to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parametric-examples", nargs="+", required=True,
                        help="matched_examples JSONs from musique_parametric (one per model)")
    parser.add_argument("--natural-examples", nargs="+", required=True,
                        help="matched_examples JSONs from musique-natural (one per model)")
    parser.add_argument("--parametric-summary", default=None,
                        help="interplay_summary.csv for musique (for quadrant shift plot)")
    parser.add_argument("--natural-summary", default=None,
                        help="interplay_summary.csv for musique-natural (for quadrant shift plot)")
    parser.add_argument("--output", default="results/consolidated_analysis/musique_search_comparison.png")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all matched_examples and build per-model dicts
    print("Loading parametric matched examples...")
    param_by_model: dict[str, dict[str, int]] = {}
    for p in args.parametric_examples:
        model, d = load_matched_examples(p)
        if model in param_by_model:
            param_by_model[model].update(d)
        else:
            param_by_model[model] = d

    print("Loading natural matched examples...")
    nat_by_model: dict[str, dict[str, int]] = {}
    for p in args.natural_examples:
        model, d = load_matched_examples(p)
        if model in nat_by_model:
            nat_by_model[model].update(d)
        else:
            nat_by_model[model] = d

    common_models = sorted(set(param_by_model) & set(nat_by_model))
    if not common_models:
        raise SystemExit("No models found in both parametric and natural examples.")

    print(f"Common models: {common_models}")

    all_stats: dict[str, dict] = {}
    model_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_stats_rows = []

    for model in common_models:
        param_arr, nat_arr, _ = build_paired_arrays(param_by_model[model], nat_by_model[model])
        if len(param_arr) < 2:
            print(f"  [{model}] not enough paired examples ({len(param_arr)}), skipping")
            continue
        stats_ = run_tests(param_arr, nat_arr)
        all_stats[model] = stats_
        model_arrays[model] = (param_arr, nat_arr)
        print(
            f"  [{display_name(model)}] n={stats_['n']}  "
            f"formal_mean={stats_['formal_mean']:.2f} ±{stats_['formal_ci95']:.2f}  "
            f"natural_mean={stats_['natural_mean']:.2f} ±{stats_['natural_ci95']:.2f}  "
            f"Wilcoxon p={stats_['wilcox_p']:.4f}  t-test p={stats_['ttest_p']:.4f}"
        )
        all_stats_rows.append({"model": model, "display_name": display_name(model), **stats_})

    pd.DataFrame(all_stats_rows).to_csv(
        out_dir / "musique_search_comparison_stats.csv", index=False
    )
    print(f"Saved {out_dir / 'musique_search_comparison_stats.csv'}")

    n_models = len(model_arrays)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12, 4 * max(n_models, 1) + 2))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6,
                              height_ratios=[n_models * 4, 4])

    violin_grid = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=outer[0], wspace=0.4)
    for col, (model, (param_arr, nat_arr)) in enumerate(model_arrays.items()):
        ax = fig.add_subplot(violin_grid[col])
        plot_violin(ax, param_arr, nat_arr, all_stats[model], model)

    ax_summary = fig.add_subplot(outer[1])
    plot_summary(ax_summary, all_stats)

    fig.suptitle("Search calls: MusiQue (Formal) vs MusiQue-Natural (Paraphrase)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Quadrant shift (optional, requires summary CSVs)
    if args.parametric_summary and args.natural_summary:
        print("\nBuilding quadrant shift plot...")
        param_qdf = load_summary_for_quadrants(args.parametric_summary, "musique")
        nat_qdf = load_summary_for_quadrants(args.natural_summary, "musique-natural")
        plot_quadrant_shift(
            param_qdf, nat_qdf,
            out_dir / "musique_quadrant_shift.png",
            out_dir / "musique_quadrant_shift_stats.csv",
        )
    else:
        print("Skipping quadrant shift plot (pass --parametric-summary and --natural-summary to enable)")


if __name__ == "__main__":
    main()
