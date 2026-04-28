"""
Compare search call counts between curated-sharechat (original) and
curated-sharechat-benchmark (paraphrased) result JSONs, paired via the
benchmark CSV. Auto-discovers model pairs from the results directory.

Usage:
    uv run python scripts/compare_sharechat_search_calls.py
    uv run python scripts/compare_sharechat_search_calls.py --results-dir results --pairs data/curated_sharechat_wildchat_benchmark.csv
    uv run python scripts/compare_sharechat_search_calls.py --models "gemini-3-pro-preview" "nemotron-3-nano:30b"
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats


ORIG_PREFIX = "curated-sharechat_baseline_"
BENCH_PREFIX = "curated-sharechat-benchmark_baseline_"
RUN_SUFFIX_RE = re.compile(r"_run_\d+\.json$")

DISPLAY_NAMES = {
    "gemini-3-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}

COLORS = ["#4C72B0", "#DD8452"]


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def discover_model_pairs(results_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Return {model_name: (orig_path, bench_path)} for all matched pairs."""
    orig_files = {
        RUN_SUFFIX_RE.sub("", f.name[len(ORIG_PREFIX):]): f
        for f in results_dir.glob(f"{ORIG_PREFIX}*.json")
    }
    bench_files = {
        RUN_SUFFIX_RE.sub("", f.name[len(BENCH_PREFIX):]): f
        for f in results_dir.glob(f"{BENCH_PREFIX}*.json")
    }
    common = sorted(set(orig_files) & set(bench_files))
    return {m: (orig_files[m], bench_files[m]) for m in common}


def load_search_calls(path: Path) -> dict[str, int]:
    with open(path) as f:
        data = json.load(f)
    return {item["problem"]: item["sampler_search_calls"] for item in data}


def build_paired_arrays(
    pairs_df: pd.DataFrame, orig_map: dict, bench_map: dict
) -> tuple[np.ndarray, np.ndarray, int]:
    orig_calls, bench_calls = [], []
    missing = 0
    for _, row in pairs_df.iterrows():
        o, b = row["text"], row["benchmark_question"]
        if o in orig_map and b in bench_map:
            orig_calls.append(orig_map[o])
            bench_calls.append(bench_map[b])
        else:
            missing += 1
    return np.array(orig_calls), np.array(bench_calls), missing


def mean_ci95(arr: np.ndarray) -> tuple[float, float]:
    """95% CI half-width for the mean using t-distribution."""
    n = len(arr)
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return arr.mean(), t_crit * sem


def run_tests(orig: np.ndarray, bench: np.ndarray) -> dict:
    wilcox = stats.wilcoxon(orig, bench, alternative="two-sided")
    ttest = stats.ttest_rel(orig, bench)
    orig_mean, orig_ci = mean_ci95(orig)
    bench_mean, bench_ci = mean_ci95(bench)
    return {
        "n": len(orig),
        "orig_mean": orig_mean,
        "orig_ci95": orig_ci,
        "orig_median": float(np.median(orig)),
        "bench_mean": bench_mean,
        "bench_ci95": bench_ci,
        "bench_median": float(np.median(bench)),
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


def plot_violin(ax, orig: np.ndarray, bench: np.ndarray, stats_: dict, model: str):
    vp = ax.violinplot([orig, bench], positions=[0, 1], showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.7)

    for x, arr, color in zip([0, 1], [orig, bench], COLORS):
        ax.scatter([x], [arr.mean()], marker="D", color=color, zorder=5, s=45,
                   label=f"mean={arr.mean():.2f}")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original\n(sharechat)", "Benchmark\n(paraphrase)"])
    ax.set_ylabel("Search calls")
    ax.set_title(display_name(model), fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

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

    orig_means = np.array([all_stats[m]["orig_mean"] for m in models])
    bench_means = np.array([all_stats[m]["bench_mean"] for m in models])
    orig_cis = np.array([all_stats[m]["orig_ci95"] for m in models])
    bench_cis = np.array([all_stats[m]["bench_ci95"] for m in models])

    ax.bar(x - w / 2, orig_means, w, yerr=orig_cis, capsize=5,
           color=COLORS[0], alpha=0.8, label="Original (sharechat)", error_kw={"linewidth": 1.5})
    ax.bar(x + w / 2, bench_means, w, yerr=bench_cis, capsize=5,
           color=COLORS[1], alpha=0.8, label="Benchmark (paraphrase)", error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean search calls (95% CI)")
    ax.set_title("Mean search calls across models", fontweight="bold")
    ax.legend()

    for i, m in enumerate(models):
        p = all_stats[m]["wilcox_p"]
        y = bench_means[i] + bench_cis[i] + 0.3
        ax.text(x[i] + w / 2, y, pval_stars(p), ha="center", va="bottom", fontsize=11)

    ax.text(0.01, 0.99, "* p<0.05  ** p<0.01  *** p<0.001  ns: not sig.\n(Wilcoxon signed-rank, error bars = 95% CI)",
            transform=ax.transAxes, va="top", fontsize=7, color="gray")


def main():
    parser = argparse.ArgumentParser(description="Compare search calls: original vs benchmark paraphrases")
    parser.add_argument("--results-dir", default="results", help="Directory containing result JSONs")
    parser.add_argument("--pairs", default="data/curated_sharechat_wildchat_benchmark.csv",
                        help="CSV with 'text' and 'benchmark_question' columns")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Restrict to specific model names (auto-discover if omitted)")
    parser.add_argument("--output", default="results/sharechat_search_comparison.png",
                        help="Output plot path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pairs_df = pd.read_csv(args.pairs)

    pairs = discover_model_pairs(results_dir)
    if args.models:
        pairs = {m: v for m, v in pairs.items() if m in args.models}
    if not pairs:
        raise SystemExit("No matched model pairs found.")

    print(f"Found {len(pairs)} model pair(s): {list(pairs)}")

    all_stats: dict[str, dict] = {}
    model_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model, (orig_path, bench_path) in pairs.items():
        orig_map = load_search_calls(orig_path)
        bench_map = load_search_calls(bench_path)
        orig_arr, bench_arr, missing = build_paired_arrays(pairs_df, orig_map, bench_map)
        if missing:
            print(f"  [{model}] {missing} pairs could not be matched")
        stats_ = run_tests(orig_arr, bench_arr)
        all_stats[model] = stats_
        model_arrays[model] = (orig_arr, bench_arr)
        print(
            f"  [{display_name(model)}] n={stats_['n']}  "
            f"orig_mean={stats_['orig_mean']:.2f} ±{stats_['orig_ci95']:.2f}  "
            f"bench_mean={stats_['bench_mean']:.2f} ±{stats_['bench_ci95']:.2f}  "
            f"Wilcoxon p={stats_['wilcox_p']:.4f}  t-test p={stats_['ttest_p']:.4f}"
        )

    n_models = len(pairs)
    # 2 columns: violin plots on left, summary bar chart spanning right
    fig = plt.figure(figsize=(12, 4 * max(n_models, 1) + 2))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.6,
                              height_ratios=[n_models * 4, 4])

    violin_grid = gridspec.GridSpecFromSubplotSpec(1, n_models, subplot_spec=outer[0], wspace=0.4)
    for col, (model, (orig_arr, bench_arr)) in enumerate(model_arrays.items()):
        ax = fig.add_subplot(violin_grid[col])
        plot_violin(ax, orig_arr, bench_arr, all_stats[model], model)

    ax_summary = fig.add_subplot(outer[1])
    plot_summary(ax_summary, all_stats)

    fig.suptitle("Search calls: curated-sharechat vs benchmark paraphrases", fontsize=13, fontweight="bold")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
