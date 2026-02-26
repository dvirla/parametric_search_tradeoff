"""
MuSiQue Multi-Hop QA Analysis

Reads experiment JSONs, produces plots, CSV summary, and markdown report.
"""

import os
import sys
import json
import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")


def setup_args():
    parser = argparse.ArgumentParser(description="Analyze MuSiQue experiment results.")
    parser.add_argument("--results_dir", type=str, default="results/musique", help="Directory with musique_*.json files.")
    parser.add_argument("--output_dir", type=str, default="results/musique/analysis", help="Output directory for plots and report.")
    return parser.parse_args()


def load_results(results_dir: str) -> pd.DataFrame:
    """Auto-discover musique_*.json files and flatten into a DataFrame."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "musique_*.json")))
    # Exclude files inside analysis/ subdirectory
    json_files = [f for f in json_files if "/analysis/" not in f]
    print(f"Found {len(json_files)} result files in {results_dir}")

    rows = []
    for path in json_files:
        with open(path, "r") as f:
            data = json.load(f)
        for entry in data:
            model = entry["model_name"]
            mode = entry["mode"]
            example_id = entry["example_id"]

            # Sub-question rows
            for sr in entry["sub_questions_results"]:
                rows.append({
                    "example_id": example_id,
                    "model": model,
                    "mode": mode,
                    "question_type": "factoid",
                    "hop_index": sr["hop_index"],
                    "question": sr["question"],
                    "gold_answer": sr["gold_answer"],
                    "model_response": sr["model_response"],
                    "is_correct": sr["is_correct"],
                    "search_calls": sr["search_calls"],
                })

            # Aggregate row
            agg = entry["aggregate_result"]
            rows.append({
                "example_id": example_id,
                "model": model,
                "mode": mode,
                "question_type": "aggregate",
                "hop_index": -1,
                "question": agg["question"],
                "gold_answer": agg["gold_answer"],
                "model_response": agg["model_response"],
                "is_correct": agg["is_correct"],
                "search_calls": agg["search_calls"],
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows ({df['model'].nunique()} models, {df['mode'].nunique()} modes)")
    return df


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: x=model, groups=factoid/aggregate, hue=with_search/no_search."""
    acc = df.groupby(["model", "mode", "question_type"])["is_correct"].mean().reset_index()
    acc.rename(columns={"is_correct": "accuracy"}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(acc["model"].unique())
    x = np.arange(len(models))
    width = 0.18

    combos = [
        ("factoid", "no_search", "#4c72b0", "Factoid / No Search"),
        ("factoid", "with_search", "#55a868", "Factoid / With Search"),
        ("aggregate", "no_search", "#c44e52", "Aggregate / No Search"),
        ("aggregate", "with_search", "#8172b2", "Aggregate / With Search"),
    ]

    for i, (qt, mode, color, label) in enumerate(combos):
        vals = []
        for m in models:
            subset = acc[(acc["model"] == m) & (acc["mode"] == mode) & (acc["question_type"] == qt)]
            vals.append(subset["accuracy"].values[0] if len(subset) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=label, color=color)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("MuSiQue: Accuracy Comparison")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  -> musique_accuracy_comparison.png")


def plot_composition_gap(df: pd.DataFrame, output_dir: str):
    """P(aggregate correct | all factoids correct) vs P(aggregate correct | some factoids wrong)."""
    # Build per-example summary
    examples = []
    for (eid, model, mode), grp in df.groupby(["example_id", "model", "mode"]):
        factoids = grp[grp["question_type"] == "factoid"]
        agg = grp[grp["question_type"] == "aggregate"]
        if len(agg) == 0:
            continue
        all_f_correct = factoids["is_correct"].all()
        agg_correct = agg["is_correct"].values[0]
        examples.append({
            "model": model, "mode": mode,
            "all_factoids_correct": all_f_correct,
            "aggregate_correct": agg_correct,
        })

    edf = pd.DataFrame(examples)
    if edf.empty:
        print("  -> Skipping composition gap (no data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(edf["model"].unique())
    modes = sorted(edf["mode"].unique())
    x = np.arange(len(models))
    width = 0.18
    bar_idx = 0

    colors = {"all_correct": "#55a868", "some_wrong": "#c44e52"}
    for mode in modes:
        for condition, color_key in [("all_correct", "all_correct"), ("some_wrong", "some_wrong")]:
            vals = []
            for m in models:
                subset = edf[(edf["model"] == m) & (edf["mode"] == mode)]
                if condition == "all_correct":
                    s = subset[subset["all_factoids_correct"] == True]
                else:
                    s = subset[subset["all_factoids_correct"] == False]
                vals.append(s["aggregate_correct"].mean() if len(s) > 0 else 0)
            label = f"{mode} / {'All Factoids Correct' if condition == 'all_correct' else 'Some Factoids Wrong'}"
            ax.bar(x + bar_idx * width, vals, width, label=label, color=colors[color_key], alpha=0.8 if "no_search" in mode else 1.0, edgecolor="black" if "with_search" in mode else "none", linewidth=1)
            bar_idx += 1

    ax.set_xlabel("Model")
    ax.set_ylabel("P(Aggregate Correct)")
    ax.set_title("MuSiQue: Composition Gap")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_composition_gap.png"), dpi=150)
    plt.close()
    print("  -> musique_composition_gap.png")


def plot_search_delta(df: pd.DataFrame, output_dir: str):
    """accuracy_with_search - accuracy_no_search, per model, for factoids and aggregate."""
    acc = df.groupby(["model", "mode", "question_type"])["is_correct"].mean().reset_index()
    acc.rename(columns={"is_correct": "accuracy"}, inplace=True)

    models = sorted(acc["model"].unique())
    rows = []
    for m in models:
        for qt in ["factoid", "aggregate"]:
            ws = acc[(acc["model"] == m) & (acc["mode"] == "with_search") & (acc["question_type"] == qt)]
            ns = acc[(acc["model"] == m) & (acc["mode"] == "no_search") & (acc["question_type"] == qt)]
            ws_val = ws["accuracy"].values[0] if len(ws) > 0 else 0
            ns_val = ns["accuracy"].values[0] if len(ns) > 0 else 0
            rows.append({"model": m, "question_type": qt, "delta": ws_val - ns_val})

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        print("  -> Skipping search delta (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    for i, qt in enumerate(["factoid", "aggregate"]):
        vals = [delta_df[(delta_df["model"] == m) & (delta_df["question_type"] == qt)]["delta"].values[0] for m in models]
        color = "#55a868" if qt == "factoid" else "#8172b2"
        ax.bar(x + i * width, vals, width, label=qt.capitalize(), color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Model")
    ax.set_ylabel("Search Delta (with - without)")
    ax.set_title("MuSiQue: Search Delta")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_search_delta.png"), dpi=150)
    plt.close()
    print("  -> musique_search_delta.png")


def plot_per_hop_accuracy(df: pd.DataFrame, output_dir: str):
    """Line chart: accuracy by hop index (0-3) across models and modes."""
    factoids = df[df["question_type"] == "factoid"].copy()
    acc = factoids.groupby(["model", "mode", "hop_index"])["is_correct"].mean().reset_index()
    acc.rename(columns={"is_correct": "accuracy"}, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted(acc["model"].unique())
    modes = sorted(acc["mode"].unique())
    markers = {"no_search": "o", "with_search": "s"}
    linestyles = {"no_search": "--", "with_search": "-"}
    cmap = plt.cm.Set1

    for mi, m in enumerate(models):
        for mode in modes:
            subset = acc[(acc["model"] == m) & (acc["mode"] == mode)].sort_values("hop_index")
            if subset.empty:
                continue
            ax.plot(
                subset["hop_index"], subset["accuracy"],
                marker=markers.get(mode, "o"), linestyle=linestyles.get(mode, "-"),
                color=cmap(mi), label=f"{m} / {mode}", linewidth=1.5, markersize=6,
            )

    ax.set_xlabel("Hop Index")
    ax.set_ylabel("Accuracy")
    ax.set_title("MuSiQue: Per-Hop Accuracy")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_per_hop_accuracy.png"), dpi=150)
    plt.close()
    print("  -> musique_per_hop_accuracy.png")


def plot_example_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: rows=examples, columns=model x mode x question_type, cells=correct/incorrect."""
    # Build a pivot-friendly column
    df2 = df.copy()
    df2["col"] = df2.apply(
        lambda r: f"{r['model']}|{r['mode']}|{r['question_type']}" + (f"_h{r['hop_index']}" if r['question_type'] == 'factoid' else ""),
        axis=1,
    )
    pivot = df2.pivot_table(index="example_id", columns="col", values="is_correct", aggfunc="first")
    pivot = pivot.fillna(-1)  # -1 for missing

    # Sort columns for readability
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.5), max(6, len(pivot) * 0.4)))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#c44e52", "#55a868"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=5)
    ax.set_title("MuSiQue: Per-Example Heatmap (green=correct, red=incorrect, gray=missing)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_example_heatmap.png"), dpi=150)
    plt.close()
    print("  -> musique_example_heatmap.png")


def generate_summary_csv(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Generate summary CSV with accuracy stats."""
    summary = df.groupby(["model", "mode", "question_type"]).agg(
        accuracy=("is_correct", "mean"),
        count=("is_correct", "count"),
        correct=("is_correct", "sum"),
        avg_search_calls=("search_calls", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "musique_summary.csv"), index=False)
    print("  -> musique_summary.csv")
    return summary


def generate_report(df: pd.DataFrame, summary: pd.DataFrame, output_dir: str):
    """Generate markdown report with summary table and figure references."""
    lines = ["# MuSiQue Multi-Hop QA Analysis Report\n"]

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Model | Mode | Question Type | Accuracy | Count | Avg Search Calls |")
    lines.append("|-------|------|--------------|----------|-------|-----------------|")
    for _, row in summary.iterrows():
        lines.append(f"| {row['model']} | {row['mode']} | {row['question_type']} | {row['accuracy']:.3f} | {int(row['count'])} | {row['avg_search_calls']:.2f} |")

    # Overall stats
    lines.append("\n## Key Metrics\n")
    models = sorted(df["model"].unique())
    for m in models:
        lines.append(f"### {m}\n")
        for mode in ["no_search", "with_search"]:
            sub = df[(df["model"] == m) & (df["mode"] == mode)]
            if sub.empty:
                continue
            f_acc = sub[sub["question_type"] == "factoid"]["is_correct"].mean()
            a_acc = sub[sub["question_type"] == "aggregate"]["is_correct"].mean()
            lines.append(f"- **{mode}**: Factoid acc = {f_acc:.3f}, Aggregate acc = {a_acc:.3f}")
        lines.append("")

    # Figures
    lines.append("## Figures\n")
    for fig_name, desc in [
        ("musique_accuracy_comparison.png", "Accuracy Comparison"),
        ("musique_composition_gap.png", "Composition Gap"),
        ("musique_search_delta.png", "Search Delta"),
        ("musique_per_hop_accuracy.png", "Per-Hop Accuracy"),
        ("musique_example_heatmap.png", "Per-Example Heatmap"),
    ]:
        lines.append(f"### {desc}\n")
        lines.append(f"![{desc}]({fig_name})\n")

    report_path = os.path.join(output_dir, "musique_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> musique_report.md")


def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results(args.results_dir)
    if df.empty:
        print("No results found. Exiting.")
        return

    print("\nGenerating analyses...")
    summary = generate_summary_csv(df, args.output_dir)
    plot_accuracy_comparison(df, args.output_dir)
    plot_composition_gap(df, args.output_dir)
    plot_search_delta(df, args.output_dir)
    plot_per_hop_accuracy(df, args.output_dir)
    plot_example_heatmap(df, args.output_dir)
    generate_report(df, summary, args.output_dir)

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    main()
