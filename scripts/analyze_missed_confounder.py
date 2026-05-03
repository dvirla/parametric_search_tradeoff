"""
Decomposes the M-class (Missed = uncertain + not searched) into:
  - genuine_missed: hop_index==0, or all prior hops correct in all runs
  - cascade_possible: hop_index>0 and at least one prior hop had parametric_accuracy < 1.0

Three analyses:
  A. Cascade fraction bar chart (per dataset x model)
  B. P(M | hop_index=k) distribution
  C. Quadrant taxonomy restricted to aggregate_correct==True examples

Usage:
    uv run python scripts/analyze_missed_confounder.py \\
      --parametric-summary results/musique_parametric/interplay_analysis/interplay_summary.csv \\
      --natural-summary results/musique-natural/interplay_analysis/interplay_summary.csv \\
      --output-dir results/consolidated_analysis/missed_confounder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

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

CASCADE_COLORS = {
    "genuine_missed":   "#E57373",  # light red
    "cascade_possible": "#B71C1C",  # dark red
}


def display_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def load_summary(path: str, dataset_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = df["model"].str.replace(":", "_", regex=False)
    df = df.rename(columns={
        "semantic_entropy": "entropy",
        "searched": "search_attributed",
    })
    df["search_attributed"] = df["search_attributed"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)
    df["certain"] = (df["entropy"] == 0.0) & (df["parametric_certain"].astype(str) == "True")
    df["dataset"] = dataset_label

    conditions = [
        (~df["certain"]) & df["search_attributed"],
        (~df["certain"]) & (~df["search_attributed"]),
        df["certain"] & df["search_attributed"],
        df["certain"] & (~df["search_attributed"]),
    ]
    df["quadrant"] = np.select(conditions, ["E", "M", "PR", "CP"], default="?")
    return df


def label_cascade(df: pd.DataFrame) -> pd.DataFrame:
    """For M-class rows, add cascade_label column: genuine_missed or cascade_possible."""
    m_rows = df[df["quadrant"] == "M"].copy()

    # For each (dataset, model, example_id) group, build a lookup of max parametric_accuracy
    # for each hop_index, so we can check prior hops efficiently.
    # We work on the full df to get prior-hop accuracy even for non-M rows.
    acc_lookup = (
        df.groupby(["dataset", "model", "example_id", "hop_index"])["parametric_accuracy"]
        .first()
        .reset_index()
    )

    def _label_row(row):
        if row["hop_index"] == 0:
            return "genuine_missed"
        # get all hops with lower hop_index in same example
        prior = acc_lookup[
            (acc_lookup["dataset"] == row["dataset"]) &
            (acc_lookup["model"] == row["model"]) &
            (acc_lookup["example_id"] == row["example_id"]) &
            (acc_lookup["hop_index"] < row["hop_index"])
        ]
        if len(prior) == 0:
            return "genuine_missed"
        if (prior["parametric_accuracy"] < 1.0).any():
            return "cascade_possible"
        return "genuine_missed"

    m_rows["cascade_label"] = m_rows.apply(_label_row, axis=1)
    return m_rows


def plot_cascade_bar(m_rows: pd.DataFrame, out_path: Path) -> None:
    """Analysis A: stacked bar of genuine_missed vs cascade_possible per dataset x model."""
    summary = (
        m_rows.groupby(["dataset", "model", "cascade_label"])
        .size()
        .reset_index(name="count")
    )
    pivot = summary.pivot_table(
        index=["dataset", "model"], columns="cascade_label", values="count", fill_value=0
    ).reset_index()
    for col in ["genuine_missed", "cascade_possible"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["genuine_missed"] + pivot["cascade_possible"]
    pivot["frac_genuine"] = pivot["genuine_missed"] / pivot["total"].replace(0, np.nan)
    pivot["frac_cascade"] = pivot["cascade_possible"] / pivot["total"].replace(0, np.nan)

    datasets = sorted(pivot["dataset"].unique())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets + 1, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = pivot[pivot["dataset"] == dataset].copy()
        sub = sub.sort_values("model")
        x = np.arange(len(sub))
        labels = [display_name(m) for m in sub["model"]]

        ax.bar(x, sub["frac_genuine"], color=CASCADE_COLORS["genuine_missed"],
               label="Genuine missed")
        ax.bar(x, sub["frac_cascade"], bottom=sub["frac_genuine"],
               color=CASCADE_COLORS["cascade_possible"], label="Cascade possible")

        for i, (_, row) in enumerate(sub.iterrows()):
            pct_c = row["frac_cascade"] * 100 if not np.isnan(row["frac_cascade"]) else 0
            if pct_c >= 5:
                ax.text(x[i], row["frac_genuine"] + row["frac_cascade"] / 2,
                        f"{pct_c:.0f}%", ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            pct_g = row["frac_genuine"] * 100 if not np.isnan(row["frac_genuine"]) else 0
            if pct_g >= 5:
                ax.text(x[i], row["frac_genuine"] / 2,
                        f"{pct_g:.0f}%", ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")

        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(x[i], 1.01, f"n={int(row['total'])}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        title_map = {"musique": "MusiQue (Formal)", "musique-natural": "MusiQue (Natural)"}
        ax.set_title(title_map.get(dataset, dataset), fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Fraction of M-class hops")

    handles = [
        mpatches.Patch(color=CASCADE_COLORS["genuine_missed"], label="Genuine missed\n(all prior hops correct)"),
        mpatches.Patch(color=CASCADE_COLORS["cascade_possible"], label="Cascade possible\n(≥1 prior hop incorrect)"),
    ]
    axes[-1].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("M-class decomposition: genuine vs. cascade-confounded", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_hop_index_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Analysis B: P(M | hop_index=k) per dataset and model."""
    df = df[df["hop_index"].notna()].copy()
    df["hop_index"] = df["hop_index"].astype(int)

    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    max_hop = int(df["hop_index"].max())
    hop_indices = list(range(max_hop + 1))

    title_map = {"musique": "MusiQue (Formal)", "musique-natural": "MusiQue (Natural)"}
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        x = np.arange(len(hop_indices))
        w = 0.8 / max(n_models, 1)

        for mi, model in enumerate(models):
            ms = sub[sub["model"] == model]
            probs = []
            for k in hop_indices:
                total = (ms["hop_index"] == k).sum()
                missed = ((ms["hop_index"] == k) & (ms["quadrant"] == "M")).sum()
                probs.append(missed / total if total > 0 else 0.0)

            offset = (mi - n_models / 2 + 0.5) * w
            ax.bar(x + offset, probs, w, label=display_name(model),
                   color=colors[mi % len(colors)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Hop {k}" for k in hop_indices])
        ax.set_xlabel("Hop index")
        ax.set_ylabel("P(M | hop_index = k)")
        ax.set_title(title_map.get(dataset, dataset), fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("M-class rate by hop position (cascade diagnostic)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_correct_only_taxonomy(df: pd.DataFrame, out_path: Path) -> None:
    """Analysis C: E/CP/PR/M taxonomy restricted to aggregate_correct==True examples."""
    df_correct = df[df["aggregate_correct"].astype(str) == "True"].copy()

    datasets = sorted(df_correct["dataset"].unique())
    models = sorted(df_correct["model"].unique())

    # N correct examples per (dataset, model)
    n_correct = (
        df_correct.groupby(["dataset", "model"])["example_id"]
        .nunique()
        .reset_index(name="n_correct_examples")
    )

    combos = []
    for dataset in datasets:
        for model in models:
            sub = df_correct[(df_correct["dataset"] == dataset) & (df_correct["model"] == model)]
            if len(sub) == 0:
                continue
            n = len(sub)
            fracs = {q: (sub["quadrant"] == q).sum() / n for q in ["E", "CP", "PR", "M"]}
            nc_row = n_correct[(n_correct["dataset"] == dataset) & (n_correct["model"] == model)]
            nc = int(nc_row["n_correct_examples"].values[0]) if len(nc_row) > 0 else 0
            combos.append({"dataset": dataset, "model": model, "n_hops": n, "n_correct": nc, **fracs})

    if not combos:
        print("No correct examples found; skipping Analysis C")
        return

    combo_df = pd.DataFrame(combos)
    x = np.arange(len(combo_df))

    title_map = {"musique": "MusiQue (Formal)", "musique-natural": "MusiQue (Natural)"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets) + 1, 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = combo_df[combo_df["dataset"] == dataset].reset_index(drop=True)
        x = np.arange(len(sub))
        labels = [display_name(m) for m in sub["model"]]

        bottoms = np.zeros(len(sub))
        for q in ["E", "CP", "PR", "M"]:
            vals = sub[q].values
            ax.bar(x, vals, bottom=bottoms, color=QUADRANT_COLORS[q], label=q)
            for i, (v, bot) in enumerate(zip(vals, bottoms)):
                if v >= 0.05:
                    ax.text(x[i], bot + v / 2, f"{v*100:.0f}%",
                            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            bottoms += vals

        for i, row in sub.iterrows():
            ax.text(i, 1.02, f"N={row['n_correct']}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Fraction of hops")
        ax.set_title(f"{title_map.get(dataset, dataset)}\n(correct examples only)", fontweight="bold")

    handles = [mpatches.Patch(color=QUADRANT_COLORS[q], label=q) for q in ["E", "CP", "PR", "M"]]
    fig.legend(handles=handles, title="Quadrant", fontsize=9,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Quadrant taxonomy — restricted to correctly-answered examples", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parametric-summary",
                        default="results/musique_parametric/interplay_analysis/interplay_summary.csv")
    parser.add_argument("--natural-summary",
                        default="results/musique-natural/interplay_analysis/interplay_summary.csv")
    parser.add_argument("--output-dir", default="results/consolidated_analysis/missed_confounder")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading summaries...")
    df_param = load_summary(args.parametric_summary, "musique")
    df_nat = load_summary(args.natural_summary, "musique-natural")
    df = pd.concat([df_param, df_nat], ignore_index=True)

    print(f"Total hops: {len(df)}  M-class: {(df['quadrant']=='M').sum()}")

    # Analysis A
    print("\nAnalysis A: cascade check...")
    m_rows = label_cascade(df)
    plot_cascade_bar(m_rows, out_dir / "fig_missed_cascade_bar.png")

    summary = (
        m_rows.groupby(["dataset", "model", "cascade_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary.to_csv(out_dir / "missed_confounder_summary.csv", index=False)
    print(f"Saved {out_dir / 'missed_confounder_summary.csv'}")

    # Sanity check: hop_index==0 rows should all be genuine_missed
    hop0_labels = m_rows[m_rows["hop_index"] == 0]["cascade_label"].unique()
    assert set(hop0_labels) <= {"genuine_missed"}, f"FAIL: hop_index=0 rows have unexpected labels: {hop0_labels}"
    total_m = len(m_rows)
    split_total = (m_rows["cascade_label"] == "genuine_missed").sum() + (m_rows["cascade_label"] == "cascade_possible").sum()
    assert total_m == split_total, "FAIL: cascade split is not exhaustive"
    print(f"Sanity check passed: {total_m} M-class rows = {split_total} labeled rows")

    # Analysis B
    print("\nAnalysis B: hop-index distribution...")
    plot_hop_index_distribution(df, out_dir / "fig_missed_by_hop_index.png")

    # Analysis C
    print("\nAnalysis C: correct-only taxonomy...")
    plot_correct_only_taxonomy(df, out_dir / "fig_missed_correct_only_taxonomy.png")

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
