"""
MuSiQue Multi-Hop QA Analysis

Reads experiment JSONs, produces plots, CSV summary, and markdown report.
Implements 7 rigorous analyses characterising the compositional gap under search augmentation.
"""

import os
import json
import argparse
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

matplotlib.use("Agg")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def setup_args():
    parser = argparse.ArgumentParser(description="Analyze MuSiQue experiment results.")
    parser.add_argument("--results_dir", type=str, default="results/musique",
                        help="Root directory containing musique_*.json files (searched recursively).")
    parser.add_argument("--output_dir", type=str, default="results/musique/analysis",
                        help="Output directory for plots and report.")
    parser.add_argument("--bad_ids", type=str, default="results/musique/bad_ids.csv",
                        help="CSV with columns id,reason for examples to exclude.")
    return parser.parse_args()


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    """Auto-discover musique_*.json files recursively and flatten into a DataFrame."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "**", "musique_*.json"), recursive=True))
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
                    "is_correct": bool(sr["is_correct"]),
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
                "is_correct": bool(agg["is_correct"]),
                "search_calls": agg["search_calls"],
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows ({df['model'].nunique()} models, {df['mode'].nunique()} modes)")
    return df


def filter_bad_ids(df: pd.DataFrame, bad_ids_path: str) -> pd.DataFrame:
    """Remove rows whose example_id appears in the bad_ids CSV."""
    if not os.path.exists(bad_ids_path):
        print(f"  [warn] bad_ids file not found: {bad_ids_path} — skipping filter")
        return df
    bad_df = pd.read_csv(bad_ids_path)
    bad_set = set(bad_df["id"].astype(str))
    before = len(df["example_id"].unique())
    mask = df["example_id"].astype(str).isin(bad_set)
    bad_example_ids = df.loc[mask, "example_id"].unique().tolist()
    df = df[~mask].copy()
    after = len(df["example_id"].unique())
    print(f"  Filtered {before - after} bad examples (IDs: {bad_example_ids}); {after} remain")
    return df


# ─── EXAMPLES DATAFRAME ───────────────────────────────────────────────────────

def build_examples_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (example_id, model, mode).  Summarises per-hop correctness and
    the aggregate outcome.  All downstream analyses consume this table.
    """
    records = []
    for (eid, model, mode), grp in df.groupby(["example_id", "model", "mode"]):
        factoids = grp[grp["question_type"] == "factoid"].sort_values("hop_index")
        agg_rows = grp[grp["question_type"] == "aggregate"]
        if agg_rows.empty:
            continue

        hop_profile = [bool(r) for r in factoids["is_correct"].tolist()]
        # Pad to 4 with False if data has fewer hops
        while len(hop_profile) < 4:
            hop_profile.append(False)

        all_hops_correct = all(hop_profile)
        agg_correct = bool(agg_rows["is_correct"].values[0])
        n_wrong_hops = sum(1 for h in hop_profile if not h)

        # 2×2 classification
        if all_hops_correct and agg_correct:
            cell = "A"
        elif not all_hops_correct and agg_correct:
            cell = "B"
        elif all_hops_correct and not agg_correct:
            cell = "C"
        else:
            cell = "D"

        records.append({
            "example_id": eid,
            "model": model,
            "mode": mode,
            "hop_0": hop_profile[0],
            "hop_1": hop_profile[1],
            "hop_2": hop_profile[2],
            "hop_3": hop_profile[3],
            "all_hops_correct": all_hops_correct,
            "agg_correct": agg_correct,
            "n_wrong_hops": n_wrong_hops,
            "cell": cell,
        })

    edf = pd.DataFrame(records)
    print(f"  Built examples_df: {len(edf)} rows")
    return edf


# ─── ANALYSIS 1: Basic Accuracy ───────────────────────────────────────────────

def report_basic_accuracy(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Per-(model, mode): per-hop accuracy (h=0..3), mean hop acc, aggregate acc.
    Prints markdown table, saves CSV.
    """
    print("\n=== Analysis 1: Basic Accuracy ===")
    factoids = df[df["question_type"] == "factoid"]
    agg_df = df[df["question_type"] == "aggregate"]

    rows = []
    for (model, mode), grp in factoids.groupby(["model", "mode"]):
        row = {"model": model, "mode": mode}
        hop_accs = []
        for h in range(4):
            h_grp = grp[grp["hop_index"] == h]
            acc = h_grp["is_correct"].mean() if len(h_grp) > 0 else float("nan")
            row[f"hop_{h}"] = acc
            hop_accs.append(acc)
        row["mean_hop_acc"] = np.nanmean(hop_accs)
        agg_sub = agg_df[(agg_df["model"] == model) & (agg_df["mode"] == mode)]
        row["aggregate_acc"] = agg_sub["is_correct"].mean() if len(agg_sub) > 0 else float("nan")
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(["model", "mode"])

    # Print markdown table
    header = "| Agent | Mode | Hop 0 | Hop 1 | Hop 2 | Hop 3 | Mean Hop | Aggregate |"
    sep    = "|-------|------|-------|-------|-------|-------|----------|-----------|"
    print(header)
    print(sep)
    for _, r in result.iterrows():
        print(f"| {r['model']} | {r['mode']} | {r['hop_0']:.3f} | {r['hop_1']:.3f} | "
              f"{r['hop_2']:.3f} | {r['hop_3']:.3f} | {r['mean_hop_acc']:.3f} | {r['aggregate_acc']:.3f} |")

    csv_path = os.path.join(output_dir, "musique_basic_accuracy.csv")
    result.to_csv(csv_path, index=False)
    print(f"  -> musique_basic_accuracy.csv")
    return result


# ─── ANALYSIS 2: Compositional Gap ────────────────────────────────────────────

def compute_compositional_gap(basic_acc: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    independence_ceiling = ∏ per_hop_acc(h)
    comp_gap = ceiling − aggregate_acc
    Bar chart grouped by model, two bars per group (modes).
    """
    print("\n=== Analysis 2: Compositional Gap ===")
    rows = []
    for _, r in basic_acc.iterrows():
        ceiling = r["hop_0"] * r["hop_1"] * r["hop_2"] * r["hop_3"]
        gap = ceiling - r["aggregate_acc"]
        rows.append({
            "model": r["model"],
            "mode": r["mode"],
            "independence_ceiling": ceiling,
            "aggregate_acc": r["aggregate_acc"],
            "comp_gap": gap,
        })
    gap_df = pd.DataFrame(rows)

    # Print table
    print(f"{'Model':<35} {'Mode':<12} {'Ceiling':>10} {'AggAcc':>10} {'Gap':>10}")
    for _, r in gap_df.iterrows():
        print(f"{r['model']:<35} {r['mode']:<12} {r['independence_ceiling']:>10.3f} {r['aggregate_acc']:>10.3f} {r['comp_gap']:>10.3f}")

    # Plot
    models = sorted(gap_df["model"].unique())
    modes = sorted(gap_df["mode"].unique())
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    mode_colors = {"no_search": "#4c72b0", "with_search": "#dd8452"}
    for i, mode in enumerate(modes):
        vals = []
        for m in models:
            row = gap_df[(gap_df["model"] == m) & (gap_df["mode"] == mode)]
            vals.append(row["comp_gap"].values[0] if len(row) > 0 else 0.0)
        bar_colors = [mode_colors.get(mode, "#999") if v >= 0 else "#c44e52" for v in vals]
        bars = ax.bar(x + i * width, vals, width, label=mode, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Model")
    ax.set_ylabel("Compositional Gap (Ceiling − Aggregate Acc)")
    ax.set_title("MuSiQue: Compositional Gap\n(positive = composition is lossy)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_compositional_gap.png"), dpi=150)
    plt.close()

    gap_df.to_csv(os.path.join(output_dir, "musique_compositional_gap.csv"), index=False)
    print("  -> musique_compositional_gap.png")
    print("  -> musique_compositional_gap.csv")
    return gap_df


# ─── ANALYSIS 3: Outcome Classification ───────────────────────────────────────

def classify_outcomes(examples_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Count cells A/B/C/D per (model, mode). Print 2×2 tables.
    Run Fisher's exact test on Cell B vs C for search vs no_search.
    """
    print("\n=== Analysis 3: Per-Question Outcome Classification ===")
    records = []
    for (model, mode), grp in examples_df.groupby(["model", "mode"]):
        n = len(grp)
        counts = grp["cell"].value_counts().to_dict()
        A = counts.get("A", 0)
        B = counts.get("B", 0)
        C = counts.get("C", 0)
        D = counts.get("D", 0)
        records.append({
            "model": model, "mode": mode,
            "n": n, "A": A, "B": B, "C": C, "D": D,
            "A_pct": A / n * 100 if n else 0,
            "B_pct": B / n * 100 if n else 0,
            "C_pct": C / n * 100 if n else 0,
            "D_pct": D / n * 100 if n else 0,
        })

    cls_df = pd.DataFrame(records)

    # Print per-model 2×2 tables
    for model in sorted(cls_df["model"].unique()):
        print(f"\n  Model: {model}")
        for _, r in cls_df[cls_df["model"] == model].iterrows():
            print(f"  Mode: {r['mode']}  (n={r['n']})")
            print(f"    {'':20} All hops correct   ≥1 hop wrong")
            print(f"    {'AGG correct':20} A={r['A']:3d} ({r['A_pct']:.1f}%)    B={r['B']:3d} ({r['B_pct']:.1f}%)")
            print(f"    {'AGG wrong':20} C={r['C']:3d} ({r['C_pct']:.1f}%)    D={r['D']:3d} ({r['D_pct']:.1f}%)")

    # Fisher's exact test: pooled across models — search vs no_search, Cell B vs C
    # Contingency: rows = mode, cols = [B, C]
    for mode_pair in [("no_search", "with_search")]:
        ns_rows = cls_df[cls_df["mode"] == mode_pair[0]]
        ws_rows = cls_df[cls_df["mode"] == mode_pair[1]]
        B_ns = ns_rows["B"].sum(); C_ns = ns_rows["C"].sum()
        B_ws = ws_rows["B"].sum(); C_ws = ws_rows["C"].sum()
        table = [[B_ns, C_ns], [B_ws, C_ws]]
        odds_ratio, pvalue = stats.fisher_exact(table)
        print(f"\n  Fisher's exact (B vs C, pooled): OR={odds_ratio:.3f}, p={pvalue:.4f}")
        print(f"    no_search:   B={B_ns}, C={C_ns}")
        print(f"    with_search: B={B_ws}, C={C_ws}")

    cls_df.to_csv(os.path.join(output_dir, "musique_outcome_classification.csv"), index=False)
    print("\n  -> musique_outcome_classification.csv")

    _plot_outcome_classification(cls_df, output_dir)
    return cls_df


def _plot_outcome_classification(cls_df: pd.DataFrame, output_dir: str):
    """
    Horizontal 100%-stacked bar chart: one bar per (model, mode), coloured by cell A/B/C/D.
    Cells are arranged within each bar so that the two "AGG correct" cells (A, B) sit on
    the left and the two "AGG wrong" cells (C, D) on the right, making the aggregate
    accuracy directly readable as the width of the coloured region.
    """
    # Cell colours and short labels
    CELL_META = {
        "A": ("#2ca02c", "A  Full success\n(all hops ✓, AGG ✓)"),
        "B": ("#98df8a", "B  Rescue\n(≥1 hop ✗, AGG ✓)"),
        "C": ("#ff7f0e", "C  Comp. failure\n(all hops ✓, AGG ✗)"),
        "D": ("#d62728", "D  Justified failure\n(≥1 hop ✗, AGG ✗)"),
    }
    CELL_ORDER = ["A", "B", "C", "D"]

    # Build one row per (model, mode) in a stable display order
    models = sorted(cls_df["model"].unique())
    rows = []
    for model in models:
        for mode in ["no_search", "with_search"]:
            row = cls_df[(cls_df["model"] == model) & (cls_df["mode"] == mode)]
            if row.empty:
                continue
            rows.append(row.iloc[0])
    plot_df = pd.DataFrame(rows).reset_index(drop=True)

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(11, 0.9 * n + 1.8))

    lefts = np.zeros(n)
    for cell in CELL_ORDER:
        pcts = plot_df[f"{cell}_pct"].values
        color, legend_label = CELL_META[cell]
        bars = ax.barh(np.arange(n), pcts, left=lefts, color=color,
                       label=legend_label, edgecolor="white", linewidth=0.6)
        # Label each segment with "count (pct%)" when wide enough
        for i, (bar, pct) in enumerate(zip(bars, pcts)):
            if pct >= 8:
                count = int(plot_df.iloc[i][cell])
                ax.text(
                    lefts[i] + pct / 2,
                    i,
                    f"{count}\n({pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=7.5, color="black", fontweight="bold",
                )
        lefts += pcts

    # Y-axis labels: "model  |  mode"
    ylabels = [
        f"{r['model']}\n({r['mode'].replace('_', ' ')})"
        for _, r in plot_df.iterrows()
    ]
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()

    # Vertical separator at 50% for visual reference
    ax.axvline(50, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of examples (%)", fontsize=10)
    ax.set_title(
        "MuSiQue: Outcome Classification per Model & Mode\n"
        "← AGG correct (A+B) ──── AGG wrong (C+D) →",
        fontsize=11,
    )

    # Legend outside to the right
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=8, title="Cell", title_fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()
    out = os.path.join(output_dir, "musique_outcome_classification.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> musique_outcome_classification.png")


# ─── ANALYSIS 4: Rescue Rate & Composition Loss Rate ──────────────────────────

def compute_rates_and_plot(examples_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    rescue_rate         = |B| / (|B| + |D|)
    composition_loss_rate = |C| / (|A| + |C|)
    Grouped paired bar chart per model.
    """
    print("\n=== Analysis 4: Rescue Rate & Composition Loss Rate ===")
    records = []
    for (model, mode), grp in examples_df.groupby(["model", "mode"]):
        counts = grp["cell"].value_counts().to_dict()
        A = counts.get("A", 0); B = counts.get("B", 0)
        C = counts.get("C", 0); D = counts.get("D", 0)
        rescue = B / (B + D) if (B + D) > 0 else float("nan")
        comp_loss = C / (A + C) if (A + C) > 0 else float("nan")
        records.append({
            "model": model, "mode": mode,
            "rescue_rate": rescue,
            "composition_loss_rate": comp_loss,
        })
        print(f"  {model} / {mode}: rescue={rescue:.3f}, comp_loss={comp_loss:.3f}")

    rates_df = pd.DataFrame(records)

    # Plot
    models = sorted(rates_df["model"].unique())
    modes = sorted(rates_df["mode"].unique())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = {
        ("no_search", "rescue_rate"):            "#4c72b0",
        ("no_search", "composition_loss_rate"):  "#55a868",
        ("with_search", "rescue_rate"):          "#dd8452",
        ("with_search", "composition_loss_rate"):"#c44e52",
    }
    offset = 0
    for mode in modes:
        for metric in ["rescue_rate", "composition_loss_rate"]:
            vals = []
            for m in models:
                row = rates_df[(rates_df["model"] == m) & (rates_df["mode"] == mode)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0.0)
            color = palette.get((mode, metric), "#aaa")
            label = f"{mode} / {metric.replace('_', ' ')}"
            ax.bar(x + offset * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
            offset += 1

    ax.set_xlabel("Model")
    ax.set_ylabel("Rate")
    ax.set_title("MuSiQue: Rescue Rate & Composition Loss Rate")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_rates.png"), dpi=150)
    plt.close()

    rates_df.to_csv(os.path.join(output_dir, "musique_rates.csv"), index=False)
    print("  -> musique_rates.png")
    print("  -> musique_rates.csv")
    return rates_df


# ─── ANALYSIS 5: Rescue Patterns ──────────────────────────────────────────────

def analyze_rescue_patterns(examples_df: pd.DataFrame, output_dir: str):
    """
    For Cell B (rescue) cases: which hop positions fail most often?
    Plot separately per mode, normalised to fraction.
    """
    print("\n=== Analysis 5: Rescue Patterns (Cell B) ===")
    rescue = examples_df[examples_df["cell"] == "B"].copy()
    if rescue.empty:
        print("  No Cell B cases found; skipping.")
        return

    models = sorted(rescue["model"].unique())
    modes = sorted(rescue["mode"].unique())
    mode_colors = {"no_search": "#4c72b0", "with_search": "#dd8452"}
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(4)
    width = 0.35

    for ax, model in zip(axes, models):
        for i, mode in enumerate(modes):
            sub = rescue[(rescue["model"] == model) & (rescue["mode"] == mode)]
            n = len(sub)
            freqs = []
            for h in range(4):
                wrong = (~sub[f"hop_{h}"]).sum() if n > 0 else 0
                freqs.append(wrong / n if n > 0 else 0.0)

            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, freqs, width,
                          label=f"{mode.replace('_', ' ')} (n={n})",
                          color=mode_colors.get(mode, "#aaa"),
                          alpha=0.85, edgecolor="black", linewidth=0.5)

            top3 = sorted(range(4), key=lambda h: freqs[h], reverse=True)[:3]
            print(f"  {model} / {mode}: top-3 failing hops: " +
                  ", ".join([f"hop {h}={freqs[h]:.2f}" for h in top3]))

        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlabel("Hop")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Hop {h}" for h in range(4)])
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of rescue cases\nwhere this hop was wrong", fontsize=9)

    fig.suptitle(
        "Rescue Patterns (Cell B): which hops failed when AGG was still correct?",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_rescue_patterns.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> musique_rescue_patterns.png")


# ─── ANALYSIS 6: Search Lift ──────────────────────────────────────────────────

def compute_search_lift(df: pd.DataFrame, examples_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Per-hop search lift: correct(search) − correct(no_search), paired by question.
    Aggregate compositional search effect vs. predicted from hop lifts.
    """
    print("\n=== Analysis 6: Per-Hop Search Lift ===")
    # Pair no_search / with_search for same example_id + model
    factoids = df[df["question_type"] == "factoid"].copy()
    agg_q   = df[df["question_type"] == "aggregate"].copy()

    # Build paired hop data
    lift_records = []
    for model in factoids["model"].unique():
        for h in range(4):
            ns = factoids[(factoids["model"] == model) & (factoids["mode"] == "no_search") & (factoids["hop_index"] == h)].set_index("example_id")["is_correct"]
            ws = factoids[(factoids["model"] == model) & (factoids["mode"] == "with_search") & (factoids["hop_index"] == h)].set_index("example_id")["is_correct"]
            common = ns.index.intersection(ws.index)
            for eid in common:
                lift_records.append({
                    "model": model, "hop_index": h, "example_id": eid,
                    "ns_correct": ns[eid], "ws_correct": ws[eid],
                    "lift": int(ws[eid]) - int(ns[eid]),
                    "ns_bucket": _bucket(ns[eid]),
                })

    lift_df = pd.DataFrame(lift_records)

    if lift_df.empty:
        print("  No paired data; skipping Search Lift analysis.")
        return pd.DataFrame()

    # Plot mean lift by hop position, coloured by model
    models = sorted(lift_df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.Set1
    for mi, m in enumerate(models):
        sub = lift_df[lift_df["model"] == m]
        mean_lift = sub.groupby("hop_index")["lift"].mean()
        ax.plot(mean_lift.index, mean_lift.values, marker="o", label=m, color=cmap(mi), linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hop Index")
    ax.set_ylabel("Mean Search Lift")
    ax.set_title("MuSiQue: Per-Hop Search Lift\n(with_search − no_search correctness, paired)")
    ax.set_xticks(range(4))
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_hop_search_lift.png"), dpi=150)
    plt.close()
    print("  -> musique_hop_search_lift.png")

    # Aggregate compositional search effect
    print("\n  Compositional search effect (actual vs. predicted):")
    comp_records = []
    for model in examples_df["model"].unique():
        ns_ex = examples_df[(examples_df["model"] == model) & (examples_df["mode"] == "no_search")]
        ws_ex = examples_df[(examples_df["model"] == model) & (examples_df["mode"] == "with_search")]
        actual_lift = ws_ex["agg_correct"].mean() - ns_ex["agg_correct"].mean()
        # predicted lift from independence model applied to per-hop lifts
        ns_hop_acc = [ns_ex[f"hop_{h}"].mean() for h in range(4)]
        ws_hop_acc = [ws_ex[f"hop_{h}"].mean() for h in range(4)]
        pred_ns = np.prod(ns_hop_acc)
        pred_ws = np.prod(ws_hop_acc)
        predicted_lift = pred_ws - pred_ns
        comp_effect = actual_lift - predicted_lift
        comp_records.append({
            "model": model,
            "actual_agg_lift": actual_lift,
            "predicted_agg_lift": predicted_lift,
            "compositional_search_effect": comp_effect,
        })
        print(f"  {model}: actual={actual_lift:+.3f}, predicted={predicted_lift:+.3f}, effect={comp_effect:+.3f}")

    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(os.path.join(output_dir, "musique_search_lift.csv"), index=False)
    print("  -> musique_search_lift.csv")
    return comp_df


def _bucket(val: bool) -> str:
    return "correct" if val else "wrong"


# ─── ANALYSIS 7: Formal Model Fitting ─────────────────────────────────────────

def fit_formal_models(examples_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Fit 3 nested models predicting P(AGG) from per-hop accuracies.
    Model 1: Independence  P(AGG) = ∏ P(hᵢ)
    Model 2: Penalty       P(AGG) = γ · ∏ P(hᵢ)
    Model 3: Penalty+Rescue P(AGG) = γ · ∏ P(hᵢ) + ρ · (1 − ∏ P(hᵢ))
    """
    print("\n=== Analysis 7: Formal Model Fitting ===")
    records = []

    for (model, mode), grp in examples_df.groupby(["model", "mode"]):
        grp = grp.reset_index(drop=True)
        n = len(grp)

        # Per-question independence ceiling
        ceilings = np.array([
            grp[f"hop_{h}"].astype(float).values for h in range(4)
        ]).prod(axis=0)  # shape (n,)

        y = grp["agg_correct"].astype(float).values  # 0/1

        # — Model 1: no params, just product of marginal accs —
        p1_hat = grp[[f"hop_{h}" for h in range(4)]].mean().prod()
        ll1 = _log_lik_fixed(y, p1_hat)
        aic1 = -2 * ll1 + 2 * 0

        # — Model 2: fit γ —
        ceiling_mean = ceilings.mean()

        def neg_ll_m2(params):
            gamma = np.clip(params[0], 1e-9, 1.0)
            p = np.clip(gamma * ceilings, 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res2 = minimize(neg_ll_m2, [0.5], bounds=[(0, 1)], method="L-BFGS-B")
        gamma_hat = float(np.clip(res2.x[0], 1e-9, 1.0))
        ll2 = -res2.fun
        aic2 = -2 * ll2 + 2 * 1

        # — Model 3: fit (γ, ρ) —
        def neg_ll_m3(params):
            gamma, rho = np.clip(params[0], 1e-9, 1.0), np.clip(params[1], 0.0, 1.0)
            p = np.clip(gamma * ceilings + rho * (1 - ceilings), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res3 = minimize(neg_ll_m3, [gamma_hat, 0.1], bounds=[(0, 1), (0, 1)], method="L-BFGS-B")
        gamma3, rho3 = float(np.clip(res3.x[0], 0, 1)), float(np.clip(res3.x[1], 0, 1))
        ll3 = -res3.fun
        aic3 = -2 * ll3 + 2 * 2

        # Bootstrap 1000 samples for CIs on γ (M2) and (γ, ρ) (M3)
        rng = np.random.default_rng(42)
        boot_gammas2 = []
        boot_gammas3 = []
        boot_rhos3 = []
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            y_b = y[idx]; c_b = ceilings[idx]

            def ng2(p):
                g = np.clip(p[0], 1e-9, 1.0)
                pp = np.clip(g * c_b, 1e-9, 1 - 1e-9)
                return np.sum(y_b * np.log(pp) + (1 - y_b) * np.log(1 - pp))

            res2b = minimize(lambda p: -ng2(p), [gamma_hat], bounds=[(0, 1)], method="L-BFGS-B")
            boot_gammas2.append(float(np.clip(res2b.x[0], 0, 1)))

            def ng3(p):
                g, r = np.clip(p[0], 1e-9, 1.0), np.clip(p[1], 0, 1)
                pp = np.clip(g * c_b + r * (1 - c_b), 1e-9, 1 - 1e-9)
                return np.sum(y_b * np.log(pp) + (1 - y_b) * np.log(1 - pp))

            res3b = minimize(lambda p: -ng3(p), [gamma3, rho3], bounds=[(0, 1), (0, 1)], method="L-BFGS-B")
            boot_gammas3.append(float(np.clip(res3b.x[0], 0, 1)))
            boot_rhos3.append(float(np.clip(res3b.x[1], 0, 1)))

        g2_lo, g2_hi = np.percentile(boot_gammas2, [2.5, 97.5])
        g3_lo, g3_hi = np.percentile(boot_gammas3, [2.5, 97.5])
        r3_lo, r3_hi = np.percentile(boot_rhos3, [2.5, 97.5])

        records.append({
            "model": model, "mode": mode,
            "n": n,
            "aic_m1": aic1, "aic_m2": aic2, "aic_m3": aic3,
            "best_model": ["M1", "M2", "M3"][np.argmin([aic1, aic2, aic3])],
            "gamma_m2": gamma_hat, "gamma_m2_lo": g2_lo, "gamma_m2_hi": g2_hi,
            "gamma_m3": gamma3, "gamma_m3_lo": g3_lo, "gamma_m3_hi": g3_hi,
            "rho_m3": rho3, "rho_m3_lo": r3_lo, "rho_m3_hi": r3_hi,
        })

        print(f"  {model} / {mode}: AIC M1={aic1:.1f} M2={aic2:.1f} M3={aic3:.1f} | "
              f"γ(M2)={gamma_hat:.3f} [{g2_lo:.3f},{g2_hi:.3f}] | "
              f"γ(M3)={gamma3:.3f} [{g3_lo:.3f},{g3_hi:.3f}], ρ={rho3:.3f} [{r3_lo:.3f},{r3_hi:.3f}]")

    fm_df = pd.DataFrame(records)
    fm_df.to_csv(os.path.join(output_dir, "musique_formal_models.csv"), index=False)

    # Plot γ_search vs γ_no_search (Model 2) per model with CI error bars
    models = sorted(fm_df["model"].unique())
    modes = sorted(fm_df["mode"].unique())
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    mode_colors = {"no_search": "#4c72b0", "with_search": "#dd8452"}

    for i, mode in enumerate(modes):
        vals, lo_errs, hi_errs = [], [], []
        for m in models:
            row = fm_df[(fm_df["model"] == m) & (fm_df["mode"] == mode)]
            if row.empty:
                vals.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            g = row["gamma_m2"].values[0]
            vals.append(g)
            lo_errs.append(g - row["gamma_m2_lo"].values[0])
            hi_errs.append(row["gamma_m2_hi"].values[0] - g)
        ax.bar(x + i * width, vals, width, label=mode,
               color=mode_colors.get(mode, "#aaa"), alpha=0.85, edgecolor="black", linewidth=0.5,
               yerr=[lo_errs, hi_errs], capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xlabel("Model")
    ax.set_ylabel("γ (compositional efficiency)")
    ax.set_title("MuSiQue: Compositional Efficiency γ (Model 2)\nwith 95% Bootstrap CI")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(title="Mode")
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "musique_formal_models.png"), dpi=150)
    plt.close()

    print("  -> musique_formal_models.png")
    print("  -> musique_formal_models.csv")
    return fm_df


def _log_lik_fixed(y: np.ndarray, p: float) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions (arcsine transformation)."""
    p1 = float(np.clip(p1, 0, 1))
    p2 = float(np.clip(p2, 0, 1))
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def _fit_gamma_m2(df_sub: pd.DataFrame) -> float:
    """Fit γ for Model 2: P(AGG) = γ · ∏P(hᵢ). Returns γ ∈ [0, 1]."""
    y = df_sub["agg_correct"].astype(float).values
    ceilings = np.array([df_sub[f"hop_{h}"].astype(float).values for h in range(4)]).prod(axis=0)

    def neg_ll(params):
        g = np.clip(params[0], 1e-9, 1.0)
        p = np.clip(g * ceilings, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(neg_ll, [0.5], bounds=[(0, 1)], method="L-BFGS-B")
    return float(np.clip(res.x[0], 0, 1))


# ─── STATISTICAL TESTS ────────────────────────────────────────────────────────

def run_statistical_tests(examples_df: pd.DataFrame, fm_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Per-model statistical comparisons (search vs. no_search):

      1. McNemar's test  — paired aggregate correctness on matched questions.
         Effect size: McNemar OR = (search helps) / (search hurts).

      2. Rescue rate (Cell B / (B+D))  — Fisher's exact + 95% bootstrap CI on OR +
         Cohen's h.

      3. Δγ (M2, search − no_search)  — bootstrap CI via joint resampling.
    """
    print("\n=== Statistical Tests (Effect Sizes & Paired Comparisons) ===")
    rng = np.random.default_rng(42)
    N_BOOT = 2000
    records = []

    for model in sorted(examples_df["model"].unique()):
        ns_df = (examples_df[(examples_df["model"] == model) & (examples_df["mode"] == "no_search")]
                 .set_index("example_id"))
        ws_df = (examples_df[(examples_df["model"] == model) & (examples_df["mode"] == "with_search")]
                 .set_index("example_id"))
        common = ns_df.index.intersection(ws_df.index)
        n_paired = len(common)

        # ── 1. McNemar's test ─────────────────────────────────────────────────
        n_help = 0; n_hurt = 0
        for eid in common:
            ns_ok = bool(ns_df.loc[eid, "agg_correct"])
            ws_ok = bool(ws_df.loc[eid, "agg_correct"])
            if not ns_ok and ws_ok:
                n_help += 1   # search turns wrong → right
            elif ns_ok and not ws_ok:
                n_hurt += 1   # search turns right → wrong

        disc = n_help + n_hurt
        if disc >= 1:
            # With continuity correction (Edwards)
            chi2_stat = (abs(n_help - n_hurt) - 1) ** 2 / disc
            mcnemar_p = float(stats.chi2.sf(chi2_stat, df=1))
        else:
            chi2_stat = 0.0; mcnemar_p = 1.0
        # OR: how many times more often does search help than hurt?
        mcnemar_or = n_help / n_hurt if n_hurt > 0 else (float("inf") if n_help > 0 else float("nan"))

        # ── 2. Rescue rate comparison ─────────────────────────────────────────
        def cell_counts(d):
            c = d["cell"].value_counts().to_dict()
            return c.get("B", 0), c.get("D", 0)

        B_ns, D_ns = cell_counts(ns_df)
        B_ws, D_ws = cell_counts(ws_df)

        # Fisher's exact (rows = mode, cols = rescue / not-rescue)
        or_fisher, p_fisher = stats.fisher_exact([[B_ws, D_ws], [B_ns, D_ns]])

        rr_ns = B_ns / (B_ns + D_ns) if (B_ns + D_ns) > 0 else 0.0
        rr_ws = B_ws / (B_ws + D_ws) if (B_ws + D_ws) > 0 else 0.0
        h_rescue = cohens_h(rr_ws, rr_ns)

        # Bootstrap CI on the rescue OR (Haldane-corrected)
        ns_imp = ns_df[ns_df["n_wrong_hops"] > 0]["agg_correct"].values.astype(float)
        ws_imp = ws_df[ws_df["n_wrong_hops"] > 0]["agg_correct"].values.astype(float)
        boot_ors = []
        for _ in range(N_BOOT):
            b_ns = rng.choice(ns_imp, size=max(len(ns_imp), 1), replace=True) if len(ns_imp) else np.zeros(1)
            b_ws = rng.choice(ws_imp, size=max(len(ws_imp), 1), replace=True) if len(ws_imp) else np.zeros(1)
            # Haldane correction to avoid 0/0
            p_n = (b_ns.sum() + 0.5) / (len(b_ns) + 1)
            p_w = (b_ws.sum() + 0.5) / (len(b_ws) + 1)
            boot_ors.append((p_w / (1 - p_w)) / (p_n / (1 - p_n)))
        or_lo, or_hi = np.percentile(boot_ors, [2.5, 97.5])

        # ── 3. Δγ (M2) with joint bootstrap CI ───────────────────────────────
        fm_ns = fm_df[(fm_df["model"] == model) & (fm_df["mode"] == "no_search")]
        fm_ws = fm_df[(fm_df["model"] == model) & (fm_df["mode"] == "with_search")]
        if not fm_ns.empty and not fm_ws.empty:
            gamma_ns = fm_ns["gamma_m2"].values[0]
            gamma_ws = fm_ws["gamma_m2"].values[0]
            gamma_diff = gamma_ws - gamma_ns
            ns_r = ns_df.reset_index()
            ws_r = ws_df.reset_index()
            boot_diffs = []
            for _ in range(N_BOOT):
                g_ns = _fit_gamma_m2(ns_r.iloc[rng.integers(0, len(ns_r), len(ns_r))].reset_index(drop=True))
                g_ws = _fit_gamma_m2(ws_r.iloc[rng.integers(0, len(ws_r), len(ws_r))].reset_index(drop=True))
                boot_diffs.append(g_ws - g_ns)
            diff_lo, diff_hi = np.percentile(boot_diffs, [2.5, 97.5])
        else:
            gamma_ns = gamma_ws = gamma_diff = diff_lo = diff_hi = float("nan")

        # ── print & record ────────────────────────────────────────────────────
        print(f"\n  {model}  (n_paired={n_paired})")
        print(f"    McNemar:  χ²={chi2_stat:.2f}, p={mcnemar_p:.4f}  "
              f"[helps={n_help}, hurts={n_hurt}]  OR={mcnemar_or:.2f}")
        print(f"    Rescue OR (search/no_search):  {or_fisher:.2f}  "
              f"95% CI [{or_lo:.2f}, {or_hi:.2f}]  p={p_fisher:.4f}  Cohen's h={h_rescue:.2f}")
        print(f"    Δγ (search − no_search):  {gamma_diff:+.3f}  "
              f"95% CI [{diff_lo:+.3f}, {diff_hi:+.3f}]")

        records.append({
            "model": model,
            "n_paired": n_paired,
            "mcnemar_n_help": n_help, "mcnemar_n_hurt": n_hurt,
            "mcnemar_chi2": chi2_stat, "mcnemar_p": mcnemar_p,
            "mcnemar_or": mcnemar_or,
            "rescue_rate_ns": rr_ns, "rescue_rate_ws": rr_ws,
            "rescue_or_fisher": or_fisher, "rescue_p_fisher": p_fisher,
            "rescue_or_lo": or_lo, "rescue_or_hi": or_hi,
            "cohens_h_rescue": h_rescue,
            "gamma_ns": gamma_ns, "gamma_ws": gamma_ws,
            "gamma_diff": gamma_diff,
            "gamma_diff_lo": diff_lo, "gamma_diff_hi": diff_hi,
        })

    tests_df = pd.DataFrame(records)
    tests_df.to_csv(os.path.join(output_dir, "musique_statistical_tests.csv"), index=False)
    print("\n  -> musique_statistical_tests.csv")

    _plot_effect_sizes(tests_df, output_dir)
    return tests_df


def _plot_effect_sizes(tests_df: pd.DataFrame, output_dir: str):
    """
    Three-panel forest plot summarising all effect sizes per model.
      Panel 1: McNemar OR on log₂ scale (search helps vs hurts, paired)
      Panel 2: Rescue rate OR with 95% bootstrap CI (log scale)
      Panel 3: Δγ (search − no_search) with 95% bootstrap CI
    """
    models = tests_df["model"].tolist()
    n = len(models)
    y = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 2.2 + 0.9 * n),
                             gridspec_kw={"wspace": 0.05})

    def _color(val, neutral=0):
        return "#2ca02c" if val >= neutral else "#d62728"

    # ── Panel 1: McNemar OR (log₂) ───────────────────────────────────────────
    ax = axes[0]
    for i, (_, row) in enumerate(tests_df.iterrows()):
        or_val = row["mcnemar_or"]
        if not np.isfinite(or_val):
            or_val = 8.0   # display cap for Inf
        log2_or = np.log2(or_val) if or_val > 0 else 0.0
        ax.barh(i, log2_or, color=_color(log2_or), alpha=0.75,
                edgecolor="black", linewidth=0.5, height=0.5)
        # annotate: p-value + discordant counts
        ax.text(log2_or + 0.05, i,
                f"p={row['mcnemar_p']:.3f}\n({row['mcnemar_n_help']}↑ {row['mcnemar_n_hurt']}↓)",
                va="center", ha="left", fontsize=7)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("log₂(OR)  [search helps / hurts]", fontsize=8)
    ax.set_title("McNemar's Test\n(paired agg. correctness)", fontsize=9, fontweight="bold")
    # Symmetric limits with headroom for labels
    xlim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), 1.5)
    ax.set_xlim(-xlim, xlim * 2.2)
    ax.invert_yaxis()

    # ── Panel 2: Rescue OR + CI (log scale) ──────────────────────────────────
    ax = axes[1]
    for i, (_, row) in enumerate(tests_df.iterrows()):
        or_val = row["rescue_or_fisher"]
        lo, hi = row["rescue_or_lo"], row["rescue_or_hi"]
        col = _color(or_val, neutral=1)
        ax.plot([lo, hi], [i, i], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(or_val, i, "D", color=col, markersize=8, zorder=3)
        ax.text(hi * 1.15, i,
                f"OR={or_val:.2f}\np={row['rescue_p_fisher']:.3f}\nh={row['cohens_h_rescue']:.2f}",
                va="center", ha="left", fontsize=7)

    ax.axvline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)
    ax.set_xlabel("Rescue OR  [log scale]", fontsize=8)
    ax.set_title("Rescue Rate OR (search / no_search)\n+ 95% Bootstrap CI + Cohen's h", fontsize=9, fontweight="bold")
    ax.invert_yaxis()

    # ── Panel 3: Δγ + CI (linear) ────────────────────────────────────────────
    ax = axes[2]
    for i, (_, row) in enumerate(tests_df.iterrows()):
        diff = row["gamma_diff"]
        lo, hi = row["gamma_diff_lo"], row["gamma_diff_hi"]
        if any(np.isnan(v) for v in [diff, lo, hi]):
            continue
        col = _color(diff)
        ax.plot([lo, hi], [i, i], color=col, linewidth=2, solid_capstyle="round")
        ax.plot(diff, i, "D", color=col, markersize=8, zorder=3)
        ax.text(hi + 0.02, i,
                f"Δ={diff:+.3f}\n[{lo:+.2f}, {hi:+.2f}]",
                va="center", ha="left", fontsize=7)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([""] * n)
    ax.set_xlabel("Δγ  (γ_search − γ_no_search)", fontsize=8)
    ax.set_title("Compositional Efficiency Δγ (M2)\n+ 95% Bootstrap CI", fontsize=9, fontweight="bold")
    ax.invert_yaxis()

    # Shared legend note
    fig.text(0.5, -0.04,
             "Green = search favourable  |  Red = search unfavourable  |  Diamonds = point estimates",
             ha="center", fontsize=8, style="italic")
    fig.suptitle("Effect Sizes: Search vs. No-Search", fontsize=12, fontweight="bold", y=1.03)

    plt.savefig(os.path.join(output_dir, "musique_effect_sizes.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> musique_effect_sizes.png")


# ─── LEGACY PLOTS (supplementary) ─────────────────────────────────────────────

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
    df2 = df.copy()
    df2["col"] = df2.apply(
        lambda r: f"{r['model']}|{r['mode']}|{r['question_type']}" + (f"_h{r['hop_index']}" if r['question_type'] == 'factoid' else ""),
        axis=1,
    )
    pivot = df2.pivot_table(index="example_id", columns="col", values="is_correct", aggfunc="first")
    pivot = pivot.fillna(-1).astype(float)
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.5), max(6, len(pivot) * 0.4)))
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#c44e52", "#55a868"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
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


# ─── REPORT ───────────────────────────────────────────────────────────────────

def generate_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    gap_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    fm_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    output_dir: str,
):
    """Full markdown report covering all 7 analyses."""
    lines = ["# MuSiQue Multi-Hop QA Analysis Report\n"]

    # ── Summary Table ──
    lines.append("## Summary\n")
    lines.append("| Model | Mode | Question Type | Accuracy | Count | Avg Search Calls |")
    lines.append("|-------|------|--------------|----------|-------|-----------------|")
    for _, row in summary.iterrows():
        lines.append(f"| {row['model']} | {row['mode']} | {row['question_type']} | "
                     f"{row['accuracy']:.3f} | {int(row['count'])} | {row['avg_search_calls']:.2f} |")

    # ── Analysis 1 ──
    lines.append("\n## Analysis 1: Basic Accuracy\n")
    lines.append("See `musique_basic_accuracy.csv` for full table.\n")

    # ── Analysis 2 ──
    lines.append("## Analysis 2: Compositional Gap\n")
    lines.append("The independence ceiling is the probability of success under perfect composition "
                 "(product of per-hop accuracies). A positive gap means composition is lossy.\n")
    lines.append("| Model | Mode | Ceiling | Aggregate Acc | Comp Gap |")
    lines.append("|-------|------|---------|--------------|---------|")
    for _, r in gap_df.iterrows():
        lines.append(f"| {r['model']} | {r['mode']} | {r['independence_ceiling']:.3f} | "
                     f"{r['aggregate_acc']:.3f} | {r['comp_gap']:+.3f} |")
    lines.append("\n![Compositional Gap](musique_compositional_gap.png)\n")

    # ── Analysis 3 ──
    lines.append("## Analysis 3: Outcome Classification\n")
    lines.append("**Cell A**: full success | **Cell B**: compositional rescue | "
                 "**Cell C**: pure compositional failure | **Cell D**: justified failure\n")
    lines.append("![Outcome Classification](musique_outcome_classification.png)\n")
    lines.append("See `musique_outcome_classification.csv` for raw counts and percentages.\n")

    # ── Analysis 4 ──
    lines.append("## Analysis 4: Rescue Rate & Composition Loss Rate\n")
    lines.append("| Model | Mode | Rescue Rate | Composition Loss Rate |")
    lines.append("|-------|------|-------------|----------------------|")
    for _, r in rates_df.iterrows():
        rescue = f"{r['rescue_rate']:.3f}" if not np.isnan(r["rescue_rate"]) else "N/A"
        comp = f"{r['composition_loss_rate']:.3f}" if not np.isnan(r["composition_loss_rate"]) else "N/A"
        lines.append(f"| {r['model']} | {r['mode']} | {rescue} | {comp} |")
    lines.append("\n![Rates](musique_rates.png)\n")

    # ── Analysis 5 ──
    lines.append("## Analysis 5: Rescue Patterns\n")
    lines.append("Which hop positions fail most often in Cell B (rescue) cases?\n")
    lines.append("![Rescue Patterns](musique_rescue_patterns.png)\n")

    # ── Analysis 6 ──
    lines.append("## Analysis 6: Search Lift\n")
    lines.append("![Hop Search Lift](musique_hop_search_lift.png)\n")
    lines.append("See `musique_search_lift.csv` for the compositional search effect per model.\n")

    # ── Analysis 7 ──
    lines.append("## Analysis 7: Formal Model Comparison\n")
    lines.append("Three nested models predict P(AGG) from per-hop accuracies. "
                 "γ = compositional efficiency; ρ = rescue probability.\n")
    lines.append("| Model | Mode | Best Model | γ (M2) [95% CI] | γ (M3) | ρ (M3) [95% CI] |")
    lines.append("|-------|------|-----------|-----------------|--------|-----------------|")
    for _, r in fm_df.iterrows():
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['best_model']} | "
            f"{r['gamma_m2']:.3f} [{r['gamma_m2_lo']:.3f},{r['gamma_m2_hi']:.3f}] | "
            f"{r['gamma_m3']:.3f} | {r['rho_m3']:.3f} [{r['rho_m3_lo']:.3f},{r['rho_m3_hi']:.3f}] |"
        )
    lines.append("\n![Formal Models](musique_formal_models.png)\n")

    # ── Statistical Tests ──
    lines.append("## Statistical Tests\n")
    lines.append(
        "Effect sizes for the search vs. no-search comparison, per model. "
        "Green = search favourable, Red = search unfavourable.\n"
    )
    lines.append("![Effect Sizes](musique_effect_sizes.png)\n")
    lines.append("| Model | McNemar χ² | p | OR (helps/hurts) | "
                 "Rescue OR [95% CI] | p | Cohen's h | Δγ [95% CI] |")
    lines.append("|-------|-----------|---|-----------------|"
                 "-------------------|---|----------|------------|")
    for _, r in tests_df.iterrows():
        mcn_or = f"{r['mcnemar_or']:.2f}" if np.isfinite(r["mcnemar_or"]) else "∞"
        lines.append(
            f"| {r['model']} | {r['mcnemar_chi2']:.2f} | {r['mcnemar_p']:.4f} | {mcn_or} | "
            f"{r['rescue_or_fisher']:.2f} [{r['rescue_or_lo']:.2f}, {r['rescue_or_hi']:.2f}] | "
            f"{r['rescue_p_fisher']:.4f} | {r['cohens_h_rescue']:.2f} | "
            f"{r['gamma_diff']:+.3f} [{r['gamma_diff_lo']:+.3f}, {r['gamma_diff_hi']:+.3f}] |"
        )
    lines.append("\nSee `musique_statistical_tests.csv` for full details.\n")

    # ── Supplementary Figures ──
    lines.append("## Supplementary Figures\n")
    for fig_name, desc in [
        ("musique_accuracy_comparison.png", "Accuracy Comparison"),
        ("musique_search_delta.png", "Search Delta"),
        ("musique_per_hop_accuracy.png", "Per-Hop Accuracy"),
        ("musique_example_heatmap.png", "Per-Example Heatmap"),
    ]:
        lines.append(f"### {desc}\n")
        lines.append(f"![{desc}]({fig_name})\n")

    report_path = os.path.join(output_dir, "musique_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("  -> musique_report.md")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load & filter
    df = load_results(args.results_dir)
    if df.empty:
        print("No results found. Exiting.")
        return
    df = filter_bad_ids(df, args.bad_ids)
    if df.empty:
        print("All results filtered. Exiting.")
        return

    # Build examples summary (used by most analyses)
    examples_df = build_examples_df(df)

    print("\nGenerating analyses...")

    # Analysis 1
    basic_acc = report_basic_accuracy(df, args.output_dir)

    # Analysis 2
    gap_df = compute_compositional_gap(basic_acc, args.output_dir)

    # Analysis 3
    classify_outcomes(examples_df, args.output_dir)

    # Analysis 4
    rates_df = compute_rates_and_plot(examples_df, args.output_dir)

    # Analysis 5
    analyze_rescue_patterns(examples_df, args.output_dir)

    # Analysis 6
    compute_search_lift(df, examples_df, args.output_dir)

    # Analysis 7 — formal model fitting
    fm_df = fit_formal_models(examples_df, args.output_dir)

    # Statistical tests (McNemar, Fisher + OR CI, Cohen's h, Δγ)
    tests_df = run_statistical_tests(examples_df, fm_df, args.output_dir)

    # Legacy supplementary plots
    print("\nGenerating supplementary plots...")
    summary = generate_summary_csv(df, args.output_dir)
    plot_accuracy_comparison(df, args.output_dir)
    plot_search_delta(df, args.output_dir)
    plot_per_hop_accuracy(df, args.output_dir)
    plot_example_heatmap(df, args.output_dir)

    # Report
    generate_report(df, summary, gap_df, rates_df, fm_df, tests_df, args.output_dir)

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    main()
