"""
Re-generate interplay plots for a 2-hop subset of an existing run.

Loads interplay_summary.csv from the source directory, filters to num_hops == 2,
reconstructs an approximate example_df via groupby aggregation, and saves all
plots to a '2hop/' subdirectory.

Usage:
    uv run python scripts/plot_2hop_subset.py \
        --source-dir results/musique_parametric/interplay_analysis
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_parametric_search_interplay import (
    plot_accuracy_per_hop,
    plot_accuracy_outcomes,
    plot_searched_vs_unsearched,
    plot_search_calibration,
    plot_search_calibration_extended,
    plot_certain_vs_uncertain_queries,
    plot_accuracy_vs_queries_bars,
    plot_entropy_buckets,
    plot_redundancy_breakdown,
    plot_sequential_redundancy_distribution,
    plot_redundancy_overlap,
    plot_search_usefulness_by_certainty,
    plot_search_usefulness_by_entropy,
    plot_searches_per_certainty_level,
    plot_searches_per_entropy_level,
    plot_missed_search_cross_hop_coverage,
    plot_queries_violin,
)


def reconstruct_example_df(hop_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate example-level metrics from hop-level CSV data."""
    rows = []
    for (model, example_id), grp in hop_df.groupby(["model", "example_id"], sort=False):
        total = grp["queries_assigned_count"].sum()
        par_red = grp.loc[grp["parametric_certain"], "queries_assigned_count"].sum()

        # Sequential redundant: hops where seq_redundant=True had at least 1
        # sequential-redundant query; approximate as (queries_assigned_count - 1).
        seq_red = (
            grp.loc[grp["seq_redundant"], "queries_assigned_count"]
            .clip(lower=1)
            .sub(1)
            .sum()
        )

        both_red = (
            grp.loc[grp["parametric_certain"] & grp["seq_redundant"], "queries_assigned_count"]
            .clip(lower=1)
            .sub(1)
            .sum()
        )

        par_only = par_red
        seq_only = max(0, seq_red - both_red)
        total_red = int(
            grp.loc[grp["parametric_certain"] | grp["seq_redundant"], "queries_assigned_count"]
            .sum()
        )
        necessary = max(0, total - total_red)

        mean_par_acc = grp["parametric_accuracy"].mean()
        frac_certain = grp["parametric_certain"].mean()
        aggregate_correct = grp["aggregate_correct"].iloc[0]
        n_runs = grp["num_runs"].iloc[0]

        rows.append({
            "model": model,
            "example_id": example_id,
            "num_hops": int(grp["num_hops"].iloc[0]),
            "total_search_calls": int(total),
            "parametric_redundant_count": int(par_red),
            "sequential_redundant_count": int(seq_red),
            "both_redundant_count": int(both_red),
            "parametric_only_count": int(par_only),
            "sequential_only_count": int(seq_only),
            "total_redundant_count": int(total_red),
            "necessary_count": int(necessary),
            "mean_parametric_accuracy": mean_par_acc,
            "frac_hops_certain": frac_certain,
            "aggregate_correct": aggregate_correct,
            "no_search_agg_runs_correct": 0,
            "no_search_agg_accuracy": float("nan"),
            "n_runs": n_runs,
            "nosearch_source": "composed",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot 2-hop subset from existing interplay results")
    parser.add_argument("--source-dir", required=True, help="Directory containing interplay_summary.csv")
    parser.add_argument("--output-dir", help="Output directory (default: <source-dir>/2hop)")
    args = parser.parse_args()

    csv_path = os.path.join(args.source_dir, "interplay_summary.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.source_dir, "2hop")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {csv_path}...")
    hop_df_full = pd.read_csv(csv_path)
    print(f"  Total rows: {len(hop_df_full)}, hop distribution:\n{hop_df_full['num_hops'].value_counts().sort_index()}")

    hop_df = hop_df_full[hop_df_full["num_hops"] == 2].copy()
    print(f"  After filtering to 2-hop: {len(hop_df)} rows, {hop_df['example_id'].nunique()} examples")

    if hop_df.empty:
        print("No 2-hop rows found. Exiting.")
        sys.exit(1)

    print("Reconstructing example-level metrics...")
    example_df = reconstruct_example_df(hop_df)
    print(f"  {len(example_df)} example rows")

    print(f"\nGenerating plots → {output_dir}")
    plot_accuracy_per_hop(hop_df, output_dir)
    plot_accuracy_outcomes(hop_df, output_dir)
    plot_searched_vs_unsearched(hop_df, output_dir)
    plot_search_calibration(hop_df, output_dir)
    plot_search_calibration_extended(hop_df, output_dir)
    plot_certain_vs_uncertain_queries(hop_df, output_dir)
    plot_accuracy_vs_queries_bars(hop_df, output_dir)
    plot_entropy_buckets(hop_df, output_dir)
    plot_redundancy_breakdown(example_df, output_dir)
    plot_sequential_redundancy_distribution(example_df, output_dir)
    plot_redundancy_overlap(example_df, output_dir)
    plot_search_usefulness_by_certainty(example_df, output_dir)
    plot_search_usefulness_by_entropy(example_df, hop_df, output_dir)
    plot_searches_per_certainty_level(hop_df, output_dir)
    plot_searches_per_entropy_level(hop_df, output_dir)
    plot_missed_search_cross_hop_coverage(hop_df, output_dir)
    try:
        plot_queries_violin(hop_df, output_dir)
    except Exception as e:
        print(f"  Violin plot skipped: {e}")

    print(f"\nDone. All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
