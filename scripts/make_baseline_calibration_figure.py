"""
Calibration curves: mean no-cue search-call count vs. 5-run semantic entropy,
per (dataset, model), with the two independent baseline rollouts (run A =
frames_cues_full/medqa_grid, run B = frames_cues_rerun/medqa_grid_rerun --
literal repeats of the identical prompt) overlaid to make replication visible
at a glance. If a model's search calls are calibrated to its own uncertainty,
both lines should rise together, left to right, and roughly overlap each other.

Usage:
    uv run python scripts/make_baseline_calibration_figure.py
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "baseline_calibration")

BLUE = "#4575b4"   # run A
RED = "#d73027"    # run B
MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}
DATASETS = ["frames", "medqa"]
DATASET_LABELS = {"frames": "FRAMES", "medqa": "MedQA"}


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "baseline_calibration_by_level.csv"))))
    stats_rows = {(r["dataset"], r["model"]): r
                  for r in csv.DictReader(open(os.path.join(OUT_DIR, "baseline_calibration_stats.csv")))}

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["dataset"], r["model"])].append(r)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2), sharex="row")
    for i, ds in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            cell = sorted(by_cell[(ds, model)], key=lambda r: float(r["entropy_level"]))
            x = [float(r["entropy_level"]) for r in cell]
            ya = [float(r["mean_calls_a"]) for r in cell]
            yb = [float(r["mean_calls_b"]) for r in cell]

            ax.plot(x, ya, "o-", color=BLUE, label="run A", lw=1.6, ms=5)
            ax.plot(x, yb, "s--", color=RED, label="run B", lw=1.6, ms=5)

            ax.set_title(MODEL_LABELS[model], fontsize=10)
            ax.tick_params(labelsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABELS[ds]}\nMean search calls", fontsize=9.5)
            if i == 1:
                ax.set_xlabel("Semantic entropy (bits)", fontsize=9.5)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"baseline_calibration_curves.{ext}")
        fig.savefig(path, dpi=200)
        print("Wrote", path)


if __name__ == "__main__":
    main()
