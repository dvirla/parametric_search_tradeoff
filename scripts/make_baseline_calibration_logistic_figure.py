"""
Reliability diagram for the logistic baseline-calibration companion
(analyze_baseline_calibration_logistic.py): observed P(search) vs. the
logistic model's predicted P(search) at each entropy level, per (dataset,
model), run A and run B overlaid. A well-calibrated model in the strict
forecasting sense sits on the y=x diagonal, not just "higher entropy -> more
search" (that's discrimination, already shown in baseline_calibration_curves.png
-- this is the complementary calibration check).

Usage:
    uv run python scripts/make_baseline_calibration_logistic_figure.py
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "baseline_calibration_logistic")

BLUE = "#4575b4"   # run A
RED = "#d73027"    # run B
MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}
DATASETS = ["frames", "medqa"]
DATASET_LABELS = {"frames": "FRAMES", "medqa": "MedQA"}


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "baseline_calibration_logistic_by_level.csv"))))
    stats_rows = {(r["dataset"], r["model"]): r
                  for r in csv.DictReader(open(os.path.join(OUT_DIR, "baseline_calibration_logistic_stats.csv")))}

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["dataset"], r["model"])].append(r)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex="row", sharey=True)
    for i, ds in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            cell = sorted(by_cell[(ds, model)], key=lambda r: float(r["entropy_level"]))
            x = [float(r["entropy_level"]) for r in cell]
            obs_a = [float(r["pct_searched_a"]) / 100.0 for r in cell]
            obs_b = [float(r["pct_searched_b"]) / 100.0 for r in cell]
            pred_a = [float(r["predicted_p_a"]) if r["predicted_p_a"] else None for r in cell]
            pred_b = [float(r["predicted_p_b"]) if r["predicted_p_b"] else None for r in cell]
            n = [int(r["n"]) for r in cell]

            ax.plot(x, obs_a, "o", color=BLUE, ms=6, label="observed, run A")
            ax.plot(x, obs_b, "s", color=RED, ms=6, mfc="none", label="observed, run B")
            if all(p is not None for p in pred_a):
                order = sorted(range(len(x)), key=lambda k: x[k])
                ax.plot([x[k] for k in order], [pred_a[k] for k in order], "-", color=BLUE, lw=1.3, alpha=0.6,
                        label="fitted P(search), run A")
            if all(p is not None for p in pred_b):
                order = sorted(range(len(x)), key=lambda k: x[k])
                ax.plot([x[k] for k in order], [pred_b[k] for k in order], "--", color=RED, lw=1.3, alpha=0.6,
                        label="fitted P(search), run B")
            for xi, yi, ni in zip(x, obs_a, n):
                ax.annotate(str(ni), (xi, yi), textcoords="offset points", xytext=(0, 6),
                            fontsize=6, ha="center", color="#555555")

            st = stats_rows.get((ds, model))
            if st and st["auc_a"] and st["auc_b"]:
                title = (f"{MODEL_LABELS[model]}\n"
                         f"AUC={float(st['auc_a']):.2f}/{float(st['auc_b']):.2f}  "
                         f"Brier={float(st['brier_a']):.3f}/{float(st['brier_b']):.3f} "
                         f"(null={float(st['brier_null_a']):.3f})")
            else:
                title = f"{MODEL_LABELS[model]}\n(too few positives to fit)"
            ax.set_title(title, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABELS[ds]}\nP(search)")
            if i == 1:
                ax.set_xlabel("semantic entropy (bits)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    fig.suptitle("Reliability diagram: observed vs. fitted P(search | entropy), split-half replicated\n"
                 "(y=fitted line on the diagonal with observed points = well-calibrated; Brier score below the "
                 "null baseline = the entropy model beats predicting the base rate)",
                 fontsize=11, y=1.0)
    fig.text(0.5, 0.01,
              "Brier score (lower is better) is reported against its null-model floor (always predicting the "
              "overall search rate) -- close to null means entropy adds little probability-calibration value\n"
              "even where AUC (discrimination/ranking) is non-trivial. Numbers above run-A points = n examples "
              "at that entropy level.",
              ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"baseline_calibration_logistic_reliability.{ext}")
        fig.savefig(path, dpi=200)
        print("Wrote", path)


if __name__ == "__main__":
    main()
