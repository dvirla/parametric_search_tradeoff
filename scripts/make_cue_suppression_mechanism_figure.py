"""
Mechanism map: x = level shift (b_is_cue, pure suppression/boost unrelated to
necessity), y = slope change (b_interaction, change in necessity-SENSITIVITY
itself). Each point is one (dataset, model, cue) cell from
analyze_cue_suppression_mechanism.py. Quadrants:
  - near y=0, x<0: blanket suppression, calibration intact
  - y<0 (below the line): calibration eroding/inverting -- necessity-blind
  - y>0: calibration sharpening -- MORE necessity-sensitive under this cue

Usage:
    uv run python scripts/make_cue_suppression_mechanism_figure.py
"""
import csv
import os

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "cue_suppression_mechanism")

MODEL_COLORS = {
    "gemma4_31b": "#d73027", "gpt-oss_120b": "#4575b4",
    "gpt-oss_20b": "#91bfdb", "nemotron-3-nano_30b": "#1a9850",
}
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}
DATASETS = ["frames", "medqa"]
DATASET_LABELS = {"frames": "FRAMES", "medqa": "MedQA"}


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "cue_suppression_mechanism.csv"))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, ds in zip(axes, DATASETS):
        sub = [r for r in rows if r["dataset"] == ds]
        for model in MODEL_COLORS:
            pts = [r for r in sub if r["model"] == model]
            xs = [float(r["level_shift"]) for r in pts]
            ys = [float(r["slope_change"]) for r in pts]
            sig = [float(r["q_slope_change"]) < 0.05 for r in pts]
            ax.scatter([x for x, s in zip(xs, sig) if not s], [y for y, s in zip(ys, sig) if not s],
                       color=MODEL_COLORS[model], alpha=0.35, s=45, edgecolors="none")
            ax.scatter([x for x, s in zip(xs, sig) if s], [y for y, s in zip(ys, sig) if s],
                       color=MODEL_COLORS[model], alpha=0.95, s=90, edgecolors="black", linewidths=0.8,
                       label=MODEL_LABELS[model])
            for r, s in zip(pts, sig):
                if s:
                    label = r["cue"].split("_", 1)[-1] if "_" in r["cue"] else r["cue"]
                    ax.annotate(label, (float(r["level_shift"]), float(r["slope_change"])),
                                fontsize=6.5, xytext=(4, 3), textcoords="offset points", color="#333333")

        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_xlabel("level shift  (calls at zero entropy; suppression <0 <boost>)")
        ax.set_ylabel("slope change  (necessity-sensitivity; erodes <0 <sharpens>)")
        ax.set_title(DATASET_LABELS[ds], fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8, loc="lower left", title="filled+outlined = FDR-sig. slope change", title_fontsize=7)
    fig.suptitle("Mechanism map: does a cue suppress search by a blanket amount\n"
                 "(x-axis) or by eroding necessity-sensitivity itself (y-axis)?",
                 fontsize=12.5, y=1.03)
    fig.text(0.5, -0.04,
              "Each point = one (model, cue). Faded = slope change not FDR-significant (q>=0.05): treat as pure\n"
              "level shift, calibration intact. Outlined = significant: labeled points show a real change in how\n"
              "necessity-sensitive search is under that cue, not just how much search happens overall.",
              ha="center", fontsize=8, color="#555555")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"cue_suppression_mechanism_map.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Wrote", path)


if __name__ == "__main__":
    main()
