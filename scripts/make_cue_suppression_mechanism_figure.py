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
# Matches make_aggregate_cue_tradeoff_figure.py's get_label() (the paper's Figure 1).
PERTURBATION_LABELS = {"natural": "short", "confident_parametric": "confident",
                        "plain": "terse", "searchmulti": "search multiturn"}


def perturbation_label(cue):
    base = cue.split("_", 1)[-1] if "_" in cue else cue
    return PERTURBATION_LABELS.get(base, base)


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "cue_suppression_mechanism.csv"))))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6))
    handles = []
    for ax, ds in zip(axes, DATASETS):
        sub = [r for r in rows if r["dataset"] == ds]
        for model in MODEL_COLORS:
            pts = [r for r in sub if r["model"] == model]
            xs = [float(r["level_shift"]) for r in pts]
            ys = [float(r["slope_change"]) for r in pts]
            sig = [float(r["q_slope_change"]) < 0.05 for r in pts]
            ax.scatter([x for x, s in zip(xs, sig) if not s], [y for y, s in zip(ys, sig) if not s],
                       color=MODEL_COLORS[model], alpha=0.35, s=45, edgecolors="none")
            h = ax.scatter([x for x, s in zip(xs, sig) if s], [y for y, s in zip(ys, sig) if s],
                            color=MODEL_COLORS[model], alpha=0.95, s=90, edgecolors="black", linewidths=0.8,
                            label=MODEL_LABELS[model])
            if ds == DATASETS[0]:
                handles.append(h)
            for r, s in zip(pts, sig):
                if s:
                    ax.annotate(perturbation_label(r["cue"]), (float(r["level_shift"]), float(r["slope_change"])),
                                fontsize=9, xytext=(4, 3), textcoords="offset points", color="#333333")

        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_xlabel("Change in search volume", fontsize=11)
        ax.set_ylabel("Change in uncertainty-tracking", fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.set_title(DATASET_LABELS[ds], fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)

    fig.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"cue_suppression_mechanism_map.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Wrote", path)


if __name__ == "__main__":
    main()
