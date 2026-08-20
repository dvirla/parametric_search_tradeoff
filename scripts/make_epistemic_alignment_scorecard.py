"""
Instantiates the Stage 0-2 taxonomy from docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md on
this project's existing data: for each (model, dataset), a stacked bar of how many
cues fall into each fragility-mechanism label (Null / Uniform volume shift /
Sharpening / Erosion / Inversion), annotated with the Stage-1 baseline-calibration
verdict (from split-half replication).

Refines cue_suppression_mechanism.csv's "mechanism" column (which only
distinguishes calibration-intact vs eroded/sharpened) by ALSO checking whether the
level shift itself was significant, splitting "level shift only" into:
  - Null:              level shift not significant, slope change not significant
  - Uniform volume shift: level shift significant, slope change not significant

Usage:
    uv run python scripts/make_epistemic_alignment_scorecard.py
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "epistemic_alignment_scorecard")
os.makedirs(OUT_DIR, exist_ok=True)

MECH_PATH = os.path.join(REPO, "results", "cue_suppression_mechanism", "cue_suppression_mechanism.csv")
CALIB_PATH = os.path.join(REPO, "results", "baseline_calibration", "baseline_calibration_stats.csv")

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}
DATASETS = ["frames", "medqa"]
DATASET_LABELS = {"frames": "FRAMES", "medqa": "MedQA"}

LABELS_ORDER = ["Inversion", "Erosion", "Null", "Uniform volume shift", "Sharpening"]
COLORS = {
    "Null": "#c7c7c7",
    "Uniform volume shift": "#2a78d6",
    "Sharpening": "#008300",
    "Erosion": "#eb6834",
    "Inversion": "#e34948",
}


def refine_label(row):
    mech = row["mechanism"]
    p_level = float(row["p_level_shift"])
    if mech == "level shift only (calibration intact)":
        return "Uniform volume shift" if p_level < 0.05 else "Null"
    if mech == "calibration eroded":
        return "Erosion"
    if mech == "calibration inverted":
        return "Inversion"
    if mech == "calibration sharpened":
        return "Sharpening"
    return "Uniform volume shift"  # "marginally changed" cells: small, non-dominant slope shift alongside a level shift


def main():
    mech_rows = list(csv.DictReader(open(MECH_PATH)))
    calib_rows = {(r["dataset"], r["model"]): r for r in csv.DictReader(open(CALIB_PATH))}

    counts = defaultdict(lambda: defaultdict(int))
    for row in mech_rows:
        key = (row["dataset"], row["model"])
        label = refine_label(row)
        counts[key][label] += 1

    cells = [(ds, m) for ds in DATASETS for m in MODELS]
    fig, ax = plt.subplots(figsize=(11, 6.5))

    y = np.arange(len(cells))
    left = np.zeros(len(cells))
    for label in LABELS_ORDER:
        vals = np.array([counts[c].get(label, 0) for c in cells])
        ax.barh(y, vals, left=left, color=COLORS[label], label=label, height=0.62, edgecolor="white", linewidth=0.6)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, yi, str(v), ha="center", va="center", fontsize=8.5,
                         color="white" if label in ("Inversion", "Erosion", "Sharpening", "Uniform volume shift") else "#333333")
        left += vals

    ytick_labels = []
    for ds, m in cells:
        calib = calib_rows.get((ds, m))
        badge = "calibrated" if calib and calib["replicates_both_sig"] == "True" else "not calibrated"
        ytick_labels.append(f"{DATASET_LABELS[ds]} — {MODEL_LABELS[m]}  [{badge}]")
    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("number of cues (out of the tested battery) falling into each mechanism label")
    ax.spines[["top", "right"]].set_visible(False)

    handles, hlabels = ax.get_legend_handles_labels()
    order = [hlabels.index(l) for l in LABELS_ORDER]
    ax.legend([handles[i] for i in order], [hlabels[i] for i in order],
               frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)

    ax.set_title("Epistemic-alignment fragility scorecard\n"
                 "Stage 1 (calibrated?) + Stage 2 (per-cue mechanism), 4 models x 2 datasets",
                 fontsize=12.5, pad=14)
    fig.text(0.5, 0.02,
              "[calibrated] = Stage-1 split-half-replicated baseline calibration. Bars = Stage-2 taxonomy over the tested cue\n"
              "battery, FDR-corrected. Blue = discrimination intact, absolute volume shifted (not accuracy-free -- see framework doc).\n"
              "Orange/red = discrimination itself degrades/inverts. Green = discrimination improves beyond baseline.",
              ha="center", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"epistemic_alignment_scorecard.{ext}")
        fig.savefig(path, dpi=200)
        print("Wrote", path)

    out_csv = os.path.join(OUT_DIR, "epistemic_alignment_scorecard.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "calibrated"] + LABELS_ORDER)
        for ds, m in cells:
            calib = calib_rows.get((ds, m))
            badge = calib and calib["replicates_both_sig"] == "True"
            w.writerow([ds, m, badge] + [counts[(ds, m)].get(l, 0) for l in LABELS_ORDER])
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
