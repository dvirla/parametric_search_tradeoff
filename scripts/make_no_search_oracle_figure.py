"""
Two-tier accuracy comparison per (dataset, model): no-search floor (mean
per-run parametric-only accuracy) vs. plain/default search-enabled accuracy,
BOTH LLM-judge graded (results/no_search_llm_grades/ + the existing live
`plain`-condition grades) -- directly comparable, no grading-artifact confound.

Supersedes the original 3-bar version (no-search / plain / oracle ceiling): the
oracle ceiling was regex-graded across the full cue battery and, once the
no-search floor was properly LLM-regraded, came out BELOW the corrected floor
on MedQA in every model (a floor can't exceed its own ceiling) -- a direct
consequence of regex undercounting MedQA by 26-36pp. Re-adding a valid oracle
bar would require LLM-regrading every cue condition too (out of scope here);
until that's done, this figure reports only the two bars that are mutually
consistent and fully validated.

Usage:
    uv run python scripts/make_no_search_oracle_figure.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "search_oracle")

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}
DATASETS = ["frames", "medqa"]
DATASET_LABELS = {"frames": "FRAMES", "medqa": "MedQA"}

GRAY = "#999999"     # no-search floor
BLUE = "#4575b4"      # plain (actual) accuracy


def main():
    ns = {(r["dataset"], r["model"]): r for r in csv.DictReader(
        open(os.path.join(REPO, "results", "no_search_accuracy", "no_search_accuracy_llm.csv")))}

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(MODELS))
        w = 0.32
        floor = [100 * float(ns[(ds, m)]["per_run_acc_mean"]) for m in MODELS]
        plain = [100 * float(ns[(ds, m)]["plain_search_acc"]) for m in MODELS]

        b1 = ax.bar(x - w / 2, floor, width=w, color=GRAY, label="no-search floor\n(mean per-run, parametric only)")
        b2 = ax.bar(x + w / 2, plain, width=w, color=BLUE, label="plain (default) accuracy\n(search enabled)")
        for xi, f, p in zip(x, floor, plain):
            d = p - f
            ax.text(xi, max(f, p) + 2, f"{d:+.1f}pp", ha="center", fontsize=8.5,
                     color="#1a9850" if d > 3 else ("#b35806" if d < -3 else "#555555"), fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("accuracy (%, LLM-judge graded)")
        ax.set_ylim(0, 100)
        ax.set_title(DATASET_LABELS[ds], fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Does search add value beyond parametric knowledge?\n"
                 "No-search floor vs. actual (plain) behavior, both LLM-judge graded, same examples",
                 fontsize=12.5, y=1.02)
    fig.text(0.5, -0.04,
              "Labels = plain minus no-search (pp). FRAMES: search adds +15 to +35pp in every model.\n"
              "MedQA: search adds essentially nothing (-1.5pp to +1.5pp) despite a much higher absolute\n"
              "knowledge floor (70-80%) than FRAMES's (24-42%) -- these models largely already know MedQA.",
              ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"no_search_oracle_comparison.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Wrote", path)


if __name__ == "__main__":
    main()
