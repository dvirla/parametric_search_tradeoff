#!/usr/bin/env python3
"""Star (radar) plots of FRAMES prompt-cue sensitivity, one per model.

Reads the Table 2 paired-contrast block from each
``results/frames_cues/SUMMARY_<model>.md`` and renders a star plot whose
vertices are the cue contrasts and whose radial score is the contrast's
*significance* = -log10(p). Vertices are coloured by direction of the effect on
``search_calls`` (sign of meanΔ): red ↓ = the cue reduces search, blue ↑ =
the cue increases search. A dashed ring marks the p=0.05 threshold
(-log10(0.05) = 1.30).

Usage:
    uv run python scripts/plot_frames_cue_stars.py \
        --summary-dir results/frames_cues \
        --output-dir results/frames_cues/star_plots
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

P_THRESHOLD = 0.05
THRESH_R = -math.log10(P_THRESHOLD)  # 1.301

DOWN_COLOR = "#ca0020"  # reduces search
UP_COLOR = "#0571b0"    # increases search

# Short axis labels keyed by the contrast number (stable across models).
SHORT_LABELS = {
    1: "boost−neut",
    2: "hedge−neut",
    3: "T−V @PLAIN",
    4: "T−V @NAT",
    5: "T−V @ELAB",
    6: "T−V @QUERY",
    7: "QUERY−PLAIN @V",
    8: "QUERY−PLAIN @T",
    9: "NAT−PLAIN @V",
    10: "NAT−PLAIN @T",
    11: "ELAB−PLAIN @V",
    12: "ELAB−PLAIN @T",
    13: "ELAB−NAT @V",
    14: "ELAB−NAT @T",
    15: "POLITE−PLAIN @V",
    16: "POLITE−PLAIN @T",
    17: "POLITE−NAT @V",
    18: "POLITE−NAT @T",
    19: "DIRECT−PLAIN @V",
    20: "DIRECT−PLAIN @T",
    21: "DIRECT−NAT @V",
    22: "DIRECT−NAT @T",
    23: "DIRECT−POLITE @V",
    24: "DIRECT−POLITE @T",
}


def parse_table2(md_path: Path) -> list[dict]:
    """Extract the Table 2 rows from a SUMMARY markdown file."""
    lines = md_path.read_text().splitlines()
    rows: list[dict] = []
    in_t2 = False
    for line in lines:
        if line.startswith("## Table 2"):
            in_t2 = True
            continue
        if in_t2 and line.startswith("## "):
            break
        if not in_t2:
            continue
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # cells: [#, Comparison, Isolates, meanΔ, medΔ, p, Verdict]
        num = int(cells[0])
        comparison = cells[1]
        isolates = cells[2]
        mean_delta = float(cells[3].replace("+", ""))
        p = float(cells[5])
        verdict = cells[6]
        rows.append(
            {
                "num": num,
                "comparison": comparison,
                "isolates": isolates,
                "mean_delta": mean_delta,
                "p": p,
                "verdict": verdict,
            }
        )
    return rows


def model_name_from_path(md_path: Path) -> str:
    return md_path.stem.replace("SUMMARY_", "")


def plot_star(rows: list[dict], model: str, ax: plt.Axes) -> None:
    n = len(rows)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # significance score; clip tiny p so the spike is finite & readable
    scores = np.array([-math.log10(max(r["p"], 1e-12)) for r in rows])
    closed_scores = np.concatenate([scores, scores[:1]])
    closed_angles = np.concatenate([angles, angles[:1]])

    rmax = max(scores.max() * 1.12, THRESH_R + 0.5)

    # outline polygon
    ax.plot(closed_angles, closed_scores, color="#444444", lw=1.2, zorder=3)
    ax.fill(closed_angles, closed_scores, color="#999999", alpha=0.12, zorder=1)

    # p=0.05 threshold ring
    ring = np.full(200, THRESH_R)
    ring_ang = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ring_ang, ring, color="green", ls="--", lw=1.0, alpha=0.7, zorder=2)

    # vertices coloured by direction
    for ang, sc, r in zip(angles, scores, rows):
        color = DOWN_COLOR if r["mean_delta"] < 0 else UP_COLOR
        sig = r["p"] < P_THRESHOLD
        ax.scatter(
            ang,
            sc,
            s=70 if sig else 35,
            color=color,
            edgecolors="black" if sig else "none",
            linewidths=0.8,
            zorder=5,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels([SHORT_LABELS.get(r["num"], str(r["num"])) for r in rows], fontsize=7)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(model, fontsize=12, pad=22, fontweight="bold")
    # annotate threshold value
    ax.text(np.pi / 2, THRESH_R, "p=0.05", color="green", fontsize=6, ha="center", va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="results/frames_cues")
    ap.add_argument("--output-dir", default="results/frames_cues/star_plots")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(summary_dir.glob("SUMMARY_*.md"))
    if not md_files:
        raise SystemExit(f"No SUMMARY_*.md files found in {summary_dir}")

    parsed = {model_name_from_path(p): parse_table2(p) for p in md_files}

    # legend handles (shared)
    legend_handles = [
        plt.Line2D([], [], marker="o", ls="", color=DOWN_COLOR, markeredgecolor="black",
                   label="↓ reduces search (meanΔ<0)"),
        plt.Line2D([], [], marker="o", ls="", color=UP_COLOR, markeredgecolor="black",
                   label="↑ increases search (meanΔ>0)"),
        plt.Line2D([], [], color="green", ls="--", label="p=0.05 threshold"),
        plt.Line2D([], [], marker="o", ls="", color="gray", markeredgecolor="black",
                   label="ringed = significant (p<0.05)"),
    ]

    # ---- per-model individual plots ----
    for model, rows in parsed.items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        plot_star(rows, model, ax)
        fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("FRAMES prompt-cue significance star — radius = −log₁₀(p)",
                     fontsize=11, y=0.98)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = out_dir / f"star_{model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # ---- combined grid ----
    n_models = len(parsed)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig = plt.figure(figsize=(7 * ncols, 7 * nrows))
    for i, (model, rows) in enumerate(parsed.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        plot_star(rows, model, ax)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FRAMES prompt-cue significance stars — radius = −log₁₀(p), "
                 "colour = effect direction on search_calls", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = out_dir / "star_grid_all_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
