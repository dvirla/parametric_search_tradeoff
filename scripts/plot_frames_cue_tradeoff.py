#!/usr/bin/env python3
"""Search↔accuracy tradeoff scatter for FRAMES prompt cues, one panel per model.

Companion to ``plot_frames_cue_stars.py``. Where the star plots show *which* cues
move search (and how significantly), this shows the **cost**: for every paired
contrast it plots ΔSearch (x) against ΔAccuracy (y, percentage points), both read
from the Table 2 block of ``results/frames_cues*/SUMMARY_<model>.md``.

Quadrant reading (x<0 = the cue reduces search, the efficient direction):
    top-left     less search, *better* accuracy   → free lunch ✅
    bottom-left  less search, *worse* accuracy     → tradeoff ⚠️
    right half   more search

Marker encoding:
    size      = −log10(p_search)  (bigger = more reliable search effect)
    fill      = significant search effect (p_search<0.05) → solid, else hollow
    red ring  = significant accuracy change (p_acc<0.05)
Contrasts with a significant search effect are labelled.

Usage:
    uv run python scripts/plot_frames_cue_tradeoff.py \
        --summary-dir results/frames_cues_full \
        --output-dir results/frames_cues_full/star_plots
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

P = 0.05
FREE_COLOR = "#0571b0"   # less search, accuracy held → efficiency win
COST_COLOR = "#ca0020"   # accuracy cost
NEUTRAL = "#777777"      # non-significant search effect

# Short labels keyed by contrast number (matches plot_frames_cue_stars.py).
SHORT_LABELS = {
    1: "boost−neut", 2: "hedge−neut", 3: "T−V@PLAIN", 4: "T−V@NAT",
    5: "T−V@ELAB", 6: "T−V@QUERY", 7: "QUERY−PLAIN@V", 8: "QUERY−PLAIN@T",
    9: "NAT−PLAIN@V", 10: "NAT−PLAIN@T", 11: "ELAB−PLAIN@V", 12: "ELAB−PLAIN@T",
    13: "ELAB−NAT@V", 14: "ELAB−NAT@T", 15: "POLITE−PLAIN@V", 16: "POLITE−PLAIN@T",
    17: "POLITE−NAT@V", 18: "POLITE−NAT@T", 19: "DIRECT−PLAIN@V", 20: "DIRECT−PLAIN@T",
    21: "DIRECT−NAT@V", 22: "DIRECT−NAT@T", 23: "DIRECT−POLITE@V", 24: "DIRECT−POLITE@T",
}


def parse_table2(md_path: Path) -> list[dict]:
    """Extract Table 2 rows. Supports the accuracy-augmented 8-column format."""
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
        # # | Comparison | Isolates | Δsearch | p(search) | ΔAcc pp | p(acc) | Tradeoff
        if len(cells) < 7:
            raise SystemExit(
                f"{md_path} Table 2 has {len(cells)} cols; expected the accuracy-augmented "
                "8-column format. Regenerate it with summarize_frames_cues_grid.py."
            )
        rows.append({
            "num": int(cells[0]),
            "comparison": cells[1],
            "d_search": float(cells[3].replace("+", "")),
            "p_search": float(cells[4]),
            "d_acc": float(cells[5].replace("+", "")),
            "p_acc": float(cells[6]),
        })
    return rows


def model_name_from_path(md_path: Path) -> str:
    return md_path.stem.replace("SUMMARY_", "")


def plot_panel(rows: list[dict], model: str, ax: plt.Axes) -> None:
    xs = np.array([r["d_search"] for r in rows])
    ys = np.array([r["d_acc"] for r in rows])
    xpad = max(0.2, np.abs(xs).max() * 0.18)
    ypad = max(0.5, np.abs(ys).max() * 0.18)
    xlo, xhi = xs.min() - xpad, xs.max() + xpad
    ylo, yhi = ys.min() - ypad, ys.max() + ypad

    # quadrant shading: left half = reduces search; tint by accuracy sign
    ax.axhspan(0, yhi, xmin=0, xmax=(0 - xlo) / (xhi - xlo),
               color=FREE_COLOR, alpha=0.06, zorder=0)
    ax.axhspan(ylo, 0, xmin=0, xmax=(0 - xlo) / (xhi - xlo),
               color=COST_COLOR, alpha=0.06, zorder=0)
    ax.axhline(0, color="#444", lw=0.9, zorder=1)
    ax.axvline(0, color="#444", lw=0.9, zorder=1)

    for r in rows:
        sig_s = r["p_search"] < P
        sig_a = r["p_acc"] < P
        size = 40 + 130 * min(-math.log10(max(r["p_search"], 1e-12)) / 13, 1.0)
        if not sig_s:
            face = "none"
            edge = NEUTRAL
        elif r["d_search"] < 0 and (r["d_acc"] >= 0 or not sig_a):
            face = FREE_COLOR
            edge = "black"
        else:
            face = COST_COLOR
            edge = "black"
        ax.scatter(r["d_search"], r["d_acc"], s=size,
                   facecolors=face, edgecolors=COST_COLOR if sig_a else edge,
                   linewidths=2.0 if sig_a else 0.8, zorder=5)
        if sig_s:
            ax.annotate(SHORT_LABELS.get(r["num"], str(r["num"])),
                        (r["d_search"], r["d_acc"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Δ search calls  (← cue reduces search)", fontsize=9)
    ax.set_ylabel("Δ accuracy (pp)  (↑ better)", fontsize=9)
    ax.set_title(model, fontsize=12, fontweight="bold")
    ax.text(0.02, 0.97, "free lunch", transform=ax.transAxes, fontsize=7,
            color=FREE_COLOR, va="top", ha="left", fontweight="bold")
    ax.text(0.02, 0.03, "tradeoff", transform=ax.transAxes, fontsize=7,
            color=COST_COLOR, va="bottom", ha="left", fontweight="bold")
    ax.grid(True, alpha=0.2)


def legend_handles():
    return [
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=FREE_COLOR,
                   markeredgecolor="black", label="↓search, accuracy held (free)"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=COST_COLOR,
                   markeredgecolor="black", label="accuracy cost / ↑search"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   markeredgecolor=NEUTRAL, label="search effect n.s. (hollow)"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   markeredgecolor=COST_COLOR, markeredgewidth=2.0,
                   label="red ring = sig. Δaccuracy (p<0.05)"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor=NEUTRAL,
                   markeredgecolor="none", label="marker size = −log₁₀ p(search)"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="results/frames_cues_full")
    ap.add_argument("--output-dir", default="results/frames_cues_full/star_plots")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(summary_dir.glob("SUMMARY_*.md"))
    if not md_files:
        raise SystemExit(f"No SUMMARY_*.md files found in {summary_dir}")
    parsed = {model_name_from_path(p): parse_table2(p) for p in md_files}

    for model, rows in parsed.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_panel(rows, model, ax)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("FRAMES cues — search vs. accuracy tradeoff", fontsize=12, y=0.98)
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        out = out_dir / f"tradeoff_{model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    if len(parsed) > 1:
        ncols = 2
        nrows = math.ceil(len(parsed) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for i, (model, rows) in enumerate(parsed.items()):
            ax = axes.flat[i]
            ax.set_visible(True)
            plot_panel(rows, model, ax)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=5, fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("FRAMES cues — search vs. accuracy tradeoff across models", fontsize=14, y=0.99)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        out = out_dir / "tradeoff_grid_all_models.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
