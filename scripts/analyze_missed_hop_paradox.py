#!/usr/bin/env python3
"""Drill-down on the "cost of a missed hop" paradox.

The missed-hop-cost figures (make_paper_figures.py, Rung 1/2) argue that skipping
an uncertain hop lowers correctness. But user-like (natural) phrasing both *reduces
search* and *raises* overall accuracy versus benchmark phrasing — which on its face
looks like "searching less is better", undermining the cost story.

This script resolves the paradox by pairing every MuSiQue example across the two
phrasings (same example_id) and focusing on the discordant cell where the model is
CORRECT under natural phrasing but WRONG under benchmark phrasing ("paradox cell").
For that cell it asks two questions:

  1. Was benchmark actually *searching more* (flailing) in exactly these cases?
  2. Does the natural rewrite *leak* intermediate (bridge) gold answers into the
     question text, collapsing hops so that less search is legitimately required?

Inputs (per model slug) are the same artifacts make_paper_figures.py consumes:
  results/musique_parametric/interplay_analysis/matched_examples_<slug>.json   (benchmark)
  results/musique-natural/interplay_analysis/matched_examples_<slug>.json      (natural)
  results/{musique_parametric,musique-natural}/interplay_analysis/interplay_summary.csv

Outputs (default results/missed_hop_paradox/):
  paired_outcomes.csv          per (model, example_id) paired record
  fig_paired_2x2.png/pdf        per-model bench×nat outcome counts (paradox cell flagged)
  fig_paradox_profile.png/pdf   searches & hop-leak, paradox cell vs its mirror
  report.md                     narrative with the headline numbers
  discordant_examples.md        side-by-side qualitative dump (sorted by bench flailing)

Run:
  uv run python scripts/analyze_missed_hop_paradox.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import viz  # noqa: E402

CELL_LABELS = {
    (False, True): "Paradox\n(nat ✓ / bench ✗)",
    (True, False): "Mirror\n(bench ✓ / nat ✗)",
    (True, True): "Both ✓",
    (False, False): "Both ✗",
}
CELL_COLORS = {
    (False, True): viz.NATURAL,     # red — the paradox cell, highlighted
    (True, False): viz.BENCHMARK,   # blue
    (True, True): viz.EFFECTIVE,
    (False, False): viz.ERR_MUTE,
}


def _parse(v):
    """matched_examples columns are either real lists or stringified Python reprs."""
    if isinstance(v, (list, dict)):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return eval(v, {"nan": float("nan"), "true": True, "false": False, "null": None})


def _is_true(v) -> bool:
    return str(v) == "True"


def _bridge_leak(question: str, golds: list[str]) -> int:
    """How many *bridge* (non-final) gold hop answers appear verbatim in the question.

    Bridge entities are intermediate answers the model would otherwise have to
    discover; their presence in the prompt collapses a reasoning hop.
    """
    ql = (question or "").lower()
    return sum(1 for g in golds[:-1] if g and len(str(g)) >= 2 and str(g).lower() in ql)


def load_matched(path: str) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {e["example_id"]: e for e in data}


def build_paired(bench_dir: str, nat_dir: str, slugs: list[str]) -> pd.DataFrame:
    rows = []
    for slug in slugs:
        bpath = os.path.join(bench_dir, f"matched_examples_{slug}.json")
        npath = os.path.join(nat_dir, f"matched_examples_{slug}.json")
        if not (os.path.exists(bpath) and os.path.exists(npath)):
            print(f"  skip {slug}: missing matched_examples")
            continue
        b, n = load_matched(bpath), load_matched(npath)
        for eid in set(b) & set(n):
            be, ne = b[eid], n[eid]
            sq = _parse(ne["sub_questions"]) or _parse(be["sub_questions"])
            golds = [s.get("gold_answer") for s in sq]
            bc, nc = _is_true(be["aggregate_correct"]), _is_true(ne["aggregate_correct"])
            rows.append({
                "model": slug,
                "model_name": viz.display_name(slug),
                "example_id": eid,
                "num_hops": len(sq),
                "bench_correct": bc,
                "nat_correct": nc,
                "cell": CELL_LABELS[(bc, nc)],
                "n_search_bench": len(_parse(be["query_records"])),
                "n_search_nat": len(_parse(ne["query_records"])),
                "bridge_leak_bench": _bridge_leak(be["aggregate_question"], golds),
                "bridge_leak_nat": _bridge_leak(ne["aggregate_question"], golds),
                "q_bench": be["aggregate_question"],
                "q_nat": ne["aggregate_question"],
            })
    return pd.DataFrame(rows)


def attach_missed_counts(paired: pd.DataFrame, bench_summary: str, nat_summary: str) -> pd.DataFrame:
    """Add per-example #truly-missed and #uncertain hops from the interplay summaries."""
    def per_example(path, variant):
        if not os.path.exists(path):
            return None
        d = viz.add_cost_cells(viz.load_interplay_summary(path, variant))
        g = (d.groupby(["model", "example_id"])
               .agg(**{f"n_truly_missed_{variant}": ("truly_missed", "sum"),
                       f"n_uncertain_{variant}": ("uncertain", "sum")})
               .reset_index())
        return g

    for path, variant in [(bench_summary, "bench"), (nat_summary, "nat")]:
        g = per_example(path, variant)
        if g is not None:
            paired = paired.merge(g, on=["model", "example_id"], how="left")
    return paired


# ──────────────────────────────────────────────────────────────────────────────
def fig_paired_2x2(paired: pd.DataFrame, outdir: Path) -> None:
    models = [m for m in viz.MODEL_ORDER if m in set(paired.model_name)]
    cells = [(False, True), (True, False), (True, True), (False, False)]
    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(models)), 5.2))
    x = np.arange(len(models)); w = 0.2
    for j, c in enumerate(cells):
        counts = [int(((paired.model_name == m) & (paired.bench_correct == c[0])
                       & (paired.nat_correct == c[1])).sum()) for m in models]
        off = (j - (len(cells) - 1) / 2) * w
        bars = ax.bar(x + off, counts, w, color=CELL_COLORS[c],
                      edgecolor=("black" if c == (False, True) else "none"),
                      linewidth=1.6, label=CELL_LABELS[c].replace("\n", " "), zorder=3)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, str(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("# paired examples", fontsize=10)
    ax.set_title("Benchmark vs. natural outcome on the same MuSiQue questions\n"
                 "the paradox cell (red outline) is where natural wins and benchmark fails",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    viz.savefig(fig, outdir, "fig_paired_2x2")


def fig_paradox_profile(paired: pd.DataFrame, outdir: Path) -> None:
    """For the paradox cell (FT) vs its mirror (TF): mean searches and bridge-leak,
    benchmark vs natural. Shows benchmark flails (more search) yet loses, and that
    natural's win coincides with intermediate answers leaking into the phrasing."""
    models = [m for m in viz.MODEL_ORDER if m in set(paired.model_name)]
    FT, TF = (False, True), (True, False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(models)); w = 0.36

    # Panel A — mean search calls in the paradox cell, benchmark vs natural.
    ax = axes[0]
    ft = paired[(paired.bench_correct == FT[0]) & (paired.nat_correct == FT[1])]
    for off, col, color, lbl in [(-w / 2, "n_search_bench", viz.BENCHMARK, "Benchmark phrasing"),
                                 (w / 2, "n_search_nat", viz.NATURAL, "Natural phrasing")]:
        means = [ft[ft.model_name == m][col].mean() for m in models]
        ses = [ft[ft.model_name == m][col].sem() for m in models]
        bars = ax.bar(x + off, means, w, yerr=ses, capsize=4, color=color, label=lbl, zorder=3)
        for bar, mn in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mn, f"{mn:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean search calls per question (±SEM)", fontsize=10)
    ax.set_title("Paradox cell: benchmark searches MORE yet fails\n"
                 "(natural wins with far fewer searches)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B — bridge-leak: intermediate gold answers verbatim in question text.
    ax = axes[1]
    for off, cell, hatch, lbl in [(-w / 2, FT, "", "Paradox cell (nat ✓ / bench ✗)"),
                                  (w / 2, TF, "//", "Mirror cell (bench ✓ / nat ✗)")]:
        sub = paired[(paired.bench_correct == cell[0]) & (paired.nat_correct == cell[1])]
        # natural-phrasing leak minus benchmark-phrasing leak = hops the rewrite gave away
        means = [(sub[sub.model_name == m].bridge_leak_nat
                  - sub[sub.model_name == m].bridge_leak_bench).mean() for m in models]
        ses = [(sub[sub.model_name == m].bridge_leak_nat
                - sub[sub.model_name == m].bridge_leak_bench).sem() for m in models]
        color = viz.NATURAL if cell == FT else viz.BENCHMARK
        bars = ax.bar(x + off, means, w, yerr=ses, capsize=4, color=color, hatch=hatch,
                      edgecolor="white", label=lbl, zorder=3)
        for bar, mn in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mn, f"{mn:+.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("Extra bridge gold-answers leaked\nby natural phrasing (nat − bench)", fontsize=10)
    ax.set_title("Natural rewrites leak intermediate answers\n(collapsing hops → less search needed)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Resolving the missed-hop paradox: where natural beats benchmark, "
                 "benchmark was flailing and the rewrite leaked hops",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    viz.savefig(fig, outdir, "fig_paradox_profile")


def write_reports(paired: pd.DataFrame, outdir: Path, n_examples: int = 25) -> None:
    models = [m for m in viz.MODEL_ORDER if m in set(paired.model_name)]
    FT = (False, True)

    # ── report.md ────────────────────────────────────────────────────────────
    lines = ["# Cost-of-a-Missed-Hop — Paradox Drill-down\n",
             "Pairing every MuSiQue example across benchmark and natural phrasing "
             "(same `example_id`). The **paradox cell** = correct under natural, wrong "
             "under benchmark — the cases that make 'less search, higher accuracy' look "
             "like missed hops are free.\n",
             "## Outcome breakdown (paired)\n",
             "| Model | Both ✓ | Paradox (nat✓/bench✗) | Mirror (bench✓/nat✗) | Both ✗ | bench acc | nat acc |",
             "|---|---|---|---|---|---|---|"]
    for m in models:
        s = paired[paired.model_name == m]
        tt = int((s.bench_correct & s.nat_correct).sum())
        ft = int((~s.bench_correct & s.nat_correct).sum())
        tf = int((s.bench_correct & ~s.nat_correct).sum())
        ff = int((~s.bench_correct & ~s.nat_correct).sum())
        lines.append(f"| {m} | {tt} | {ft} | {tf} | {ff} | "
                     f"{s.bench_correct.mean():.3f} | {s.nat_correct.mean():.3f} |")

    lines += ["\n## Inside the paradox cell\n",
              "Two non-rival explanations, neither of which says missed hops are costless:\n",
              "| Model | n | mean search bench | mean search nat | "
              "bridge-leak bench | bridge-leak nat |",
              "|---|---|---|---|---|---|"]
    for m in models:
        s = paired[(paired.model_name == m) & ~paired.bench_correct & paired.nat_correct]
        lines.append(f"| {m} | {len(s)} | {s.n_search_bench.mean():.2f} | "
                     f"{s.n_search_nat.mean():.2f} | {s.bridge_leak_bench.mean():.2f} | "
                     f"{s.bridge_leak_nat.mean():.2f} |")
    if "n_truly_missed_nat" in paired.columns:
        lines += ["\n### Truly-missed hops in the paradox cell\n",
                  "If natural were 'getting away with' skipping hops, we'd expect many "
                  "truly-missed hops here yet still correct. Compare to mirror cell.\n",
                  "| Model | paradox: mean truly-missed (nat) | mirror: mean truly-missed (nat) |",
                  "|---|---|---|"]
        for m in models:
            ft = paired[(paired.model_name == m) & ~paired.bench_correct & paired.nat_correct]
            tf = paired[(paired.model_name == m) & paired.bench_correct & ~paired.nat_correct]
            lines.append(f"| {m} | {ft.n_truly_missed_nat.mean():.2f} | {tf.n_truly_missed_nat.mean():.2f} |")

    lines += ["\n## Takeaway\n",
              "- **Benchmark flails, it does not abstain.** In the paradox cell the "
              "benchmark run issues *more* searches than the natural run yet still gets "
              "the answer wrong — extra search here is unproductive looping, not the "
              "healthy search the cost story credits.\n",
              "- **Natural phrasing leaks bridge entities.** The rewrites place "
              "intermediate gold answers verbatim in the prompt, collapsing reasoning "
              "hops. Less search is then *legitimately* required, so higher natural "
              "accuracy is not evidence that skipping uncertain hops is free.\n",
              "- Implication for the paper: the Rung-1/Rung-2 cost claims should be made "
              "*within matched difficulty* (control for leaked hops), and the cross-"
              "phrasing accuracy gap should be attributed to prompt leakage + benchmark "
              "looping rather than to a benefit of searching less.\n"]
    (outdir / "report.md").write_text("\n".join(lines))
    print(f"Saved {outdir}/report.md")

    # ── discordant_examples.md — qualitative side-by-side ──────────────────────
    dlines = ["# Paradox-cell examples (natural ✓ / benchmark ✗)\n",
              "Sorted by benchmark search count (most flailing first). Bridge entities "
              "the natural rewrite leaked are the intermediate gold answers.\n"]
    ex = paired[~paired.bench_correct & paired.nat_correct].sort_values(
        "n_search_bench", ascending=False)
    for m in models:
        sub = ex[ex.model_name == m].head(n_examples)
        if sub.empty:
            continue
        dlines.append(f"\n## {m}\n")
        for _, r in sub.iterrows():
            dlines += [f"### `{r.example_id}`  ({r.num_hops} hops)",
                       f"- **Benchmark Q** ({r.n_search_bench} searches, leak={r.bridge_leak_bench}): "
                       f"{r.q_bench}",
                       f"- **Natural Q** ({r.n_search_nat} searches, leak={r.bridge_leak_nat}): "
                       f"{r.q_nat}\n"]
    (outdir / "discordant_examples.md").write_text("\n".join(dlines))
    print(f"Saved {outdir}/discordant_examples.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", default="results/musique_parametric/interplay_analysis")
    p.add_argument("--nat-dir", default="results/musique-natural/interplay_analysis")
    p.add_argument("--bench-summary",
                   default="results/musique_parametric/interplay_analysis/interplay_summary.csv")
    p.add_argument("--nat-summary",
                   default="results/musique-natural/interplay_analysis/interplay_summary.csv")
    p.add_argument("--slugs", nargs="+", default=viz.BASE_MODEL_SLUGS)
    p.add_argument("--output-dir", default="results/missed_hop_paradox")
    p.add_argument("--n-examples", type=int, default=25, help="examples per model in the dump")
    args = p.parse_args()

    viz.apply_theme()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    paired = build_paired(args.bench_dir, args.nat_dir, args.slugs)
    if paired.empty:
        print("No paired examples found — check --bench-dir/--nat-dir."); return
    paired = attach_missed_counts(paired, args.bench_summary, args.nat_summary)
    paired.to_csv(outdir / "paired_outcomes.csv", index=False)
    print(f"Saved {outdir}/paired_outcomes.csv  ({len(paired)} paired rows)")

    fig_paired_2x2(paired, outdir)
    fig_paradox_profile(paired, outdir)
    write_reports(paired, outdir, args.n_examples)
    print(f"\nAll paradox artifacts written to {outdir}/")


if __name__ == "__main__":
    main()
