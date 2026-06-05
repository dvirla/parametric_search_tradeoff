"""Unified phrasing-effect comparison (replaces analyze_phrasing_corrected.py
and analyze_phrasing_natural2.py).

Compares any number of MuSiQue phrasings (formal / natural / natural2 / ...) for
any set of models, sourcing all eval results from the SQLite results store via
``results_db.paired_eval`` — no hardcoded filename templates and no per-model
slug remapping (canonicalisation happens in the DB). Adding a new phrasing is a
single ``--phrasings`` value, not a new script.

Reproducing the two old scripts:
  * corrected (formal vs natural, full + leak-excluded):
        --phrasings formal natural --leak-exclude
  * natural2 (formal vs natural vs natural2):
        --phrasings formal natural natural2

For each (model, phrasing, scope) it reports accuracy (Wilson CI) and mean
searches/example (95% CI), with significance vs the reference phrasing
(McNemar for accuracy, Wilcoxon for searches). A per-hop accuracy breakdown and
a tidy long-format stats CSV are also written.

Usage:
  uv run python scripts/analyze_phrasing.py --phrasings formal natural natural2 \
      --output-dir results/phrasing_effect
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import viz
from src import results_db as rdb

# Phrasing -> bar colour. Unlisted phrasings cycle through a fallback palette.
_PHRASING_COLOR = {
    "formal":   viz.BENCHMARK,
    "natural":  viz.NATURAL,
    "natural2": "#2ca02c",
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]
_PHRASING_LABEL = {
    "formal":   "Formal (benchmark)",
    "natural":  "Natural (paraphrase 1)",
    "natural2": "Natural2 (paraphrase 2)",
}


def phrasing_color(ph: str, i: int) -> str:
    return _PHRASING_COLOR.get(ph, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def phrasing_label(ph: str) -> str:
    return _PHRASING_LABEL.get(ph, ph.capitalize())


# ─── Pairing metadata (hops + verbatim-leak flags), DB-independent ──────────────

def _bridge_leak(question: str, golds: list) -> int:
    ql = (question or "").lower()
    return sum(1 for g in golds[:-1] if g and len(str(g)) >= 2 and str(g).lower() in ql)


def load_pairing_meta(path: str):
    """{example_id: hops} and the set of verbatim-leaked example_ids."""
    hops, leaked = {}, set()
    if not path or not os.path.exists(path):
        return hops, leaked
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            eid = r["example_id"]
            hops[eid] = r.get("reasoning_hops")
            golds = [s.get("answer") for s in (r.get("sub_questions") or [])]
            if _bridge_leak(r.get("text", ""), golds) > 0:
                leaked.add(eid)
    return hops, leaked


# ─── Stats helpers ──────────────────────────────────────────────────────────────

def mcnemar_p(ref_correct, oth_correct):
    b = int(((ref_correct == 1) & (oth_correct == 0)).sum())
    c = int(((ref_correct == 0) & (oth_correct == 1)).sum())
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def wilcoxon_p(ref_searches, oth_searches):
    if np.any(ref_searches != oth_searches):
        try:
            return wilcoxon(ref_searches, oth_searches).pvalue
        except ValueError:
            return 1.0
    return 1.0


# ─── Core computation ───────────────────────────────────────────────────────────

def _correct_col(pe, ph):
    return pe[f"{ph}_correct"].apply(
        lambda v: 1 if (v is True or v == 1 or str(v) == "True") else 0).to_numpy()


def _search_col(pe, ph):
    return pd.to_numeric(pe[f"{ph}_searches"], errors="coerce").fillna(0).to_numpy(float)


def compute(conn, base_dataset, phrasings, models, grading, reference,
            hops_map, leaked, do_clean):
    """Return (stats_rows, hop_rows) in tidy long format."""
    stats_rows, hop_rows = [], []
    for slug in models:
        canon = rdb.canonical_model(slug)
        pe = rdb.paired_eval(conn, base_dataset, phrasings, canon, grading=grading)
        if pe.empty:
            print(f"  ! no paired data for {slug} across {phrasings}")
            continue

        scopes = {"full": pe}
        if do_clean and leaked:
            scopes["clean"] = pe[~pe["example_id"].isin(leaked)]

        for scope, sub in scopes.items():
            n = len(sub)
            if n == 0:
                continue
            ref_c = _correct_col(sub, reference) if reference in phrasings else None
            ref_s = _search_col(sub, reference) if reference in phrasings else None
            for ph in phrasings:
                c = _correct_col(sub, ph)
                s = _search_col(sub, ph)
                k = int(c.sum())
                acc, (alo, ahi) = k / n, viz.wilson_ci(k, n)
                m, hw = viz.mean_ci95(s)
                is_ref = (ph == reference)
                p_acc = 1.0 if (is_ref or ref_c is None) else mcnemar_p(ref_c, c)
                p_srch = 1.0 if (is_ref or ref_s is None) else wilcoxon_p(ref_s, s)
                stats_rows.append(dict(
                    model=canon, scope=scope, phrasing=ph, metric="accuracy",
                    val=acc, lo=alo, hi=ahi, n=n, p_vs_ref=p_acc,
                    delta_vs_ref=(acc - (ref_c.sum() / n)) if ref_c is not None else 0.0))
                stats_rows.append(dict(
                    model=canon, scope=scope, phrasing=ph, metric="searches",
                    val=m, lo=m - hw, hi=m + hw, n=n, p_vs_ref=p_srch,
                    delta_vs_ref=(m - ref_s.mean()) if ref_s is not None else 0.0))

            # per-hop accuracy (full scope only, matching the old by-hop figure)
            if scope == "full" and hops_map:
                eids = sub["example_id"].to_numpy()
                for h in (2, 3, 4):
                    mask = np.array([hops_map.get(e) == h for e in eids])
                    if not mask.any():
                        continue
                    for ph in phrasings:
                        hop_rows.append(dict(
                            model=canon, hops=h, phrasing=ph, n=int(mask.sum()),
                            accuracy=float(_correct_col(sub, ph)[mask].mean())))
    return stats_rows, hop_rows


# ─── Figures ────────────────────────────────────────────────────────────────────

def fig_metric(stats_df, metric, ylabel, phrasings, models, reference, scope, output_dir):
    viz.apply_theme()
    sub = stats_df[(stats_df.metric == metric) & (stats_df.scope == scope)]
    if sub.empty:
        return
    x = np.arange(len(models))
    w = min(0.8 / len(phrasings), 0.38)
    offs = np.linspace(-(len(phrasings) - 1) * w / 2, (len(phrasings) - 1) * w / 2, len(phrasings))
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 3, 4.5))
    for i, ph in enumerate(phrasings):
        vals, los, his, ps = [], [], [], []
        for slug in models:
            canon = rdb.canonical_model(slug)
            row = sub[(sub.model == canon) & (sub.phrasing == ph)]
            if row.empty:
                vals.append(np.nan); los.append(0); his.append(0); ps.append(1.0)
            else:
                r = row.iloc[0]
                vals.append(r.val); los.append(r.val - r.lo); his.append(r.hi - r.val)
                ps.append(r.p_vs_ref)
        bars = ax.bar(x + offs[i], vals, w, yerr=[los, his], capsize=3,
                      color=phrasing_color(ph, i), label=phrasing_label(ph),
                      error_kw=dict(ecolor=viz.ERR_DARK, lw=1), zorder=3)
        if ph != reference:
            for bar, v, hi, p in zip(bars, vals, his, ps):
                star = viz.sig_stars(p)
                if star and np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + hi,
                            star, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([viz.display_name(rdb.canonical_model(s)) for s in models],
                       rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    if metric == "accuracy":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(f"{ylabel} — {', '.join(phrasings)} ({scope}, vs {reference})",
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    viz.savefig(fig, output_dir, f"phrasing_{metric}_{scope}")


def fig_by_hop(hop_df, phrasings, models, output_dir):
    if hop_df.empty:
        return
    viz.apply_theme()
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, slug in zip(axes, models):
        canon = rdb.canonical_model(slug)
        sub = hop_df[hop_df.model == canon]
        hop_vals = sorted(sub.hops.unique())
        x = np.arange(len(hop_vals))
        w = min(0.8 / len(phrasings), 0.38)
        offs = np.linspace(-(len(phrasings) - 1) * w / 2, (len(phrasings) - 1) * w / 2, len(phrasings))
        for i, ph in enumerate(phrasings):
            vals = [sub[(sub.hops == h) & (sub.phrasing == ph)].accuracy.mean() for h in hop_vals]
            ax.bar(x + offs[i], vals, w, color=phrasing_color(ph, i), label=phrasing_label(ph))
        ax.set_xticks(x); ax.set_xticklabels([f"{h}-hop" for h in hop_vals])
        ax.set_title(viz.display_name(canon)); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy by reasoning depth (full set)", y=1.02)
    fig.tight_layout()
    viz.savefig(fig, output_dir, "phrasing_accuracy_by_hop")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrasings", nargs="+", default=["formal", "natural", "natural2"])
    ap.add_argument("--models", nargs="+", default=viz.BASE_MODEL_SLUGS)
    ap.add_argument("--base-dataset", default="musique")
    ap.add_argument("--grading", default="reevaluated")
    ap.add_argument("--reference", default="formal",
                    help="Baseline phrasing for significance tests.")
    ap.add_argument("--pairing", default="data/musique_val_natural.jsonl",
                    help="Paraphrase metadata (reasoning_hops + leak detection).")
    ap.add_argument("--leak-exclude", action="store_true",
                    help="Also emit a 'clean' scope dropping verbatim-leaked example_ids.")
    ap.add_argument("--db", default=rdb.DEFAULT_DB_PATH)
    ap.add_argument("--output-dir", default="results/phrasing_effect")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    conn = rdb.connect(args.db)
    hops_map, leaked = load_pairing_meta(args.pairing)
    print(f"Phrasings={args.phrasings}  models={args.models}  grading={args.grading}")
    print(f"Pairing: {len(hops_map)} hops, {len(leaked)} verbatim-leaked ids"
          f"{' (clean scope on)' if args.leak_exclude else ''}")

    stats_rows, hop_rows = compute(
        conn, args.base_dataset, args.phrasings, args.models, args.grading,
        args.reference, hops_map, leaked, args.leak_exclude)
    conn.close()

    stats_df = pd.DataFrame(stats_rows)
    hop_df = pd.DataFrame(hop_rows)
    stats_df.to_csv(os.path.join(args.output_dir, "phrasing_stats.csv"), index=False)
    if not hop_df.empty:
        hop_df.to_csv(os.path.join(args.output_dir, "phrasing_by_hop.csv"), index=False)

    scopes = ["full"] + (["clean"] if args.leak_exclude and leaked else [])
    for scope in scopes:
        fig_metric(stats_df, "accuracy", "Aggregate accuracy", args.phrasings,
                   args.models, args.reference, scope, args.output_dir)
        fig_metric(stats_df, "searches", "Mean searches / example", args.phrasings,
                   args.models, args.reference, scope, args.output_dir)
    fig_by_hop(hop_df, args.phrasings, args.models, args.output_dir)

    # console summary
    print("\n=== SUMMARY (val | vs reference p) ===")
    for metric in ("accuracy", "searches"):
        print(f"-- {metric}")
        for scope in scopes:
            for slug in args.models:
                canon = rdb.canonical_model(slug)
                sub = stats_df[(stats_df.metric == metric) & (stats_df.scope == scope)
                               & (stats_df.model == canon)]
                parts = [f"{r.phrasing}={r.val:.3f}(p={r.p_vs_ref:.3g}{viz.sig_stars(r.p_vs_ref)})"
                         for _, r in sub.iterrows()]
                print(f"   [{scope:5s}] {viz.display_name(canon):18s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
