"""Phrasing-effect figures recomputed with FAIR grading.

Both formal (benchmark) and natural (paraphrase) sides are read from their
gpt-oss:120b-regraded logs, so accuracy is graded by the *same* judge on both
sides (see memory: the earlier natural-accuracy advantage was a one-sided
grading artifact). Search behaviour is read from the same logs.

Produces, for each of the 3 main models, FULL (n=600 paired) and
LEAK-EXCLUDED (drops the 66 natural rewrites that leak a bridge entity
verbatim) variants of:
  * accuracy: formal vs natural   (Wilson CI bars, McNemar p)
  * search behaviour: mean searches/example (mean±CI95 bars, Wilcoxon p)
  * accuracy by num_hops

Inputs (all already on disk):
  results/musique-formal/musique-formal_baseline_<slug>_run_1_reevaluated.json
  results/musique-natural/musique-natural_baseline_<slug>_run_1_reevaluated.json
  data/musique_val_natural.jsonl   (text->example_id pairing + per-hop golds)

Usage:
  uv run python scripts/analyze_phrasing_corrected.py --output-dir results/phrasing_effect_corrected
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import viz

SLUGS = viz.BASE_MODEL_SLUGS  # gemini, nemotron_30b, qwen3.5_122b
FORMAL_TMPL = "results/musique-formal/musique-formal_baseline_{slug}_run_1_reevaluated.json"
NATURAL_TMPL = "results/musique-natural/musique-natural_baseline_{slug}_run_1_reevaluated.json"


def bridge_leak(question: str, golds: list) -> int:
    ql = (question or "").lower()
    return sum(1 for g in golds[:-1] if g and len(str(g)) >= 2 and str(g).lower() in ql)


def load_pairing(path: str):
    """text -> (example_id, hops); plus the set of verbatim-leaked example_ids."""
    text2meta, hops, leaked = {}, {}, set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            eid = r["example_id"]
            text2meta[r["text"]] = eid
            hops[eid] = r.get("reasoning_hops")
            golds = [s.get("answer") for s in (r.get("sub_questions") or [])]
            if bridge_leak(r["text"], golds) > 0:
                leaked.add(eid)
    return text2meta, hops, leaked


def load_formal(slug):
    d = json.load(open(FORMAL_TMPL.format(slug=slug)))
    return {e["example_id"]: (bool(e["sampler_correct"]), e.get("sampler_search_calls") or 0) for e in d}


def load_natural(slug, text2eid):
    d = json.load(open(NATURAL_TMPL.format(slug=slug)))
    out = {}
    for e in d:
        eid = text2eid.get(e.get("problem"))
        if eid is not None:
            out[eid] = (bool(e["sampler_correct"]), e.get("sampler_search_calls") or 0)
    return out


def mcnemar_p(f, n, ids):
    b = sum(1 for k in ids if f[k][0] and not n[k][0])
    c = sum(1 for k in ids if not f[k][0] and n[k][0])
    return binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0


def wilcoxon_p(f, n, ids):
    fs = [f[k][1] for k in ids]
    ns = [n[k][1] for k in ids]
    if any(a != b for a, b in zip(fs, ns)):
        try:
            return wilcoxon(fs, ns).pvalue
        except ValueError:
            return 1.0
    return 1.0


# ─── figures ────────────────────────────────────────────────────────────────

def _grouped_panel(ax, slugs, formal_vals, formal_err, nat_vals, nat_err, stars, ylabel, title):
    x = np.arange(len(slugs))
    w = 0.38
    ax.bar(x - w / 2, formal_vals, w, yerr=formal_err, capsize=3,
           color=viz.BENCHMARK, label="Formal (benchmark)", error_kw=dict(ecolor=viz.ERR_DARK, lw=1))
    ax.bar(x + w / 2, nat_vals, w, yerr=nat_err, capsize=3,
           color=viz.NATURAL, label="Natural (paraphrase)", error_kw=dict(ecolor=viz.ERR_DARK, lw=1))
    # err arrays are shape (2, N): row 1 is the upper extent above each bar.
    formal_hi = np.asarray(formal_vals) + np.asarray(formal_err)[1]
    nat_hi = np.asarray(nat_vals) + np.asarray(nat_err)[1]
    top = float(max(formal_hi.max(), nat_hi.max()))
    for i, s in enumerate(stars):
        if s:
            y = max(formal_hi[i], nat_hi[i]) + top * 0.03
            ax.text(i, y, s, ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([viz.display_name(s) for s in slugs], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, top * 1.18)


def make_figures(rows, output_dir):
    viz.apply_theme()
    # rows: list of dicts with keys model, scope, metric, formal, formal_lo, formal_hi, natural, ...
    df = pd.DataFrame(rows)

    for metric, ylabel in [("accuracy", "Aggregate accuracy"), ("searches", "Mean searches / example")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, scope, title in [(axes[0], "full", "Full set (n=600)"),
                                  (axes[1], "clean", "Leak-excluded (n=534)")]:
            sub = df[(df.metric == metric) & (df.scope == scope)].set_index("model").loc[SLUGS]
            fv = sub.formal.values
            nv = sub.natural.values
            ferr = [(sub.formal.values - sub.formal_lo.values), (sub.formal_hi.values - sub.formal.values)]
            nerr = [(sub.natural.values - sub.natural_lo.values), (sub.natural_hi.values - sub.natural.values)]
            stars = [viz.sig_stars(p) for p in sub.p.values]
            _grouped_panel(ax, SLUGS, fv, np.array(ferr), nv, np.array(nerr), stars, ylabel, title)
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Phrasing effect — {ylabel} (both sides graded by gpt-oss:120b)", y=1.02)
        fig.tight_layout()
        viz.savefig(fig, output_dir, f"phrasing_{metric}_corrected")
        plt.close(fig)
        print(f"  -> phrasing_{metric}_corrected.png/pdf")


def make_hop_figure(per_hop_rows, output_dir):
    viz.apply_theme()
    df = pd.DataFrame(per_hop_rows)
    df = df[df.scope == "full"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, slug in zip(axes, SLUGS):
        sub = df[df.model == slug].sort_values("hops")
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w / 2, sub.formal, w, color=viz.BENCHMARK, label="Formal")
        ax.bar(x + w / 2, sub.natural, w, color=viz.NATURAL, label="Natural")
        ax.set_xticks(x); ax.set_xticklabels([f"{h}-hop" for h in sub.hops])
        ax.set_title(viz.display_name(slug)); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8)
    fig.suptitle("Accuracy by reasoning depth (both graded by gpt-oss:120b, full set)", y=1.02)
    fig.tight_layout()
    viz.savefig(fig, output_dir, "phrasing_accuracy_by_hop_corrected")
    plt.close(fig)
    print("  -> phrasing_accuracy_by_hop_corrected.png/pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/phrasing_effect_corrected")
    ap.add_argument("--pairing", default="data/musique_val_natural.jsonl")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    text2eid, hops, leaked = load_pairing(args.pairing)
    print(f"Pairing: {len(text2eid)} natural texts, {len(leaked)} verbatim-leaked example_ids excluded in 'clean'.")

    rows, hop_rows = [], []
    for slug in SLUGS:
        f = load_formal(slug)
        n = load_natural(slug, text2eid)
        common_full = sorted(set(f) & set(n))
        scopes = {"full": common_full,
                  "clean": [k for k in common_full if k not in leaked]}
        for scope, ids in scopes.items():
            N = len(ids)
            # accuracy
            kf = sum(f[k][0] for k in ids); kn = sum(n[k][0] for k in ids)
            flo, fhi = viz.wilson_ci(kf, N); nlo, nhi = viz.wilson_ci(kn, N)
            rows.append(dict(model=slug, scope=scope, metric="accuracy",
                             formal=kf / N, formal_lo=flo, formal_hi=fhi,
                             natural=kn / N, natural_lo=nlo, natural_hi=nhi,
                             delta=(kn - kf) / N, n=N, p=mcnemar_p(f, n, ids)))
            # searches
            fs = np.array([f[k][1] for k in ids], float)
            ns = np.array([n[k][1] for k in ids], float)
            fm, fhw = viz.mean_ci95(fs)   # (mean, 95% CI half-width)
            nm, nhw = viz.mean_ci95(ns)
            rows.append(dict(model=slug, scope=scope, metric="searches",
                             formal=fm, formal_lo=fm - fhw, formal_hi=fm + fhw,
                             natural=nm, natural_lo=nm - nhw, natural_hi=nm + nhw,
                             delta=nm - fm, n=N, p=wilcoxon_p(f, n, ids)))
            # by hop (accuracy)
            for h in (2, 3, 4):
                hk = [k for k in ids if hops.get(k) == h]
                if not hk:
                    continue
                hop_rows.append(dict(model=slug, scope=scope, hops=h, n=len(hk),
                                     formal=np.mean([f[k][0] for k in hk]),
                                     natural=np.mean([n[k][0] for k in hk])))

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(args.output_dir, "phrasing_corrected_stats.csv"), index=False)
    pd.DataFrame(hop_rows).to_csv(os.path.join(args.output_dir, "phrasing_corrected_by_hop.csv"), index=False)
    print("  -> phrasing_corrected_stats.csv / phrasing_corrected_by_hop.csv")

    make_figures(rows, args.output_dir)
    make_hop_figure(hop_rows, args.output_dir)

    # console summary
    print("\n=== SUMMARY (formal | natural | delta | McNemar/Wilcoxon p) ===")
    for metric in ("accuracy", "searches"):
        print(f"-- {metric}")
        for scope in ("full", "clean"):
            for _, r in stats_df[(stats_df.metric == metric) & (stats_df.scope == scope)].iterrows():
                print(f"   [{scope:5s}] {viz.display_name(r.model):18s} "
                      f"F={r.formal:6.3f}  N={r.natural:6.3f}  Δ={r.delta:+6.3f}  p={r.p:.4f} {viz.sig_stars(r.p)}")


if __name__ == "__main__":
    main()
