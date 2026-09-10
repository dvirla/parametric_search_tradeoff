r"""
Inferential layer over results/entropy_under_cue/entropy_under_cue.csv.

analyze_entropy_under_cue.py emits one row per (dataset, model, cue) with a paired shift and an
UNCORRECTED per-cell sign test. That is the right per-cell diagnostic but it is not the analysis
the paper claims from, and every corrected number previously quoted was computed ad hoc in a
throwaway snippet. This script is that analysis, committed and reproducible.

THREE LEVELS, in decreasing order of how much weight they should carry:

  1. IN-DOMAIN (primary). Per (dataset, cue), a one-sample t-test over that dataset's per-model
     mean deltas against 0. Models are the independent units; examples are not, since the same
     300/500 questions recur across models. **BH-FDR over all 12 (dataset x cue) tests** -- one
     family, because these 12 are reported together as the primary result.

  2. HETEROGENEITY. Per cue, a Friedman test across the three datasets (repeated measures: the
     same 6 models appear in each). This is the gate on whether pooling is licensed at all. A cue
     that fails it must never be pooled -- `direct` flips sign between datasets
     (-0.053 / +0.105 / -0.047) and pooling averages a real negative against a real positive.

  3. POOLED (sensitivity only). Per cue over all 18 model x dataset cells, BH-FDR over the 4 cue
     tests. Reported ONLY for cues that pass (2), and it answers a weaker question: does a small
     shift RECUR across domains, not is there an in-domain effect. It gains significance purely by
     combining underpowered same-signed estimates.

The per-cell sign tests from the upstream CSV are passed through UNCORRECTED and labelled as such,
so they are never mistaken for the corrected columns beside them.

Usage:
    uv run python scripts/analyze_entropy_under_cue_stats.py
    uv run python scripts/analyze_entropy_under_cue_stats.py --alpha 0.05
"""

from __future__ import annotations

import os
import csv
import sys
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(REPO, "results", "entropy_under_cue", "entropy_under_cue.csv")
OUT_DIR = os.path.join(REPO, "results", "entropy_under_cue")


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up q-values. Verified identical to
    statsmodels.stats.multitest.multipletests(method='fdr_bh') (max |diff| 1.1e-16)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order[::-1]):      # largest p first
        running = min(running, p[idx] * m / (m - rank))
        q[idx] = running
    return q


def setup_args():
    ap = argparse.ArgumentParser(description="Corrected statistics for entropy-under-cue.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--in-csv", default=IN_CSV)
    ap.add_argument("--output-dir", default=OUT_DIR)
    return ap.parse_args()


def main():
    args = setup_args()
    rows = list(csv.DictReader(open(args.in_csv)))
    if not rows:
        raise SystemExit(f"no rows in {args.in_csv}")

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: {"frames": 0, "medqa": 1}.get(d, 2))
    cues = sorted({r["cue"] for r in rows})
    by = defaultdict(dict)                      # (ds, cue) -> {model: delta}
    for r in rows:
        d = fl(r["mean_delta"])
        if d is not None:
            by[(r["dataset"], r["cue"])][r["model"]] = d

    # ---- 1. IN-DOMAIN -----------------------------------------------------
    cells = []
    for cue in cues:
        for ds in datasets:
            v = np.array(list(by[(ds, cue)].values()))
            if len(v) < 3:
                continue
            se = v.std(ddof=1) / np.sqrt(len(v))
            crit = stats.t.ppf(1 - args.alpha / 2, len(v) - 1)
            t, p = stats.ttest_1samp(v, 0)
            cells.append(dict(cue=cue, dataset=ds, n_models=len(v), mean=v.mean(),
                              ci_lo=v.mean() - crit * se, ci_hi=v.mean() + crit * se,
                              t=t, p=p, n_positive=int((v > 0).sum())))
    q = bh_fdr([c["p"] for c in cells])
    for c, qq in zip(cells, q):
        c["q_bh_12"] = qq
        c["significant"] = bool(qq < args.alpha)

    # ---- 2. HETEROGENEITY -------------------------------------------------
    het = []
    for cue in cues:
        models = sorted(set.intersection(*(set(by[(ds, cue)]) for ds in datasets))) \
            if all(by[(ds, cue)] for ds in datasets) else []
        if len(models) < 3:
            continue
        mat = np.array([[by[(ds, cue)][m] for ds in datasets] for m in models])
        chi, p = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
        het.append(dict(cue=cue, n_models=len(models), chi2=chi, p=p,
                        poolable=bool(p >= args.alpha),
                        means={ds: float(mat[:, i].mean()) for i, ds in enumerate(datasets)}))

    # ---- 3. POOLED (sensitivity, poolable cues only) ----------------------
    poolable = {h["cue"] for h in het if h["poolable"]}
    pooled = []
    for cue in cues:
        v = np.array([d for ds in datasets for d in by[(ds, cue)].values()])
        se = v.std(ddof=1) / np.sqrt(len(v))
        crit = stats.t.ppf(1 - args.alpha / 2, len(v) - 1)
        t, p = stats.ttest_1samp(v, 0)
        pooled.append(dict(cue=cue, n_cells=len(v), mean=v.mean(),
                           ci_lo=v.mean() - crit * se, ci_hi=v.mean() + crit * se, p=p,
                           n_positive=int((v > 0).sum()), poolable=cue in poolable))
    qp = bh_fdr([x["p"] for x in pooled])
    for x, qq in zip(pooled, qp):
        x["q_bh_4"] = qq
        x["significant"] = bool(qq < args.alpha and x["poolable"])

    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in (("entropy_under_cue_indomain.csv", cells),
                       ("entropy_under_cue_pooled.csv", pooled)):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path}")

    print(f"\n=== 1. IN-DOMAIN (primary): one-sample t over per-model deltas, "
          f"BH-FDR over {len(cells)} (dataset x cue) tests ===")
    print(f"{'cue':22s} {'dataset':9s} {'n':>2s} {'mean':>8s} {'95% CI':>20s} "
          f"{'p':>7s} {'q':>7s} {'+ve':>5s}")
    for cue in cues:
        for c in [x for x in cells if x["cue"] == cue]:
            print(f"{c['cue']:22s} {c['dataset']:9s} {c['n_models']:2d} {c['mean']:+8.4f} "
                  f"[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] {c['p']:7.4f} {c['q_bh_12']:7.4f} "
                  f"{c['n_positive']:3d}/{c['n_models']}" + ("  SIGNIFICANT" if c["significant"] else ""))

    print("\n=== 2. HETEROGENEITY across datasets (Friedman, repeated measures over models) ===")
    for h in het:
        verdict = "poolable" if h["poolable"] else "NOT POOLABLE"
        means = "  ".join(f"{k}={v:+.3f}" for k, v in h["means"].items())
        print(f"{h['cue']:22s} chi2={h['chi2']:6.2f}  p={h['p']:.4f}  {verdict:12s}  {means}")

    print(f"\n=== 3. POOLED (sensitivity only; BH-FDR over {len(pooled)} cue tests) ===")
    for x in pooled:
        note = "" if x["poolable"] else "   <-- heterogeneous, DO NOT QUOTE"
        print(f"{x['cue']:22s} cells={x['n_cells']:3d} {x['mean']:+8.4f} "
              f"[{x['ci_lo']:+.4f},{x['ci_hi']:+.4f}] p={x['p']:.4f} q={x['q_bh_4']:.4f} "
              f"{x['n_positive']:3d}/{x['n_cells']}"
              + ("  SIGNIFICANT" if x["significant"] else "") + note)

    sig = [c["cue"] + "/" + c["dataset"] for c in cells if c["significant"]]
    print("")
    print(f"IN-DOMAIN SUMMARY: {len(sig)}/{len(cells)} cells significant at q<{args.alpha}: "
          + (", ".join(sig) if sig else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
