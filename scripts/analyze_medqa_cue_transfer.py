#!/usr/bin/env python3
"""Scale-free re-analysis of the gemma-4 FRAMES-SFT -> MedQA cue-robustness transfer.

The original (2026-08-04) transfer verdict used absolute Δsearch-calls and #significant cues. Both are
confounded here: the MedQA baseline does zero search on 95.8% of examples at plain, so cue-suppression
is unmeasurable on it, and #sig tracks statistical power rather than effect size at n=500. This script
reports, alongside those numbers:

  * zero-search-at-plain share       — the floor check that invalidates the baseline/SFT comparison
  * mean|Δ| as % of each arm's plain — relative magnitude
  * mean |matched-pairs rank-biserial r| — scale-free effect size (verified uncorrelated with a
    model's plain search level across the non-SFT MedQA arms, unlike absolute Δcalls)
  * a plain<->plain rerun noise floor (results/medqa_grid_rerun)
  * the same metrics for every MedQA-grid arm, and for the SFT on its in-domain FRAMES test ids

Written up in docs/frames_cue_robustness_sft.md ("TRANSFER ... REVISED 2026-09-02").
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon

sys.path.insert(0, "scripts")
from regrade_regex import heuristic_match

MEDQA_PLAIN = "orig_plain"
MEDQA_CUES = [("orig_polite", "POLITE"), ("orig_natural", "SHORT"), ("orig_elaborate", "ELABORATE"),
              ("orig_query", "QUERY"), ("orig_direct", "DIRECT"), ("terse_plain", "TERSE")]
FRAMES_PLAIN = "verbose_plain"
FRAMES_CUES = [("verbose_polite", "POLITE"), ("verbose_natural", "SHORT"),
               ("verbose_elaborate", "ELABORATE"), ("verbose_query", "QUERY"),
               ("verbose_direct", "DIRECT"), ("terse_plain", "TERSE")]
BASE = "results/medqa_grid/gemma4_31b"
SFT = "results/medqa_grid/gemma4-frames-robust-q4km_latest"


def load(dirp: str, cond: str) -> dict:
    """{example_id: {"s": search_calls, "c": regex-correct}} for one condition file."""
    fs = [f for f in glob.glob(f"{dirp}/*_{cond}.json") if not f.endswith("backup.json")]
    if not fs:
        return {}
    out = {}
    for r in json.load(open(fs[0])):
        s = r.get("sampler_search_calls")
        if s in (None, "None"):
            continue  # crashed row: no search count, drop the pair
        out[str(r["example_id"])] = {
            "s": int(s),
            "c": bool(heuristic_match(r.get("correct_answer"), r.get("sampler_response") or "")),
        }
    return out


def rank_biserial(diffs) -> float:
    """Matched-pairs rank-biserial r for the Wilcoxon signed-rank test: (ΣR+ − ΣR−)/ΣR.

    Unit-free and in [-1, 1]; ties (unchanged examples) are dropped, so it measures how
    *consistently directional* the cue's effect is among the examples it moves at all.
    """
    nz = np.array([d for d in diffs if d != 0], float)
    if not len(nz):
        return 0.0
    R = rankdata(np.abs(nz))
    return (R[nz > 0].sum() - R[nz < 0].sum()) / R.sum()


def stars(p): return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "  "


def mcnemar_p(a, b):
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    return 1.0 if x + y == 0 else binomtest(min(x, y), x + y, 0.5).pvalue


def analyze(dirp, name, plain=MEDQA_PLAIN, cues=MEDQA_CUES, common=None):
    conds = [plain] + [c for c, _ in cues]
    data = {c: load(dirp, c) for c in conds}
    missing = [c for c in conds if not data[c]]
    if missing:
        return None
    ids = set(data[plain]) if common is None else set(common)
    for c in conds:
        ids &= set(data[c])
    ids = sorted(ids)
    pl = np.array([data[plain][i]["s"] for i in ids], float)
    pl_ok = [data[plain][i]["c"] for i in ids]
    rows = []
    for cond, lab in cues:
        cu = np.array([data[cond][i]["s"] for i in ids], float)
        cu_ok = [data[cond][i]["c"] for i in ids]
        d = cu - pl
        rows.append(dict(
            cue=lab, d=d.mean(), pct=100 * d.mean() / pl.mean() if pl.mean() else np.nan,
            p=wilcoxon(d).pvalue if np.any(d) else 1.0, r=abs(rank_biserial(d)),
            dacc=100 * (np.mean(cu_ok) - np.mean(pl_ok)), pacc=mcnemar_p(pl_ok, cu_ok)))
    return dict(name=name, n=len(ids), ids=set(ids), plain=pl.mean(), acc=np.mean(pl_ok),
                zero=np.mean(pl == 0), rows=rows,
                mean_d=np.mean([abs(r["d"]) for r in rows]),
                mean_pct=np.mean([abs(r["pct"]) for r in rows]),
                mean_r=np.mean([r["r"] for r in rows]),
                nsig=sum(1 for r in rows if r["p"] < .05))


def report(a):
    print(f"\n### {a['name']}   n={a['n']}  plain={a['plain']:.2f} calls  "
          f"zero-search@plain={100*a['zero']:.1f}%  regex-acc={a['acc']:.3f}")
    print(f"{'cue':<11}{'Dcalls':>9}{'%plain':>9}{'|r|':>7}{'p':>11}   {'Dacc pp':>8}{'p':>9}")
    for r in a["rows"]:
        print(f"{r['cue']:<11}{r['d']:>+9.2f}{r['pct']:>+8.1f}%{r['r']:>7.3f}{r['p']:>11.2e}{stars(r['p'])}"
              f"{r['dacc']:>+8.1f}{r['pacc']:>9.3f}{stars(r['pacc'])}")
    print(f"{'MEAN':<11}{a['mean_d']:>9.2f}{a['mean_pct']:>8.1f}%{a['mean_r']:>7.3f}"
          f"   #sig={a['nsig']}/{len(a['rows'])}")


def main():
    print("=" * 92)
    print("PART 1 — MedQA transfer, on the example ids both arms completed")
    print("=" * 92)
    b0, s0 = analyze(BASE, "baseline"), analyze(SFT, "SFT")
    common = b0["ids"] & s0["ids"]
    base = analyze(BASE, "MedQA baseline gemma4:31b", common=common)
    sft = analyze(SFT, "MedQA SFT frames-robust (7-cond)", common=common)
    report(base); report(sft)

    print("\n" + "=" * 92)
    print("PART 2 — noise floor: baseline plain vs baseline plain RERUN (identical prompt)")
    print("=" * 92)
    p1, p2 = load(BASE, MEDQA_PLAIN), load("results/medqa_grid_rerun/gemma4_31b", MEDQA_PLAIN)
    cc = sorted(set(p1) & set(p2) & common)
    if cc:
        a1 = np.array([p1[i]["s"] for i in cc], float); a2 = np.array([p2[i]["s"] for i in cc], float)
        d = a2 - a1
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        o1 = [p1[i]["c"] for i in cc]; o2 = [p2[i]["c"] for i in cc]
        print(f"n={len(cc)}  run1={a1.mean():.3f}  run2={a2.mean():.3f}  D={d.mean():+.3f} calls "
              f"({100*d.mean()/a1.mean():+.1f}% of plain)  wilcoxon p={p:.2e}{stars(p)}")
        print(f"  acc {np.mean(o1):.3f} -> {np.mean(o2):.3f}  mcnemar p={mcnemar_p(o1,o2):.3f}")
        print(f"  => rerun noise {abs(d.mean()):.3f} calls vs baseline mean|Dcue| {base['mean_d']:.3f} "
              f"calls: the baseline's cue effects are real, its FLATNESS is a floor artifact.")
    else:
        print("no plain rerun available")
    print("  NOTE: no plain<->plain rerun exists for the SFT, so it has no measured noise floor.")

    print("\n" + "=" * 92)
    print("PART 3 — every MedQA-grid arm: is cue-sensitivity just a function of search level?")
    print("=" * 92)
    print(f"{'arm':<32}{'n':>5}{'plain':>7}{'zero@pl':>9}{'mean|D|':>9}{'|D|/pl':>8}{'mean|r|':>9}{'#sig':>6}{'acc':>7}")
    arms = []
    for dirp in sorted(glob.glob("results/medqa_grid/*/")):
        nm = os.path.basename(dirp.rstrip("/"))
        a = analyze(dirp, nm)
        if not a:
            continue
        tag = " <-SFT" if "robust" in nm else (" <-base" if nm == "gemma4_31b" else "")
        print(f"{nm:<32}{a['n']:>5}{a['plain']:>7.2f}{100*a['zero']:>8.1f}%{a['mean_d']:>9.3f}"
              f"{a['mean_pct']:>7.1f}%{a['mean_r']:>9.3f}{a['nsig']:>6}{a['acc']:>7.3f}{tag}")
        arms.append(a)
    others = [a for a in arms if "robust" not in a["name"] and a["n"] >= 400]
    pl = np.array([a["plain"] for a in others])
    print(f"\nAcross the {len(others)} non-SFT arms with n>=400:")
    for lab, v in (("mean|Dcalls|", [a["mean_d"] for a in others]), ("mean|r|", [a["mean_r"] for a in others])):
        rho = spearmanr(pl, v)
        print(f"  Spearman(plain search level, {lab:<12}) = {rho.statistic:+.3f}  (p={rho.pvalue:.3f})")
    print("  => mean|r| is scale-free; absolute Dcalls is a proxy for search level. Compare on |r|.")
    searchy = sorted([a for a in arms if a["zero"] < .15], key=lambda a: -a["plain"])
    print("\nArms where suppression is measurable at all (zero-search@plain < 15%):")
    for a in searchy:
        print(f"  {a['name']:<32} plain={a['plain']:>5.2f}  |D|/plain={a['mean_pct']:>5.1f}%  mean|r|={a['mean_r']:.3f}")

    print("\n" + "=" * 92)
    print("PART 4 — the SFT against itself: out-of-domain MedQA vs in-domain FRAMES test ids")
    print("=" * 92)
    FB, FS = "results/frames_cues_full/gemma4_31b", "results/frames_cue_eval_test/gemma4-frames-robust-q4km"
    fb0 = analyze(FB, "x", FRAMES_PLAIN, FRAMES_CUES)
    fs0 = analyze(FS, "x", FRAMES_PLAIN, FRAMES_CUES)
    if fb0 and fs0:
        fcom = fb0["ids"] & fs0["ids"]
        report(analyze(FB, "FRAMES baseline gemma4:31b (matched test ids)", FRAMES_PLAIN, FRAMES_CUES, fcom))
        report(analyze(FS, "FRAMES SFT frames-robust (matched test ids)", FRAMES_PLAIN, FRAMES_CUES, fcom))
    else:
        print("FRAMES dirs incomplete; skipped")

    print("\n" + "=" * 92)
    print("PART 5 — does the transferred search buy anything on MedQA?")
    print("=" * 92)
    sp = load(SFT, MEDQA_PLAIN)
    ids = sorted(common)
    ss = np.array([sp[i]["s"] for i in ids]); sc = np.array([sp[i]["c"] for i in ids])
    print(f"plain accuracy: baseline {base['acc']:.3f} @ {base['plain']:.2f} calls  vs  "
          f"SFT {sft['acc']:.3f} @ {sft['plain']:.2f} calls")
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 999)):
        m = (ss >= lo) & (ss <= hi)
        if m.sum():
            print(f"  SFT acc | {lo}-{hi} searches: {sc[m].mean():.3f}  (n={m.sum()})")
    print("  (confounded by question difficulty — read as 'no positive signal', not as causal harm)")


if __name__ == "__main__":
    main()
