#!/usr/bin/env python3
"""MedQA cue-robustness TRANSFER figure for the gemma-4-31B FRAMES-SFT.

The gemma-4 SFT was trained ONLY on FRAMES cues; this tests whether the cue-robustness transfers to
MedQA (a different dataset the model never trained on). Mirrors make_gemma_cue_figure.py:
columns = {baseline gemma4:31b, SFT frames-robust}; rows = {usable (baseline plain-correct), whole};
per panel: grouped bars of Δ Search (green) + Δ Accuracy (blue) vs each model's own PLAIN, per cue,
with paired-significance stars. Both use the local MedQA BM25 index (comparable search-call scale).
Conditions mirror FRAMES: orig_plain (ref) + orig_{polite,natural,elaborate,query,direct} + terse_plain.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
import sys; sys.path.insert(0, "scripts")
from regrade_regex import heuristic_match

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})
MODELS = [("results/medqa_grid/gemma4_31b", "baseline gemma4:31b"),
          ("results/medqa_grid/gemma4-frames-robust-q4km_latest", "SFT frames-robust")]
PLAIN = "orig_plain"
# Labels match make_aggregate_cue_tradeoff_figure.py's get_label() (the paper's Figure 1) exactly,
# so the same cue reads identically across every figure in the paper.
CUES = [("orig_polite","POLITE"),("orig_natural","SHORT"),("orig_elaborate","ELABORATE"),
        ("orig_query","QUERY"),("orig_direct","DIRECT"),("terse_plain","TERSE")]
CONDS = [PLAIN] + [c for c,_ in CUES]
SEARCH_C, ACC_C = "#4daf4a", "#377eb8"


def load(dirp, cond):
    fs = glob.glob(f"{dirp}/*_{cond}.json")
    if not fs: return {}
    d = {}
    for r in json.load(open(fs[0])):
        eid = str(r["example_id"]); gold = r.get("correct_answer"); resp = r.get("sampler_response") or ""
        d[eid] = {"s": r.get("sampler_search_calls"),
                  "c": bool(heuristic_match(gold, resp)) if gold is not None else None}
    return d


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def mcnemar_p(a, b):
    x = sum(1 for u, v in zip(a, b) if u and not v); y = sum(1 for u, v in zip(a, b) if v and not u)
    n = x + y
    return 1.0 if n == 0 else binomtest(min(x, y), n, 0.5).pvalue


def main():
    data = {m: {c: load(d, c) for c in CONDS} for d, m in MODELS}
    # ALL common example_ids (no subset: the SFT never trained on MedQA, nothing is held out).
    allc = None
    for _, m in MODELS:
        for c in CONDS:
            s = set(data[m][c]); allc = s if allc is None else (allc & s)
    common = sorted(allc, key=lambda x: int(x) if x.isdigit() else x)
    print(f"whole MedQA set (all questions, no subset): {len(common)}")

    # Search shown in ABSOLUTE calls (not %): the baseline barely searches on MedQA (~0.1 plain), so
    # a %-of-plain axis explodes and misleads. Accuracy stays in pp.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), constrained_layout=True, sharey=True)
    for ci, (_, m) in enumerate(MODELS):
        ax = axes[ci]
        pl_s = [data[m][PLAIN][i]["s"] for i in common]; pl_ok = [data[m][PLAIN][i]["c"] for i in common]
        pl_mean = np.mean(pl_s)
        sv, sp, av, ap, absd = [], [], [], [], []
        for cond, _ in CUES:
            cu_s = [data[m][cond][i]["s"] for i in common]; cu_ok = [data[m][cond][i]["c"] for i in common]
            d_calls = np.mean(cu_s) - pl_mean
            diffs = [a - b for a, b in zip(cu_s, pl_s)]
            sv.append(d_calls); sp.append(wilcoxon(diffs).pvalue if any(diffs) else 1.0)
            av.append((np.mean(cu_ok) - np.mean(pl_ok)) * 100); ap.append(mcnemar_p(pl_ok, cu_ok))
            absd.append(abs(d_calls))
        x = np.arange(len(CUES)); w = 0.38
        ax.bar(x - w/2, sv, w, color=SEARCH_C, label="Δ Search (calls)")
        ax.bar(x + w/2, [a/10 for a in av], w, color=ACC_C, label="Δ Regex Acc (pp/10)")
        for xi, s, a, spv, apv in zip(x, sv, av, sp, ap):
            ax.text(xi - w/2, s + (0.05 if s >= 0 else -0.05), f"{s:+.2f}{stars(spv)}", ha="center",
                    va="bottom" if s >= 0 else "top", fontsize=8.5, color="#123")
            ax.text(xi + w/2, a/10 + (0.05 if a >= 0 else -0.05), f"{a:+.0f}{stars(apv)}", ha="center",
                    va="bottom" if a >= 0 else "top", fontsize=8.5, color="#123")
        ax.axhline(0, color="#333", lw=0.8); ax.margins(y=0.18)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in CUES], rotation=25, ha="right", fontsize=9.5)
        ax.tick_params(labelsize=9.5)
        ax.set_title(m, fontsize=12)
        if ci == 0:
            ax.set_ylabel("$\\Delta$ vs own plain (calls; acc bar = pp/10)", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    out = "results/medqa_regex_regrade/medqa_cue_transfer.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
