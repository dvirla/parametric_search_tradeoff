#!/usr/bin/env python3
"""Combined search+accuracy briefing-style figure for the FRAMES cue-robustness SFT + Q4 control.

Mirrors make_cue_briefing_figures.py FIG 5 (brief_combined_search_acc): per-panel grouped bars of
Δ Search (green) and Δ Accuracy (blue) vs each model's own PLAIN across the 6 cue conditions, with
paired-significance stars. Layout: rows = {usable-64, whole-102}, cols = the three quant/training
variants (MXFP4 base, Q4 base, Q4 base+LoRA SFT) so the cue effect visibly collapses left->right.

Contrasts encoded: Q4 base vs MXFP4 base = quantization effect; Q4 SFT vs Q4 base = pure fine-tuning.
"""
import json, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, "scripts")
from regrade_regex import heuristic_match

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})

BASE = "results/frames_cue_eval_test"
MODELS = [  # (dir, panel label)
    ("gpt-oss_20b", "MXFP4 base"),
    ("gpt-oss-vanilla-q4km", "Q4 base"),
    ("gpt-oss-frames-robust-q4km", "Q4 base+LoRA (SFT)"),
]
PLAIN = "verbose_plain"
# Labels match make_aggregate_cue_tradeoff_figure.py's get_label() (the paper's Figure 1) exactly,
# so the same cue reads identically across every figure in the paper.
CUES = [("verbose_polite", "POLITE"), ("verbose_natural", "SHORT"),
        ("verbose_elaborate", "ELABORATE"), ("verbose_query", "QUERY"),
        ("verbose_direct", "DIRECT"), ("terse_plain", "TERSE")]
CONDS = [PLAIN] + [c for c, _ in CUES]
SEARCH_C, ACC_C = "#4daf4a", "#377eb8"


def load(model, cond):
    recs = json.load(open(glob.glob(f"{BASE}/{model}/*_{cond}.json")[0]))
    d = {}
    for r in recs:
        eid = str(r["example_id"]); gold = r.get("correct_answer"); resp = r.get("sampler_response") or ""
        d[eid] = {"search": r.get("sampler_search_calls"),
                  "correct": bool(heuristic_match(gold, resp)) if gold is not None else None}
    return d


def stars(p):
    if p is None or np.isnan(p): return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def mcnemar_p(base_ok, cue_ok):
    """Exact McNemar on paired correctness (discordant pairs)."""
    b = sum(1 for x, y in zip(base_ok, cue_ok) if x and not y)
    c = sum(1 for x, y in zip(base_ok, cue_ok) if y and not x)
    n = b + c
    if n == 0: return 1.0
    return binomtest(min(b, c), n, 0.5).pvalue


def main():
    data = {m: {c: load(m, c) for c in CONDS} for m, _ in MODELS}
    all_ids = set(str(x) for x in json.load(open("data/sft/frames/test_ids.json")))
    usable = set(str(x) for x in json.load(open("data/sft/frames/usable_test_ids.json")))

    rowsets = [("Usable-64 (correct plain ref)", usable), ("Whole set (102)", all_ids)]
    N = len(MODELS)
    fig, axes = plt.subplots(2, N, figsize=(3.3 * N, 6.4), constrained_layout=True, sharey="row")

    for r, (rlabel, idset) in enumerate(rowsets):
        common = set(idset)
        for m, _ in MODELS:
            for c in CONDS: common &= set(data[m][c].keys())
        common = sorted(common, key=int)
        for cidx, (m, mlabel) in enumerate(MODELS):
            ax = axes[r][cidx]
            pl_s = [data[m][PLAIN][i]["search"] for i in common]
            pl_ok = [data[m][PLAIN][i]["correct"] for i in common]
            pl_mean = np.mean(pl_s)
            s_vals, s_ps, a_vals, a_ps, absd = [], [], [], [], []
            for cond, _ in CUES:
                cu_s = [data[m][cond][i]["search"] for i in common]
                cu_ok = [data[m][cond][i]["correct"] for i in common]
                # Δ search as % of plain level; paired Wilcoxon on raw diffs
                s_pct = (np.mean(cu_s) - pl_mean) / pl_mean * 100
                diffs = [a - b for a, b in zip(cu_s, pl_s)]
                sp = wilcoxon(diffs).pvalue if any(diffs) else 1.0
                # Δ accuracy in pp; McNemar
                a_pp = (np.mean(cu_ok) - np.mean(pl_ok)) * 100
                ap = mcnemar_p(pl_ok, cu_ok)
                s_vals.append(s_pct); s_ps.append(sp); a_vals.append(a_pp); a_ps.append(ap)
                absd.append(abs(np.mean(cu_s) - pl_mean))
            x = np.arange(len(CUES)); w = 0.38
            ax.bar(x - w / 2, s_vals, w, color=SEARCH_C, label="Δ Search (%)")
            ax.bar(x + w / 2, a_vals, w, color=ACC_C, label="Δ Regex Acc (pp)")
            for xi, sv, av, sp, ap in zip(x, s_vals, a_vals, s_ps, a_ps):
                ax.text(xi - w / 2, sv + (1.2 if sv >= 0 else -1.2), f"{sv:+.0f}{stars(sp)}",
                        ha="center", va="bottom" if sv >= 0 else "top", fontsize=8, color="#123")
                ax.text(xi + w / 2, av + (1.2 if av >= 0 else -1.2), f"{av:+.0f}{stars(ap)}",
                        ha="center", va="bottom" if av >= 0 else "top", fontsize=8, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.margins(y=0.16)
            ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in CUES], rotation=25, ha="right", fontsize=9)
            ax.tick_params(labelsize=9)
            ax.set_title(mlabel, fontsize=10.5)
            if cidx == 0:
                ax.set_ylabel(f"{rlabel}: $\\Delta$ vs plain", fontsize=9.5)
    axes[0][N - 1].legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    out = "results/frames_cue_eval_test_regrade/brief_combined_sft_control.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
