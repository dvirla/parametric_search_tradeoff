#!/usr/bin/env python3
"""Combined search+accuracy briefing figure for the gemma-4-31B cue-robustness SFT.

Mirrors make_sft_control_figure.py (gpt-oss). Columns = {baseline gemma4:31b, SFT frames-robust},
scored over the whole 101-question held-out test split. Per panel: grouped bars of Δ Search (green)
and Δ Accuracy (blue) vs each model's own PLAIN, per cue, with paired-significance stars. The plain search LEVEL is annotated
so the "SFT is flat but anchored ~1 call higher than baseline plain" story is visible (both models are
Q4_K_M + local-BM25, so directly comparable — no quant control needed).
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
MODELS = [("results/frames_cues_full/gemma4_31b", "baseline gemma4:31b"),
          ("results/frames_cue_eval_test/gemma4-frames-robust-q4km", "SFT frames-robust")]
PLAIN = "verbose_plain"
# Order + grouping matches results/cue_briefing/brief_combined_search_acc_primary.png's FIG 5:
# [POLITE, TERSE-PLAIN] | [MULTITURN, SEARCHMULTI] | [SHORT, ELABORATE, QUERY, DIRECT]
# (no PLAIN<->PLAIN reference bar or NO-SEARCH-NEEDED here -- no rerun/confident_parametric data
# for this model/split -- so only 2 of that figure's 3 group-dividers apply.)
CUES = [("verbose_polite","POLITE"),("terse_plain","TERSE (PLAIN)"),
        ("verbose_multiturn","MULTITURN"),("verbose_searchmulti","SEARCHMULTI"),
        ("verbose_natural","SHORT"),("verbose_elaborate","ELABORATE"),
        ("verbose_query","QUERY"),("verbose_direct","DIRECT")]
CONDS = [PLAIN] + [c for c,_ in CUES]
SEARCH_C, ACC_C = "#4daf4a", "#377eb8"
# AgentAsSampler.acall() counts search calls over pydantic-ai's all_messages(), which includes the
# injected message_history as a literal prefix -- so raw sampler_search_calls for verbose_searchmulti
# is inflated by exactly 1 fake search call from the mocked history itself (multiturn's chit-chat
# history has no tool calls, so it's unaffected). Exact correction, not an approximation.
SEARCHMULTI_OFFSET = {"verbose_searchmulti": 1}


def load(dirp, cond):
    fs = glob.glob(f"{dirp}/*_{cond}.json")
    if not fs: return {}
    offset = SEARCHMULTI_OFFSET.get(cond, 0)
    d = {}
    for r in json.load(open(fs[0])):
        eid = str(r["example_id"]); gold = r.get("correct_answer"); resp = r.get("sampler_response") or ""
        sc = r.get("sampler_search_calls")
        d[eid] = {"s": max(0, sc - offset) if sc is not None else None,
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
    all_ids = set(str(x) for x in json.load(open("data/sft/frames_gemma4/test_ids.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True, sharey=False)
    rlabel = "Whole test (101)"
    common = set(all_ids)
    for _, m in MODELS:
        for c in CONDS: common &= set(data[m][c])
    common = sorted(common, key=int)
    for ci, (_, m) in enumerate(MODELS):
        ax = axes[ci]
        pl_s = [data[m][PLAIN][i]["s"] for i in common]; pl_ok = [data[m][PLAIN][i]["c"] for i in common]
        pl_mean = np.mean(pl_s)
        sv, sp, av, ap, absd = [], [], [], [], []
        for cond, _ in CUES:
            cu_s = [data[m][cond][i]["s"] for i in common]; cu_ok = [data[m][cond][i]["c"] for i in common]
            s_pct = (np.mean(cu_s) - pl_mean) / pl_mean * 100
            diffs = [a - b for a, b in zip(cu_s, pl_s)]
            sv.append(s_pct); sp.append(wilcoxon(diffs).pvalue if any(diffs) else 1.0)
            av.append((np.mean(cu_ok) - np.mean(pl_ok)) * 100); ap.append(mcnemar_p(pl_ok, cu_ok))
            absd.append(abs(np.mean(cu_s) - pl_mean))
        x = np.arange(len(CUES)); w = 0.38
        ax.bar(x - w/2, sv, w, color=SEARCH_C, label="Δ Search (%)")
        ax.bar(x + w/2, av, w, color=ACC_C, label="Δ Regex Acc (pp)")
        for xi, s, a, spv, apv in zip(x, sv, av, sp, ap):
            ax.text(xi - w/2, s + (1.5 if s >= 0 else -1.5), f"{s:+.0f}{stars(spv)}", ha="center",
                    va="bottom" if s >= 0 else "top", fontsize=7, color="#123")
            ax.text(xi + w/2, a + (1.5 if a >= 0 else -1.5), f"{a:+.0f}{stars(apv)}", ha="center",
                    va="bottom" if a >= 0 else "top", fontsize=7, color="#123")
        ax.axhline(0, color="#333", lw=0.8); ax.margins(y=0.16)
        # Group dividers matching brief_combined_search_acc_primary.png's FIG 5:
        # [POLITE, TERSE-PLAIN] | [MULTITURN, SEARCHMULTI] | [SHORT, ELABORATE, QUERY, DIRECT]
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        ax.axvline(3.5, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in CUES], rotation=25, ha="right", fontsize=8.5)
        nsig = sum(1 for p in sp if p < 0.05)
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.text(0.97, 0.035, f"mean|Δsearch| = {np.mean(absd):.2f}\n{nsig}/{len(CUES)} cues significant\nplain = {pl_mean:.1f} calls",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color="#333",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=0.9))
        if ci == 0:
            ax.set_ylabel(f"{rlabel}\n\nΔ vs own PLAIN", fontsize=9.5)
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.suptitle("gemma-4-31B cue-robustness SFT — search & accuracy shift under cues (vs own plain)\n"
                 "both Q4_K_M + local BM25 (directly comparable). Note the 'plain' annotation: SFT is flatter but anchored higher.",
                 fontsize=11.5, fontweight="bold")
    out = "results/frames_cue_eval_test_regrade/gemma_cue_robustness.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
