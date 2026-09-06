#!/usr/bin/env python3
"""Combined search+accuracy briefing figure for the gemma-4-31B cue-robustness SFT.

Mirrors make_sft_control_figure.py (gpt-oss). Columns = {baseline gemma4:31b, SFT frames-robust}.
Per panel: grouped bars of Δ Search (green) and Δ Accuracy (blue) vs each model's own PLAIN, per
cue, with paired-significance stars. The plain search LEVEL is annotated so the "SFT is flat but
anchored ~1 call higher than baseline plain" story is visible (both models are Q4_K_M + local-BM25,
so directly comparable — no quant control needed).

  --dataset frames    (default) FRAMES, scored over the 101-question held-out test split.
                      -> results/frames_cue_eval_test_regrade/gemma_cue_robustness.png
  --dataset hotpotqa  the SAME contrast out of domain: the SFT never saw HotpotQA, so all 300
                      questions are held out and no split file is needed. Tests whether the
                      cue-robustness TRANSFERS.
                      -> results/hotpotqa_cue_briefing/gemma_cue_robustness_hotpotqa.png

Dataset-specific handling, all in DATASETS below:
  * HotpotQA has ONE phrasing, so there is no TERSE bar (8 cues, not 9).
  * HotpotQA's run-to-run floor is a `plain_rep2` replicate inside the SAME results dir, not a
    separate _rerun tree. A model with no replicate yet gets an EMPTY PLAIN<->PLAIN bar rather
    than being dropped or silently compared against a zero-difference null.
  * HotpotQA excludes the 14/300 yes/no golds from accuracy (substring matching is meaningless
    on them), matching scripts/grade_hotpotqa_regex.py.
"""
import json, glob, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
import sys; sys.path.insert(0, "scripts")
from regrade_regex import heuristic_match, normalize

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})
RERUN_LABEL = "PLAIN↔PLAIN"
SEARCH_C, ACC_C = "#4daf4a", "#377eb8"

# AgentAsSampler.acall() counts search calls over pydantic-ai's all_messages(), which includes the
# injected message_history as a literal prefix -- so raw sampler_search_calls for the search-multiturn
# condition is inflated by exactly 1 fake search call from the mocked history itself (multiturn's
# chit-chat history has no tool calls, so it's unaffected). Exact correction, not an approximation.
#
# Cue order + grouping matches results/cue_briefing/brief_combined_search_acc_primary.png's FIG 5:
# [PLAIN<->PLAIN ref] | [Style] | [Conversation State] | [Directives]
# Labels match make_aggregate_cue_tradeoff_figure.py's get_label() (the paper's Figure 1) exactly,
# so the same cue reads identically across every figure in the paper.
DATASETS = {
    "frames": dict(
        models=[("results/frames_cues_full/gemma4_31b", "baseline gemma4:31b"),
                ("results/frames_cue_eval_test/gemma4-frames-robust-q4km", "SFT frames-robust")],
        # Independent second verbose_plain run per model, in its own _rerun tree.
        rerun_dirs={"baseline gemma4:31b": "results/frames_cues_rerun/gemma4_31b",
                    "SFT frames-robust": "results/frames_cue_eval_test_rerun/gemma4-frames-robust-q4km"},
        rerun_cond="verbose_plain",
        plain="verbose_plain",
        cues=[("verbose_polite", "POLITE"), ("terse_plain", "TERSE"),
              ("verbose_multiturn", "MULTITURN"), ("verbose_searchmulti", "SEARCH MULTITURN"),
              ("verbose_natural", "SHORT"), ("verbose_elaborate", "ELABORATE"),
              ("verbose_query", "QUERY"), ("verbose_direct", "DIRECT"),
              ("verbose_confident_parametric", "CONFIDENT")],
        offsets={"verbose_searchmulti": 1},
        # Only the held-out split is scored: the SFT trained on the rest of FRAMES.
        id_file="data/sft/frames_gemma4/test_ids.json",
        id_sort_key=int,
        boolean_meta=None,
        out="results/frames_cue_eval_test_regrade/gemma_cue_robustness.png",
        row_label="whole test",
    ),
    "hotpotqa": dict(
        # Both arms live in the same grid dir, one subdir per model.
        models=[("results/hotpotqa_cue_grid/gemma4_31b", "baseline gemma4:31b"),
                ("results/hotpotqa_cue_grid/gemma4-frames-robust-q4km_latest", "SFT frames-robust")],
        # The floor is a `plain_rep2` replicate inside the SAME dir, not a separate tree.
        rerun_dirs={"baseline gemma4:31b": "results/hotpotqa_cue_grid/gemma4_31b",
                    "SFT frames-robust": "results/hotpotqa_cue_grid/gemma4-frames-robust-q4km_latest"},
        rerun_cond="plain_rep2",
        plain="plain",
        # No TERSE: the HotpotQA grid has a single phrasing.
        cues=[("polite", "POLITE"),
              ("multiturn", "MULTITURN"), ("searchmulti", "SEARCH MULTITURN"),
              ("natural", "SHORT"), ("elaborate", "ELABORATE"),
              ("query", "QUERY"), ("direct", "DIRECT"),
              ("confident_parametric", "CONFIDENT")],
        offsets={"searchmulti": 1},
        # No split file: the SFT never saw HotpotQA, so all 300 questions are held out.
        id_file=None,
        id_sort_key=str,
        # The 14/300 yes/no golds are dropped from BOTH metrics: substring matching is
        # meaningless on them ("no" occurs constantly in prose). Same rule as
        # scripts/grade_hotpotqa_regex.py, so these numbers match its tables.
        boolean_meta="data/hotpotqa_300.jsonl",
        out="results/hotpotqa_cue_briefing/gemma_cue_robustness_hotpotqa.png",
        row_label="held-out, out-of-domain",
    ),
}


def load(dirp, cond, offsets):
    fs = glob.glob(f"{dirp}/*_{cond}.json")
    if not fs: return {}
    d = {}
    for r in json.load(open(fs[0])):
        eid = str(r["example_id"]); gold = r.get("correct_answer"); resp = r.get("sampler_response") or ""
        sc = r.get("sampler_search_calls")
        # Prefer the count the runner itself recorded once the source fix landed; fall back to
        # the known per-condition constant for rows collected before it.
        recorded = r.get("history_search_calls")
        offset = offsets.get(cond, 0) if recorded is None else recorded
        # A row with stop_reason set is a salvaged best-effort answer from a run that hit the
        # agent loop cap; excluded, as in grade_hotpotqa_regex.py's own aggregates.
        if r.get("stop_reason"):
            continue
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="frames", choices=sorted(DATASETS))
    args = ap.parse_args()
    CFG = DATASETS[args.dataset]
    MODELS, PLAIN, CUES = CFG["models"], CFG["plain"], CFG["cues"]
    CONDS = [PLAIN] + [c for c, _ in CUES]
    offsets = CFG["offsets"]

    data = {m: {c: load(d, c, offsets) for c in CONDS} for d, m in MODELS}
    rerun = {m: load(CFG["rerun_dirs"][m], CFG["rerun_cond"], offsets) for _, m in MODELS}
    # A model whose replicate has not been collected yet gets an EMPTY PLAIN<->PLAIN bar.
    # It must NOT be intersected into `common` (that would empty the whole comparison), and it
    # must NOT be silently omitted either -- without a floor, that model's cue bars are being
    # tested against a zero-difference null, which is exactly the flaw this reference bar exists
    # to expose. The caption says which models are missing one.
    no_floor = [m for _, m in MODELS if not rerun[m]]
    for m in no_floor:
        print(f"  [warn] no '{CFG['rerun_cond']}' replicate for {m!r} — its "
              f"{RERUN_LABEL} noise-floor bar will be blank")

    common = None
    for _, m in MODELS:
        for c in CONDS:
            common = set(data[m][c]) if common is None else common & set(data[m][c])
        if rerun[m]:
            common &= set(rerun[m])
    if CFG["id_file"]:
        common &= set(str(x) for x in json.load(open(CFG["id_file"])))
    common = sorted(common, key=CFG["id_sort_key"])
    if not common:
        raise SystemExit("no example_ids common to every condition of every model")
    # The yes/no-gold exclusion applies to ACCURACY ONLY, not to search volume: substring
    # grading is meaningless on a "yes"/"no" gold, but the number of searches the agent chose
    # to run on that question is a perfectly valid observation. Keeping two id sets matches how
    # the rest of the HotpotQA figures are built (search over all rows, EM over non-boolean).
    # NOTE this makes the accuracy bars differ slightly from scripts/analyze_hotpotqa_transfer.py,
    # which grades all 300 including the 14 boolean golds.
    acc_ids = common
    if CFG["boolean_meta"]:
        boolean_ids = {str(r["example_id"]) for r in
                       (json.loads(l) for l in open(CFG["boolean_meta"]))
                       if r.get("answer_is_boolean")}
        acc_ids = [i for i in common if i not in boolean_ids]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), constrained_layout=True, sharey=False)
    rlabel = (f"{CFG['row_label']} ({len(common)})" if len(acc_ids) == len(common)
              else f"{CFG['row_label']} ({len(common)} search / {len(acc_ids)} acc)")
    labels = [RERUN_LABEL] + [l for _, l in CUES]
    for ci, (_, m) in enumerate(MODELS):
        ax = axes[ci]
        pl_s = [data[m][PLAIN][i]["s"] for i in common]
        pl_ok = [data[m][PLAIN][i]["c"] for i in acc_ids]
        pl_mean = np.mean(pl_s)
        if rerun[m]:
            rr_s = [rerun[m][i]["s"] for i in common]
            rr_ok = [rerun[m][i]["c"] for i in acc_ids]
            rr_diffs = [a - b for a, b in zip(rr_s, pl_s)]
            sv = [(np.mean(rr_s) - pl_mean) / pl_mean * 100]
            sp = [wilcoxon(rr_diffs).pvalue if any(rr_diffs) else 1.0]
            av = [(np.mean(rr_ok) - np.mean(pl_ok)) * 100]
            ap = [mcnemar_p(pl_ok, rr_ok)]
        else:
            sv, sp, av, ap = [np.nan], [np.nan], [np.nan], [np.nan]
        for cond, _ in CUES:
            cu_s = [data[m][cond][i]["s"] for i in common]
            cu_ok = [data[m][cond][i]["c"] for i in acc_ids]
            s_pct = (np.mean(cu_s) - pl_mean) / pl_mean * 100
            diffs = [a - b for a, b in zip(cu_s, pl_s)]
            sv.append(s_pct); sp.append(wilcoxon(diffs).pvalue if any(diffs) else 1.0)
            av.append((np.mean(cu_ok) - np.mean(pl_ok)) * 100); ap.append(mcnemar_p(pl_ok, cu_ok))
        x = np.arange(len(labels)); w = 0.38
        ax.axvspan(-0.5, 0.5, color="#9e9e9e", alpha=0.15)
        ax.bar(x - w/2, sv, w, color=SEARCH_C, label="Δ Search (%)")
        ax.bar(x + w/2, av, w, color=ACC_C, label="Δ Regex Acc (pp)")
        for xi, sval, aval, spv, apv in zip(x, sv, av, sp, ap):
            if not np.isnan(sval):
                ax.text(xi - w/2, sval + (1.5 if sval >= 0 else -1.5), f"{sval:+.0f}{stars(spv)}",
                        ha="center", va="bottom" if sval >= 0 else "top", fontsize=8.5, color="#123")
            if not np.isnan(aval):
                ax.text(xi + w/2, aval + (1.5 if aval >= 0 else -1.5), f"{aval:+.0f}{stars(apv)}",
                        ha="center", va="bottom" if aval >= 0 else "top", fontsize=8.5, color="#123")
        if m in no_floor:
            ax.text(0, 0, "no\nreplicate", ha="center", va="center", fontsize=7.5,
                    color="#777", style="italic")
        ax.axhline(0, color="#333", lw=0.8); ax.margins(y=0.16)
        # Group dividers: [PLAIN<->PLAIN ref] | [Style] | [Conversation State] | [Directives]
        n_style = sum(1 for c, _ in CUES if c.endswith("polite") or c == "terse_plain")
        ax.axvline(0.5, color="gray", linestyle="--", lw=1.2)
        ax.axvline(0.5 + n_style, color="gray", linestyle="--", lw=1.2)
        ax.axvline(2.5 + n_style, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
        ax.tick_params(labelsize=9.5)
        ax.set_title(f"{m}  (plain = {pl_mean:.2f} calls)", fontsize=11)
        if ci == 0:
            # Two lines: the long HotpotQA n-string ("300 search / 286 acc") overruns the
            # axis height on one line and gets clipped.
            ax.set_ylabel(f"$\\Delta$ vs own plain\n{rlabel}", fontsize=9.5)
    axes[1].legend(loc="upper left", fontsize=9, framealpha=0.9)
    out = CFG["out"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out); print("wrote", out)


if __name__ == "__main__":
    main()
