#!/usr/bin/env python3
"""Resolved-SFT arm, reported in THE PAPER'S metric.

`analyze_resolved_sft.py` reports %Δ search calls. The paper reports in-domain SFT results as
**mean |Δ| search calls vs. the model's own plain, plus the count of perturbations significant**
(e.g. "1.30 (6 of 9) -> 0.87 (2 of 9)"), with a paired Wilcoxon per perturbation -- see
docs/frames_cue_robustness_sft.md and make_gemma_cue_figure.py. This script emits that metric for
the resolved checkpoint on both datasets so §sec:sft_interventions can be rewritten without
changing the unit it reports in. %Δ is printed alongside because the paper's HotpotQA transfer
paragraph already uses it.

    uv run python scripts/report_resolved_sft_paper_metric.py

Searchmulti offset: decided PER FILE, never per condition -- the two arms differ within the same
dataset (baseline pre-fix, resolved post-fix). See grade_hotpotqa_regex.py: file_is_legacy.
"""
import glob
import json
import os
import statistics
import sys

from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.regrade_regex import heuristic_match, strip_reasoning_channel  # noqa: E402

SFT_SLUG = "gemma4-frames-resolved-q4km"
BASE_SLUG = "gemma4_31b"
TEST_IDS = "data/sft/frames_gemma4_resolved/test_ids.json"

FRAMES_PLAIN = "verbose_plain"
FRAMES_SEEN = ["verbose_polite", "terse_plain", "verbose_natural",
               "verbose_elaborate", "verbose_query", "verbose_direct"]
FRAMES_UNSEEN = ["verbose_confident_parametric", "verbose_multiturn", "verbose_searchmulti"]
HPQ_SEEN = ["natural", "elaborate", "polite", "direct", "query"]
HPQ_UNSEEN = ["confident_parametric", "multiturn", "searchmulti"]


def calls_corrected(rows, cond, key="sampler_search_calls"):
    """Per-file mocked-history correction (searchmulti only)."""
    out = {str(r["example_id"]): (r.get(key) or 0) for r in rows}
    if "searchmulti" not in cond:
        return out
    vals = list(out.values())
    legacy = bool(vals) and min(vals) >= 1          # pre-fix signature
    return {e: (v - 1 if legacy else v) for e, v in out.items()}


def frames_arm(pattern, sep):
    test = set(map(str, json.load(open(TEST_IDS))))
    out = {}
    for fp in glob.glob(pattern):
        cond = os.path.basename(fp).split(sep, 1)[-1].replace(".json", "")
        rows = [r for r in json.load(open(fp)) if str(r["example_id"]) in test]
        if rows:
            out[cond] = rows
    return out


def frames_acc(rows, ids):
    by = {str(r["example_id"]): r for r in rows}
    return 100.0 * statistics.mean(
        [heuristic_match(by[i]["correct_answer"],
                         strip_reasoning_channel(by[i].get("sampler_response") or "")) for i in ids])


def report(label, plain_calls, plain_acc, conds, seen, unseen):
    """conds: {cond: (calls_dict, acc_or_None)}; plain_calls: dict."""
    print(f"\n### {label}")
    pm = statistics.mean(plain_calls.values())
    zero = 100.0 * statistics.mean([v == 0 for v in plain_calls.values()])
    print(f"    plain = {pm:.2f} calls   zero-search = {zero:.1f}%   acc = {plain_acc:.1f}%   n = {len(plain_calls)}")
    print(f"    {'perturbation':<30}{'Δcalls':>9}{'%plain':>9}{'p':>10}  sig  group")
    stats_by = {}
    for cond, (cc, _) in conds.items():
        ids = sorted(set(plain_calls) & set(cc))
        d = [cc[i] - plain_calls[i] for i in ids]
        p = wilcoxon(d).pvalue if any(d) else 1.0
        dc = statistics.mean(d)
        grp = "seen" if cond in seen else ("UNSEEN" if cond in unseen else "-")
        stats_by[cond] = (dc, p, grp)
        print(f"    {cond:<30}{dc:>+9.2f}{100.0*dc/pm:>+8.1f}%{p:>10.2e}  "
              f"{'*' if p < 0.05 else ' ':<4} {grp}")
    for name, grp in (("ALL", None), ("SEEN", "seen"), ("UNSEEN", "UNSEEN")):
        sub = {c: v for c, v in stats_by.items() if grp is None or v[2] == grp}
        if not sub:
            continue
        m = statistics.mean(abs(v[0]) for v in sub.values())
        mp = statistics.mean(100.0 * abs(v[0]) / pm for v in sub.values())
        nsig = sum(1 for v in sub.values() if v[1] < 0.05)
        print(f"      {name:<7} mean|Δ| = {m:.2f} calls ({mp:.1f}%)   significant: {nsig} of {len(sub)}")
    return stats_by


def main():
    print("=" * 96)
    print("RESOLVED SFT vs BASELINE -- the paper's metric: mean |Δ| search calls + # significant")
    print("=" * 96)

    # ---------------- FRAMES ----------------
    sft = frames_arm(f"results/frames_cue_eval_resolved/*/frames-cues_baseline_{SFT_SLUG}_*.json", "-q4km_")
    base = frames_arm(f"results/frames_cues_full/{BASE_SLUG}/frames-cues_baseline_gemma4:31b_*.json", "gemma4:31b_")
    for label, arm in (("FRAMES -- SFT resolved", sft), ("FRAMES -- baseline gemma4:31b", base)):
        pc = calls_corrected(arm[FRAMES_PLAIN], FRAMES_PLAIN)
        conds = {c: (calls_corrected(arm[c], c), None)
                 for c in FRAMES_SEEN + FRAMES_UNSEEN if c in arm}
        report(label, pc, frames_acc(arm[FRAMES_PLAIN], sorted(pc)), conds, FRAMES_SEEN, FRAMES_UNSEEN)

    # ---------------- HotpotQA (graded per_row already carries the correction) ----------------
    import pandas as pd
    df = pd.read_csv("results/hotpotqa_cue_grid_regex/per_row.csv")
    for label, slug in (("HOTPOTQA -- SFT resolved", SFT_SLUG), ("HOTPOTQA -- baseline gemma4:31b", BASE_SLUG)):
        d = df[df.model == slug]
        g = {c: gg for c, gg in d.groupby("run_name")}
        pc = dict(zip(g["plain"].example_id.astype(str), g["plain"].search_calls))
        pa = 100.0 * g["plain"]["strict"].mean()
        conds = {c: (dict(zip(g[c].example_id.astype(str), g[c].search_calls)), None)
                 for c in HPQ_SEEN + HPQ_UNSEEN if c in g}
        report(label, pc, pa, conds, HPQ_SEEN, HPQ_UNSEEN)


def floors():
    """Run-to-run plain<->plain floor, in the same unit as the table above (calls)."""
    print("\n" + "=" * 96)
    print("FLOOR -- plain vs an independent plain replicate, in CALLS (the paper's unit)")
    print("=" * 96)
    test = set(map(str, json.load(open(TEST_IDS))))

    def one(tag, d1, d2, c1, c2, restrict):
        a = glob.glob(os.path.join(d1, f"*_{c1}.json"))
        b = glob.glob(os.path.join(d2, f"*_{c2}.json"))
        if not a or not b:
            print(f"  {tag:<34} (missing)"); return
        A = {str(r["example_id"]): (r.get("sampler_search_calls") or 0) for r in json.load(open(a[0]))}
        B = {str(r["example_id"]): (r.get("sampler_search_calls") or 0) for r in json.load(open(b[0]))}
        ids = sorted(set(A) & set(B))
        if restrict:
            ids = [i for i in ids if i in test]
        d = [B[i] - A[i] for i in ids]
        p = wilcoxon(d).pvalue if any(d) else 1.0
        m = statistics.mean(A[i] for i in ids)
        print(f"  {tag:<34} n={len(ids):>4}  Δ={statistics.mean(d):+.2f} calls "
              f"({100.0*statistics.mean(d)/m:+.1f}%)  |Δ|={abs(statistics.mean(d)):.2f}  p={p:.3f}")

    one("FRAMES   SFT resolved", f"results/frames_cue_eval_resolved/{SFT_SLUG}",
        f"results/frames_cue_eval_resolved_rerun/{SFT_SLUG}", "verbose_plain", "verbose_plain", True)
    one("FRAMES   baseline", f"results/frames_cues_full/{BASE_SLUG}",
        f"results/frames_cues_rerun/{BASE_SLUG}", "verbose_plain", "verbose_plain", True)
    one("HOTPOTQA SFT resolved", f"results/hotpotqa_cue_grid/{SFT_SLUG}",
        f"results/hotpotqa_cue_grid/{SFT_SLUG}", "plain", "plain_rep2", False)
    one("HOTPOTQA baseline", f"results/hotpotqa_cue_grid/{BASE_SLUG}",
        f"results/hotpotqa_cue_grid/{BASE_SLUG}", "plain", "plain_rep2", False)


if __name__ == "__main__":
    main()
    floors()
