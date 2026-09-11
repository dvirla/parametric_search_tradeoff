"""
Single-source-of-truth analysis for the RESOLVED gemma-4 FRAMES SFT
(`gemma4-frames-resolved-q4km`), across both evaluation arms and both datasets.

Regenerates every number in docs/resolved_sft_handoff.md. Run it before trusting any
figure in that document -- the tables there are pasted from this script's stdout.

    uv run python scripts/analyze_resolved_sft.py                 # all sections
    uv run python scripts/analyze_resolved_sft.py --section search

Metric conventions (match the rest of the project):
  * PRIMARY search metric = % deviation in mean search calls from the SAME arm's own plain
    condition. Comparing arms means comparing those deviations, NOT absolute call counts --
    the arms sit at different plain levels, so equal absolute drops are different robustness.
  * Arm-vs-arm significance = bootstrap over QUESTIONS of the paired difference in that
    deviation. Both arms are evaluated on the same question set, so the resample is shared.
  * Grading = deterministic regex (`heuristic_match`), never an LLM judge, and always after
    `strip_reasoning_channel` so leaked gemma-4 reasoning cannot inflate substring matches.
  * HotpotQA search counts come from the graded per_row.csv, which applies
    HISTORY_SEARCH_OFFSET -- searchmulti's mocked history contains its own tool call and the
    raw `sampler_search_calls` counter is wrong for that condition.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import statistics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.regrade_regex import heuristic_match, strip_reasoning_channel  # noqa: E402

SFT_SLUG = "gemma4-frames-resolved-q4km"
BASE_SLUG = "gemma4_31b"
SEED = 0

# FRAMES: the 7 conditions the SFT trained on, then the 3 deliberately held out.
FRAMES_SEEN = ["verbose_polite", "terse_plain", "verbose_natural",
               "verbose_elaborate", "verbose_query", "verbose_direct"]
FRAMES_UNSEEN = ["verbose_confident_parametric", "verbose_multiturn", "verbose_searchmulti"]
# HotpotQA names the same cues without the verbose_ prefix; terse_plain has no counterpart.
HPQ_SEEN = ["natural", "elaborate", "polite", "direct", "query"]
HPQ_UNSEEN = ["confident_parametric", "multiturn", "searchmulti"]


def _boot_paired(fn, ids, B=4000):
    """Bootstrap a paired statistic over questions; returns (obs, lo95, hi95)."""
    rng = random.Random(SEED)
    obs = fn(ids)
    draws = sorted(fn([rng.choice(ids) for _ in ids]) for _ in range(B))
    return obs, draws[int(0.025 * B)], draws[int(0.975 * B)]


# ---------------------------------------------------------------- FRAMES search
def _frames_search():
    test = set(map(str, json.load(open("data/sft/frames_gemma4_resolved/test_ids.json"))))

    def load(pat, sep):
        out = {}
        for fp in glob.glob(pat):
            cond = os.path.basename(fp).split(sep, 1)[-1].replace(".json", "")
            rows = [r for r in json.load(open(fp)) if str(r["example_id"]) in test]
            if rows:
                out[cond] = rows
        return out

    sft = load(f"results/frames_cue_eval_resolved/*/frames-cues_baseline_{SFT_SLUG}_*.json", "-q4km_")
    base = load(f"results/frames_cues_full/{BASE_SLUG}/frames-cues_baseline_gemma4:31b_*.json", "gemma4:31b_")
    return sft, base, "verbose_plain", FRAMES_SEEN, FRAMES_UNSEEN, "sampler_search_calls"


def _hpq_search():
    import pandas as pd
    df = pd.read_csv("results/hotpotqa_cue_grid_regex/per_row.csv")
    out = {}
    for model, slug in ((SFT_SLUG, "sft"), (BASE_SLUG, "base")):
        d = df[df.model == model]
        out[slug] = {c: g for c, g in d.groupby("run_name")}
    return out


def section_search():
    print("=" * 92)
    print("ARM 1 -- SEARCH (tools PRESENT).  Primary metric: %Δ search calls vs own plain.")
    print("=" * 92)

    # ---- FRAMES (102 held-out test questions)
    sft, base, plain, seen, unseen, k = _frames_search()
    sc = lambda rows: {str(r["example_id"]): (r.get(k) or 0) for r in rows}
    ac = lambda rows: 100.0 * statistics.mean(
        [heuristic_match(r["correct_answer"], strip_reasoning_channel(r.get("sampler_response") or "")) for r in rows])
    print("\n### FRAMES  (n=102 held-out test questions)")
    print(f"{'condition':<30} {'SFT sc':>7} {'SFT Δ%':>8} {'SFT acc':>8} | {'base sc':>8} {'base Δ%':>8} {'base acc':>9}")
    pm = statistics.mean(sc(sft[plain]).values()); bm = statistics.mean(sc(base[plain]).values())
    devs = {}
    for c in [plain] + seen + unseen:
        if c not in sft or c not in base:
            continue
        s, b = statistics.mean(sc(sft[c]).values()), statistics.mean(sc(base[c]).values())
        ds, db = 100.0 * (s - pm) / pm, 100.0 * (b - bm) / bm
        devs[c] = (ds, db)
        print(f"{c:<30} {s:>7.2f} {ds:>+7.1f}% {ac(sft[c]):>7.1f}% | {b:>8.2f} {db:>+7.1f}% {ac(base[c]):>8.1f}%")
    for lbl, cs in (("SEEN", seen), ("UNSEEN", unseen)):
        cs = [c for c in cs if c in devs]
        print(f"  {lbl:<6} mean|Δ|:  SFT {statistics.mean(abs(devs[c][0]) for c in cs):>5.1f}%   "
              f"baseline {statistics.mean(abs(devs[c][1]) for c in cs):>5.1f}%")

    # ---- HotpotQA (300 questions, graded per_row with searchmulti offset applied)
    h = _hpq_search()
    print("\n### HOTPOTQA  (n=300; search_calls carry HISTORY_SEARCH_OFFSET correction)")
    print(f"{'condition':<30} {'SFT sc':>7} {'SFT Δ%':>8} {'SFT acc':>8} | {'base sc':>8} {'base Δ%':>8} {'base acc':>9}")
    pm = h["sft"]["plain"].search_calls.mean(); bm = h["base"]["plain"].search_calls.mean()
    devs = {}
    for c in ["plain"] + HPQ_SEEN + HPQ_UNSEEN:
        if c not in h["sft"] or c not in h["base"]:
            continue
        s, b = h["sft"][c].search_calls.mean(), h["base"][c].search_calls.mean()
        ds, db = 100.0 * (s - pm) / pm, 100.0 * (b - bm) / bm
        devs[c] = (ds, db)
        print(f"{c:<30} {s:>7.2f} {ds:>+7.1f}% {h['sft'][c].strict.mean()*100:>7.1f}% | "
              f"{b:>8.2f} {db:>+7.1f}% {h['base'][c].strict.mean()*100:>8.1f}%")
    for lbl, cs in (("SEEN", HPQ_SEEN), ("UNSEEN", HPQ_UNSEEN)):
        cs = [c for c in cs if c in devs]
        print(f"  {lbl:<6} mean|Δ|:  SFT {statistics.mean(abs(devs[c][0]) for c in cs):>5.1f}%   "
              f"baseline {statistics.mean(abs(devs[c][1]) for c in cs):>5.1f}%")

    # ---- paired arm-vs-arm test on the unseen cues
    print("\n### Paired arm-vs-arm test on %Δ (bootstrap over questions, 95% CI)")
    print(f"{'dataset':<10} {'cue':<24} {'SFT Δ%':>8} {'base Δ%':>9} {'SFT-base':>9} {'95% CI':>20}")
    fs, fb, fplain, _, _, kk = _frames_search()
    for ds_name, N, B, pk, cues in (
        ("FRAMES", {c: sc(v) for c, v in fs.items()}, {c: sc(v) for c, v in fb.items()}, fplain, FRAMES_UNSEEN),
        ("HOTPOTQA",
         {c: dict(zip(g.example_id.astype(str), g.search_calls)) for c, g in h["sft"].items()},
         {c: dict(zip(g.example_id.astype(str), g.search_calls)) for c, g in h["base"].items()},
         "plain", HPQ_UNSEEN),
    ):
        for c in cues:
            if c not in N or c not in B:
                continue
            ids = sorted(set(N[pk]) & set(N[c]) & set(B[pk]) & set(B[c]))
            def pct(P, C, ii):
                p = statistics.mean([P[i] for i in ii]); q = statistics.mean([C[i] for i in ii])
                return 100.0 * (q - p) / p
            f = lambda ii: pct(N[pk], N[c], ii) - pct(B[pk], B[c], ii)
            obs, lo, hi = _boot_paired(f, ids)
            print(f"{ds_name:<10} {c:<24} {pct(N[pk],N[c],ids):>+7.1f}% {pct(B[pk],B[c],ids):>+8.1f}% "
                  f"{obs:>+8.1f}%  [{lo:+.1f}, {hi:+.1f}]  {'SEPARABLE' if (lo>0 or hi<0) else 'not sep.'}")


# ------------------------------------------------------------- parametric arm
def _load_param(root, model, tag):
    g = {}
    for fp in glob.glob(f"results/{root}/{model}/*_run_*.json"):
        stem = os.path.basename(fp)
        m = re.search(r"_run_(\d+)\.json$", stem)
        cond = stem[:m.start()].split(tag, 1)[-1].lstrip("_") or "plain"
        g.setdefault(cond, []).extend(json.load(open(fp)))
    return g


def section_parametric():
    MARK = re.compile(r"<channel\|>|<\|channel>")
    print("\n" + "=" * 92)
    print("ARM 2 -- PARAMETRIC (tools ABSENT).  Does the model still answer, and correctly?")
    print("=" * 92)
    for root, label, btag in (("frames_parametric", "FRAMES", "gemma4:31b"),
                              ("hotpotqa_parametric", "HOTPOTQA", "gemma4:31b")):
        N = _load_param(root, SFT_SLUG, SFT_SLUG)
        B = _load_param(root, BASE_SLUG, btag)
        print(f"\n### {label} — 5 runs pooled")
        print(f"{'condition':<24} {'n':>6} {'trunc%':>7} {'acc%':>7} | {'base n':>7} {'base acc%':>10}")
        for c in ["plain", "elaborate", "direct", "multiturn", "confident_parametric"]:
            if c not in N:
                continue
            rows = N[c]
            # A response is only truly truncated if NOTHING follows the leaked marker. Do NOT
            # use a word-count threshold: correct one-word answers to `direct` look identical.
            trunc = sum(1 for r in rows if MARK.search(r.get("sampler_response") or "")
                        and not MARK.split(r["sampler_response"])[-1].strip())
            acc = 100.0 * statistics.mean([heuristic_match(
                r["correct_answer"], strip_reasoning_channel(r.get("sampler_response") or "")) for r in rows])
            bacc = (100.0 * statistics.mean([heuristic_match(
                r["correct_answer"], strip_reasoning_channel(r.get("sampler_response") or "")) for r in B[c]])
                if c in B else float("nan"))
            print(f"{c:<24} {len(rows):>6} {100.0*trunc/len(rows):>6.2f}% {acc:>6.1f}% | "
                  f"{len(B.get(c,[])):>7} {bacc:>9.1f}%")


def section_entropy():
    print("\n" + "=" * 92)
    print("ARM 2b -- SEMANTIC ENTROPY (belief).  Same gpt-oss:120b judge/prompt as all baselines.")
    print("=" * 92)

    def load(root, model, tag):
        g = {}
        for fp in glob.glob(f"results/{root}/{model}/*llm_clusters_5run.json"):
            cond = os.path.basename(fp).replace("_llm_clusters_5run.json", "").split(tag, 1)[-1].lstrip("_") or "plain"
            g[cond] = {str(r["example_id"]): r["semantic_entropy"] for r in json.load(open(fp))
                       if r.get("semantic_entropy") is not None}
        return g

    for root, label, btag in (("frames_parametric", "FRAMES", "gemma4:31b"),
                              ("hotpotqa_parametric", "HOTPOTQA", "gemma4:31b")):
        N, B = load(root, SFT_SLUG, SFT_SLUG), load(root, BASE_SLUG, btag)
        if "plain" not in N or "plain" not in B:
            print(f"\n### {label} — clusters missing"); continue
        common = sorted(set(N["plain"]) & set(B["plain"]))
        print(f"\n### {label} — plain entropy: SFT {statistics.mean([N['plain'][i] for i in common]):.3f} "
              f"vs baseline {statistics.mean([B['plain'][i] for i in common]):.3f} bits (n={len(common)})")
        print(f"{'cue':<24} {'SFT Δbits':>10} {'base Δbits':>11} {'SFT-base':>9} {'95% CI':>20}")
        for c in ["elaborate", "direct", "multiturn", "confident_parametric"]:
            if c not in N or c not in B:
                continue
            ids = sorted(set(N["plain"]) & set(N[c]) & set(B["plain"]) & set(B[c]))
            d = lambda P, C, ii: statistics.mean([C[i] for i in ii]) - statistics.mean([P[i] for i in ii])
            f = lambda ii: d(N["plain"], N[c], ii) - d(B["plain"], B[c], ii)
            obs, lo, hi = _boot_paired(f, ids)
            print(f"{c:<24} {d(N['plain'],N[c],ids):>+9.3f} {d(B['plain'],B[c],ids):>+10.3f} "
                  f"{obs:>+8.3f}  [{lo:+.3f}, {hi:+.3f}] {'sep.' if (lo>0 or hi<0) else 'not sep.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=["search", "parametric", "entropy", "all"], default="all")
    a = ap.parse_args()
    if a.section in ("search", "all"):
        section_search()
    if a.section in ("parametric", "all"):
        section_parametric()
    if a.section in ("entropy", "all"):
        section_entropy()
