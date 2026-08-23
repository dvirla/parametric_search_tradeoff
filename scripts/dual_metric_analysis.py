"""
Dual-metric robustness check: does the search-vs-accuracy decoupling trend
(and the per-cue accuracy pattern, esp. `direct`) hold under BOTH the
LLM-judge grading (sampler_correct, primary) and a SQuAD-style EM regrade
(regex_strict, secondary)?

Reuses grading logic from scripts/regrade_regex.py. Reads raw grid files
directly (not the already-aggregated per_condition.csv) so we can also pull
sampler_search_calls per row, which regrade_regex.py's aggregation drops.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from src.services.common import normalize_response  # noqa: E402
from regrade_regex import heuristic_match  # noqa: E402


def frames_cond(fname):
    m = re.match(r"^frames-cues_baseline_(?P<model>.+?)_(?P<condition>(?:terse|verbose|epi_strong)_[a-z0-9_]+)\.json$", fname)
    return None if m is None else m.group("condition")


def medqa_cond(fname):
    m = re.match(r"^medqa-(?:500|terse)_baseline_(?P<model>.+?)_(?P<phrasing>orig|terse)_(?P<cue>[a-z0-9_]+)\.json$", fname)
    return None if m is None else f"{m.group('phrasing')}_{m.group('cue')}"


DATASETS = {
    "frames": dict(grid_dir="results/frames_cues_full", cond_fn=frames_cond),
    "medqa": dict(grid_dir="results/medqa_grid", cond_fn=medqa_cond),
}


def cue_of(condition: str) -> str:
    if condition.startswith("epi_strong_"):
        return condition.replace("epi_strong_", "")
    return condition.split("_", 1)[1]


# AgentAsSampler.acall() (src/services/agent_sampler.py) counts search calls over
# pydantic-ai's all_messages(), which includes the injected message_history -- so
# raw sampler_search_calls for these history-injected cues is inflated by exactly
# this many FAKE search calls from the mocked history itself (each round injects
# one tool_call; see analyze_necessity_vs_template_search_5run.py for the full
# explanation, and scripts/compare_searchmulti_rounds.py where this was first
# discovered and corrected).
MOCK_HISTORY_OFFSET = {"searchmulti": 1, "searchmulti2": 2, "searchmulti3": 3}


def plain_cond_for(condition: str) -> str:
    if condition.startswith("epi_strong_"):
        return "verbose_plain"
    phrasing = condition.split("_", 1)[0]
    return f"{phrasing}_plain"


UNGRADED_SKIPPED = []  # (dataset, model, condition) files where sampler_correct is None for every row


def load_all():
    """Return list of dicts: dataset, model, condition, llm_correct, regex_strict, search_calls.

    Skips (dataset, model, condition) files where `sampler_correct` is None for
    every row -- these were never LLM-graded (grading pass didn't run / crashed
    silently), and bool(None)==False would otherwise fabricate a fake accuracy
    collapse under those cues. See UNGRADED_SKIPPED for what got dropped.
    """
    out = []
    for dataset, cfg in DATASETS.items():
        grid_dir = os.path.join(REPO, cfg["grid_dir"])
        for path in sorted(glob.glob(os.path.join(grid_dir, "*", "*.json"))):
            model = os.path.basename(os.path.dirname(path))
            fname = os.path.basename(path)
            cond = cfg["cond_fn"](fname)
            if cond is None:
                continue
            rows = json.load(open(path))
            if rows and all(row.get("sampler_correct") is None for row in rows):
                UNGRADED_SKIPPED.append((dataset, model, cond))
                continue
            cue = cue_of(cond)
            mock_offset = MOCK_HISTORY_OFFSET.get(cue, 0)
            for row in rows:
                gold = row.get("correct_answer") or ""
                resp_raw = row.get("sampler_response") or ""
                resp = normalize_response(resp_raw)
                raw_calls = row.get("sampler_search_calls")
                corrected_calls = max(0, raw_calls - mock_offset) if raw_calls is not None else None
                out.append(dict(
                    dataset=dataset,
                    model=model,
                    condition=cond,
                    cue=cue,
                    llm_correct=bool(row.get("sampler_correct")),
                    regex_strict=heuristic_match(gold, resp),
                    search_calls=corrected_calls,
                ))
    return out


def aggregate(rows):
    """Per (dataset, model, condition): n, llm_acc, regex_acc, mean_search_calls."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["model"], r["condition"])].append(r)
    out = []
    for (ds, model, cond), items in groups.items():
        n = len(items)
        llm_acc = np.mean([it["llm_correct"] for it in items])
        regex_acc = np.mean([it["regex_strict"] for it in items])
        sc = [it["search_calls"] for it in items if it["search_calls"] is not None]
        mean_sc = float(np.mean(sc)) if sc else float("nan")
        out.append(dict(dataset=ds, model=model, condition=cond, cue=cue_of(cond),
                         n=n, llm_acc=llm_acc, regex_acc=regex_acc, mean_search_calls=mean_sc))
    return out


def main():
    rows = load_all()
    print(f"Loaded {len(rows)} raw example-rows across both datasets.")
    print(f"Skipped {len(UNGRADED_SKIPPED)} (dataset,model,condition) files with 100% ungraded sampler_correct:")
    skipped_cues = sorted(set(c for _, _, c in UNGRADED_SKIPPED))
    print(f"  conditions affected: {skipped_cues}\n")
    cond_table = aggregate(rows)
    print(f"{len(cond_table)} (dataset, model, condition) cells.\n")

    # ------------------------------------------------------------------
    # 1) Overall level-agreement: llm_acc vs regex_acc across every cell
    # ------------------------------------------------------------------
    llm_accs = np.array([r["llm_acc"] for r in cond_table])
    regex_accs = np.array([r["regex_acc"] for r in cond_table])
    pear_r, pear_p = stats.pearsonr(llm_accs, regex_accs)
    spear_r, spear_p = stats.spearmanr(llm_accs, regex_accs)
    print("=== 1) Level agreement across all (dataset, model, condition) cells ===")
    print(f"  n={len(cond_table)}")
    print(f"  Pearson  r={pear_r:.4f} (p={pear_p:.2e})")
    print(f"  Spearman r={spear_r:.4f} (p={spear_p:.2e})")
    print(f"  Mean llm_acc={llm_accs.mean():.4f}, mean regex_acc={regex_accs.mean():.4f}, "
          f"mean gap={np.mean(llm_accs - regex_accs):.4f}")
    print()

    # per-dataset breakdown
    for ds in DATASETS:
        sub = [r for r in cond_table if r["dataset"] == ds]
        la = np.array([r["llm_acc"] for r in sub])
        ra = np.array([r["regex_acc"] for r in sub])
        pr, pp = stats.pearsonr(la, ra)
        sr, sp = stats.spearmanr(la, ra)
        print(f"  [{ds}] n={len(sub)}  Pearson r={pr:.4f} (p={pp:.2e})  Spearman r={sr:.4f} (p={sp:.2e})")
    print()

    # ------------------------------------------------------------------
    # 2) Per-cue delta-vs-plain agreement (does the cue *effect* replicate?)
    # ------------------------------------------------------------------
    by_key = {(r["dataset"], r["model"], r["condition"]): r for r in cond_table}
    deltas = []
    for r in cond_table:
        plain_cond = plain_cond_for(r["condition"])
        if plain_cond == r["condition"]:
            continue
        plain = by_key.get((r["dataset"], r["model"], plain_cond))
        if plain is None:
            continue
        deltas.append(dict(
            dataset=r["dataset"], model=r["model"], cue=r["cue"], condition=r["condition"],
            d_llm=r["llm_acc"] - plain["llm_acc"],
            d_regex=r["regex_acc"] - plain["regex_acc"],
            d_search=r["mean_search_calls"] - plain["mean_search_calls"],
        ))

    d_llm = np.array([d["d_llm"] for d in deltas])
    d_regex = np.array([d["d_regex"] for d in deltas])
    d_search = np.array([d["d_search"] for d in deltas])

    pr, pp = stats.pearsonr(d_llm, d_regex)
    sr, sp = stats.spearmanr(d_llm, d_regex)
    same_sign = np.mean(np.sign(d_llm) == np.sign(d_regex))
    print("=== 2) Cue-EFFECT agreement (delta vs. matched plain condition) ===")
    print(f"  n={len(deltas)} non-plain (model,condition) cells")
    print(f"  Pearson  r={pr:.4f} (p={pp:.2e})")
    print(f"  Spearman r={sr:.4f} (p={sp:.2e})")
    print(f"  Same-sign rate (delta_llm vs delta_regex): {same_sign:.1%}")
    print()

    # per-cue pooled means
    print("  Per-cue pooled mean delta-vs-plain (pp), llm vs regex_strict, and mean search-call delta:")
    print(f"  {'cue':22s} {'n':>4s} {'d_llm':>8s} {'d_regex':>8s} {'same_sign?':>11s} {'d_search':>9s}")
    by_cue = defaultdict(list)
    for d in deltas:
        by_cue[d["cue"]].append(d)
    for cue in sorted(by_cue, key=lambda c: -len(by_cue[c])):
        items = by_cue[cue]
        dl = np.mean([x["d_llm"] for x in items])
        dr = np.mean([x["d_regex"] for x in items])
        ds_ = np.mean([x["d_search"] for x in items])
        sign_match = "YES" if np.sign(dl) == np.sign(dr) or abs(dl) < 0.02 else "no"
        print(f"  {cue:22s} {len(items):>4d} {100*dl:>+7.1f}% {100*dr:>+7.1f}% {sign_match:>11s} {ds_:>+8.2f}")
    print()

    # ------------------------------------------------------------------
    # 3) Search/accuracy DECOUPLING under both metrics
    #    (does search behavior move a lot while accuracy stays flat,
    #     under llm_acc AND under regex_acc?)
    # ------------------------------------------------------------------
    print("=== 3) Search-behavior vs accuracy decoupling, both metrics ===")
    mask = ~np.isnan(d_search)
    pr_ls, pp_ls = stats.pearsonr(d_search[mask], d_llm[mask])
    pr_rs, pp_rs = stats.pearsonr(d_search[mask], d_regex[mask])
    print(f"  corr(delta_search_calls, delta_llm_acc)   r={pr_ls:.4f} (p={pp_ls:.2e})  n={mask.sum()}")
    print(f"  corr(delta_search_calls, delta_regex_acc)  r={pr_rs:.4f} (p={pp_rs:.2e})  n={mask.sum()}")
    print(f"  mean |delta_search_calls| = {np.mean(np.abs(d_search[mask])):.2f} calls "
          f"vs mean |delta_llm_acc| = {np.mean(np.abs(d_llm[mask])):.3f}, "
          f"mean |delta_regex_acc| = {np.mean(np.abs(d_regex[mask])):.3f}")
    print()

    # ------------------------------------------------------------------
    # 4) Zoom on `direct` cue specifically (paper already flags EM artifact here)
    # ------------------------------------------------------------------
    print("=== 4) `direct` cue zoom (per dataset, pooled across models) ===")
    for ds in DATASETS:
        items = [d for d in deltas if d["cue"] == "direct" and d["dataset"] == ds]
        if not items:
            continue
        dl = np.mean([x["d_llm"] for x in items])
        dr = np.mean([x["d_regex"] for x in items])
        print(f"  [{ds}] n={len(items)} models  delta_llm_acc={100*dl:+.1f}pp  delta_regex_acc={100*dr:+.1f}pp")
        for x in sorted(items, key=lambda x: x["model"]):
            print(f"      {x['model']:28s} d_llm={100*x['d_llm']:+6.1f}pp  d_regex={100*x['d_regex']:+6.1f}pp")
    print()

    # write a CSV appendix table
    out_path = os.path.join(REPO, "results", "dual_metric_cue_deltas.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "cue", "condition", "d_llm", "d_regex", "d_search"])
        w.writeheader()
        for d in sorted(deltas, key=lambda x: (x["dataset"], x["cue"], x["model"])):
            w.writerow(d)
    print(f"Wrote per-cell delta table: {out_path}")

    out_path2 = os.path.join(REPO, "results", "dual_metric_condition_table.csv")
    with open(out_path2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "condition", "cue", "n", "llm_acc", "regex_acc", "mean_search_calls"])
        w.writeheader()
        for r in sorted(cond_table, key=lambda x: (x["dataset"], x["model"], x["condition"])):
            w.writerow(r)
    print(f"Wrote per-cell level table: {out_path2}")


if __name__ == "__main__":
    main()
