"""
Summarize paper-relevant statistics over the curated FRAMES cue-robustness SFT set.

Re-derives exactly which raw rollouts curate_frames_sft_data.py kept (same
grouping/reference/threshold logic, imported directly so the numbers can never
drift from what actually went into the training file) and reports, per
condition: how many kept trajectories contain a search call vs none, and how
their search-call count compares to the same question's plain-original
reference (fewer / same / more). The curated JSONL itself only stores
{"messages": [...]} -- condition/search_calls/is_correct are dropped during
curation -- so this script re-joins that metadata from the raw rollouts file.

Usage:
    uv run python scripts/summarize_sft_curation_stats.py \
        --rollouts data/sft/frames_gemma4/rollouts.jsonl \
        --require-correct-plain-ref --threshold 1
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.curate_frames_sft_data import (
    PLAIN_CONDITION,
    group_by_example,
    is_test,
    load_rollouts,
    plain_reference,
)


def kept_records_with_ref(cond_map, ref, threshold, max_per_condition=None):
    """Same selection as curate_frames_sft_data.kept_for_example, but returns
    (record, ref) pairs instead of stripping the reference, so callers can
    compute search_calls - ref per kept row."""
    if ref is None:
        return []
    kept = []
    plain_correct = [r for r in cond_map.get(PLAIN_CONDITION, []) if r["is_correct"]]
    if max_per_condition:
        plain_correct = plain_correct[:max_per_condition]
    kept.extend((r, ref) for r in plain_correct)
    for cond, rollouts in cond_map.items():
        if cond == PLAIN_CONDITION:
            continue
        cand = [r for r in rollouts if r["is_correct"]]
        cand = [r for r in cand if abs(r["search_calls"] - ref) <= threshold]
        if max_per_condition:
            cand = cand[:max_per_condition]
        kept.extend((r, ref) for r in cand)
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--threshold", type=int, default=1)
    p.add_argument("--test-pct", type=int, default=20)
    p.add_argument("--require-correct-plain-ref", action="store_true")
    p.add_argument("--max-per-condition", type=int, default=None)
    args = p.parse_args()

    rollouts = load_rollouts(args.rollouts)
    grouped = group_by_example(rollouts)
    train_ids = [e for e in grouped if not is_test(e, args.test_pct)]
    test_ids = [e for e in grouped if is_test(e, args.test_pct)]

    kept_all = []  # (record, ref)
    n_dropped_q = 0
    for e in train_ids:
        ref = plain_reference(grouped[e], require_correct=args.require_correct_plain_ref)
        if ref is None:
            n_dropped_q += 1
            continue
        kept_all.extend(kept_records_with_ref(grouped[e], ref, args.threshold, args.max_per_condition))

    total_raw = len(rollouts)
    total_q = len(grouped)
    n_kept = len(kept_all)

    print("=== CURATION SUMMARY ===")
    print(f"raw rollouts: {total_raw} | questions: {total_q} "
          f"({len(train_ids)} train, {len(test_ids)} test, ~{args.test_pct}% held out)")
    print(f"require_correct_plain_ref={args.require_correct_plain_ref} threshold={args.threshold}")
    print(f"train questions dropped (no valid plain ref): {n_dropped_q} | "
          f"usable train questions: {len(train_ids) - n_dropped_q}")
    print(f"KEPT (final SFT set): {n_kept} / {total_raw} raw rollouts "
          f"({100.0 * n_kept / total_raw:.1f}%)")

    # ---- per-condition breakdown ----
    by_cond = defaultdict(list)  # condition -> [(record, ref), ...]
    for r, ref in kept_all:
        by_cond[r["condition"]].append((r, ref))

    cond_order = ["verbose_plain", "verbose_polite", "terse_plain", "verbose_natural",
                  "verbose_elaborate", "verbose_query", "verbose_direct",
                  "verbose_confident_parametric", "verbose_multiturn", "verbose_searchmulti"]
    conds = [c for c in cond_order if c in by_cond] + sorted(c for c in by_cond if c not in cond_order)

    print("\n=== PER-CONDITION: search usage + closeness to plain reference (kept set) ===")
    header = (f"{'condition':30} {'n':>6} {'%total':>7} {'sc_med':>7} {'sc_mean':>8} "
              f"{'w/search':>9} {'no_search':>10} {'fewer':>7} {'same':>6} {'more':>6}")
    print(header)
    rows_out = []
    for c in conds:
        pairs = by_cond[c]
        n = len(pairs)
        sc = [r["search_calls"] for r, _ in pairs]
        with_search = sum(1 for x in sc if x > 0)
        no_search = sum(1 for x in sc if x == 0)
        if c == PLAIN_CONDITION:
            fewer = same = more = None  # the plain condition IS the reference; comparison is vacuous
        else:
            fewer = sum(1 for r, ref in pairs if r["search_calls"] < ref)
            same = sum(1 for r, ref in pairs if r["search_calls"] == ref)
            more = sum(1 for r, ref in pairs if r["search_calls"] > ref)
        line = (f"{c:30} {n:>6} {100.0*n/n_kept:>6.1f}% {statistics.median(sc):>7.1f} "
                f"{statistics.mean(sc):>8.2f} "
                f"{100.0*with_search/n:>8.1f}% {100.0*no_search/n:>9.1f}% "
                + (f"{100.0*fewer/n:>6.1f}% {100.0*same/n:>5.1f}% {100.0*more/n:>5.1f}%"
                   if fewer is not None else f"{'--':>7} {'--':>6} {'--':>6}"))
        print(line)
        rows_out.append({
            "condition": c, "n": n, "pct_of_total": 100.0 * n / n_kept,
            "search_calls_median": statistics.median(sc), "search_calls_mean": statistics.mean(sc),
            "pct_with_search": 100.0 * with_search / n, "pct_no_search": 100.0 * no_search / n,
            "pct_fewer_than_ref": None if fewer is None else 100.0 * fewer / n,
            "pct_same_as_ref": None if same is None else 100.0 * same / n,
            "pct_more_than_ref": None if more is None else 100.0 * more / n,
        })

    # ---- aggregate (cue conditions only, i.e. excluding the plain anchor) ----
    cue_pairs = [(r, ref) for r, ref in kept_all if r["condition"] != PLAIN_CONDITION]
    if cue_pairs:
        n_cue = len(cue_pairs)
        fewer = sum(1 for r, ref in cue_pairs if r["search_calls"] < ref)
        same = sum(1 for r, ref in cue_pairs if r["search_calls"] == ref)
        more = sum(1 for r, ref in cue_pairs if r["search_calls"] > ref)
        with_search = sum(1 for r, _ in cue_pairs if r["search_calls"] > 0)
        print(f"\n=== AGGREGATE across all cue conditions (excl. plain anchor), n={n_cue} ===")
        print(f"contains >=1 search call: {with_search} ({100.0*with_search/n_cue:.1f}%) | "
              f"zero search: {n_cue-with_search} ({100.0*(n_cue-with_search)/n_cue:.1f}%)")
        print(f"vs plain reference -- fewer: {fewer} ({100.0*fewer/n_cue:.1f}%) | "
              f"same: {same} ({100.0*same/n_cue:.1f}%) | "
              f"more: {more} ({100.0*more/n_cue:.1f}%)")

    # ---- raw (pre-curation) correctness + search-usage, for context ----
    print("\n=== RAW ROLLOUTS (pre-curation, all conditions, for context) ===")
    raw_by_cond = defaultdict(list)
    for r in rollouts:
        raw_by_cond[r["condition"]].append(r)
    for c in conds:
        rs = raw_by_cond.get(c, [])
        if not rs:
            continue
        ncorr = sum(1 for r in rs if r["is_correct"])
        nosearch = sum(1 for r in rs if r["search_calls"] == 0)
        print(f"{c:30} n={len(rs):>6} correct={100.0*ncorr/len(rs):>5.1f}% "
              f"zero_search={100.0*nosearch/len(rs):>5.1f}%")


if __name__ == "__main__":
    main()
