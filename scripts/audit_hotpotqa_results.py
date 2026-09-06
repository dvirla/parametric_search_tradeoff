r"""
Integrity audit for the HotpotQA result files (cue grid + parametric probe).

Checks each file for the failure modes this project has actually hit, not just "does it parse":

  * JSON valid, top level is a list.
  * Row count vs the tier size (a SHORT file is only an error if the run is not still going --
    the report separates SHORT from every other class so in-flight runs aren't called corrupt).
  * `example_id` present on every row, UNIQUE, and drawn from the tier's id set. Duplicates are
    the signature of two processes writing the same file (which happened: killing an ollama
    server orphans its driver, which respawns evals that race the replacement).
  * `sampler_response` non-null and non-blank -- a blank response means the row was recorded but
    the model returned nothing, and it silently grades as wrong.
  * Response not an error payload (Traceback / APITimeout / "does not support tools" -- the last
    is the gemma4-on-old-ollama failure that the eval treats as non-retryable and skips).
  * `sampler_search_calls` present, integer, >= 0; and for `no_search` files it MUST be 0 --
    a nonzero value there means the agent had a search tool it should not have had.
  * `stop_reason` non-null count: BaseAgent sets it when it salvaged a best-effort answer from a
    run that hit the loop cap or timed out. Those rows are real but degraded and should be
    filterable, so they're reported rather than silently accepted.
  * searchmulti files: reports min search_calls, which distinguishes rows collected BEFORE the
    mocked-history fix (min >= 1, a constant +1 offset) from rows collected after.

Usage:
    uv run python scripts/audit_hotpotqa_results.py
    uv run python scripts/audit_hotpotqa_results.py --roots results/hotpotqa_parametric --verbose
"""

from __future__ import annotations

import os
import re
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict

# Deliberately NARROW. A bare "rate limit" false-positived on a real answer about aircraft
# "turn-rate limits"; error payloads are recognisable by exception-shaped tokens instead.
_ERR_RE = re.compile(r"traceback \(most recent call last\)|APITimeoutError|APIConnectionError|"
                     r"RateLimitError|InternalServerError|does not support tools|"
                     r"ConnectionRefusedError|ValueError:|KeyError:", re.I)


def setup_args():
    p = argparse.ArgumentParser(description="Audit HotpotQA result files for integrity.")
    p.add_argument("--roots", nargs="+",
                   default=["results/hotpotqa_cue_grid", "results/hotpotqa_parametric"])
    p.add_argument("--dataset", default="hotpotqa-300")
    p.add_argument("--expected-rows", type=int, default=300)
    p.add_argument("--subset-file", default=None)
    p.add_argument("--verbose", action="store_true", help="List every file, not just problems.")
    return p.parse_args()


def audit_file(path, gold_ids, expected, verbose=False):
    """Return (status, list_of_issues, stats). status: OK | SHORT | BAD."""
    issues, stats = [], {}
    try:
        rows = json.load(open(path))
    except Exception as e:
        return "BAD", [f"unreadable JSON: {e}"], stats
    if not isinstance(rows, list):
        return "BAD", [f"top level is {type(rows).__name__}, expected list"], stats

    stats["rows"] = len(rows)
    is_no_search = "_no_search_" in os.path.basename(path)

    ids = [r.get("example_id") for r in rows]
    missing_id = sum(1 for i in ids if i is None)
    if missing_id:
        issues.append(f"{missing_id} rows without example_id")
    dupes = [i for i, c in Counter(i for i in ids if i is not None).items() if c > 1]
    if dupes:
        issues.append(f"DUPLICATE example_id x{len(dupes)} (concurrent writers?) e.g. {dupes[:3]}")
    if gold_ids:
        extra = set(i for i in ids if i is not None) - gold_ids
        if extra:
            issues.append(f"{len(extra)} example_id not in the tier, e.g. {sorted(extra)[:3]}")

    blank = sum(1 for r in rows if not str(r.get("sampler_response") or "").strip())
    if blank:
        issues.append(f"{blank} blank/null sampler_response")
    errs = sum(1 for r in rows if _ERR_RE.search(str(r.get("sampler_response") or "")))
    if errs:
        issues.append(f"{errs} responses look like error payloads")
    nogold = sum(1 for r in rows if not str(r.get("correct_answer") or "").strip())
    if nogold:
        issues.append(f"{nogold} rows with empty correct_answer")

    sc = [r.get("sampler_search_calls") for r in rows]
    bad_sc = sum(1 for x in sc if not isinstance(x, (int, float)) or x < 0)
    if bad_sc:
        issues.append(f"{bad_sc} rows with missing/negative sampler_search_calls")
    good_sc = [x for x in sc if isinstance(x, (int, float)) and x >= 0]
    if good_sc:
        stats["sc_min"], stats["sc_mean"] = min(good_sc), sum(good_sc) / len(good_sc)
        if is_no_search and max(good_sc) > 0:
            issues.append(f"no_search file has search_calls>0 (max {max(good_sc)})")

    stops = Counter(r.get("stop_reason") for r in rows)
    n_stop = sum(v for k, v in stops.items() if k is not None)
    stats["stop_reason"] = n_stop
    if n_stop:
        detail = ", ".join(f"{k}:{v}" for k, v in stops.items() if k is not None)
        issues.append(f"{n_stop} rows with stop_reason set ({detail}) -- salvaged/degraded")

    if issues:
        return "BAD", issues, stats
    if len(rows) < expected:
        return "SHORT", [f"{len(rows)}/{expected} rows"], stats
    if len(rows) > expected:
        return "BAD", [f"{len(rows)}/{expected} rows -- MORE than the tier"], stats
    return "OK", [], stats


def main():
    args = setup_args()
    subset = args.subset_file or f"data/{args.dataset.replace('-', '_')}.jsonl"
    gold_ids = set()
    if os.path.exists(subset):
        gold_ids = {json.loads(l)["example_id"] for l in open(subset)}
        print(f"tier {args.dataset}: {len(gold_ids)} example_ids from {subset}\n")
    else:
        print(f"[warn] {subset} missing -- skipping id-membership check\n")

    totals = Counter()
    per_root = defaultdict(Counter)
    problems, shorts = [], []
    n_rows = 0

    for root in args.roots:
        paths = sorted(glob.glob(os.path.join(root, "*", "*.json")))
        paths = [p for p in paths if not re.search(r"superseded|\.bak", p)]
        for path in paths:
            status, issues, stats = audit_file(path, gold_ids, args.expected_rows)
            totals[status] += 1
            per_root[root][status] += 1
            n_rows += stats.get("rows", 0)
            rel = os.path.relpath(path)
            if status == "BAD":
                problems.append((rel, issues))
            elif status == "SHORT":
                shorts.append((rel, stats.get("rows", 0)))
            if args.verbose and status == "OK":
                print(f"  OK    {rel}")

    print("=" * 78)
    for root in args.roots:
        c = per_root[root]
        print(f"{root}: {c['OK']} OK, {c['SHORT']} short (in progress), {c['BAD']} problems")
    print(f"TOTAL: {sum(totals.values())} files, {n_rows:,} rows  "
          f"[{totals['OK']} OK / {totals['SHORT']} short / {totals['BAD']} problem]")

    if shorts:
        print(f"\n--- SHORT ({len(shorts)}) — expected while a run is still going ---")
        for rel, n in shorts:
            print(f"  {n:>4}/{args.expected_rows}  {rel}")
    if problems:
        print(f"\n--- PROBLEMS ({len(problems)}) ---")
        for rel, issues in problems:
            print(f"  {rel}")
            for i in issues:
                print(f"      - {i}")
    else:
        print("\nNo integrity problems found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
