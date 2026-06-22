#!/usr/bin/env python3
"""Pre-populate the Brave search cache from existing agent traces.

Every search the agents have already run is recorded in the trace JSONs as a
(tool_call: search, arguments) paired by tool_call_id with a (tool_call_response:
result) holding the returned [{title, snippet}] list. This script replays those
pairs into the on-disk cache used by BraveSearchService, so a *new* phrasing run
that issues an identical query string serves it from disk instead of paying for
another Brave API call.

It reuses BraveSearchService._cache_path/_cache_store so backfilled entries are
byte-identical to what a live search would have written (same key, same format).

Run wherever the eval will run (the cache is a local dir, default .brave_cache/):
  uv run python scripts/backfill_brave_cache.py
  uv run python scripts/backfill_brave_cache.py --traces 'results/**/*traces*.json' --cache-dir .brave_cache
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# BraveSearchService.__init__ requires an API key even though backfill never calls
# the network; set a placeholder so we can reuse its exact cache keying/storage.
os.environ.setdefault("BRAVE_API_KEY", "backfill-noop")
from src.services.brave_search import BraveSearchService  # noqa: E402

DEFAULT_MAX_RESULTS = 10  # the search Tool's default when the model omits max_results


def _arguments(part: dict) -> dict:
    args = part.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def iter_search_pairs(trace_files: list[str]):
    """Yield (query, max_results, results) for every recorded search call."""
    for f in trace_files:
        try:
            data = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            print(f"  skip unreadable {f}")
            continue
        for ex in data:
            calls: dict[str, dict] = {}      # tool_call_id -> {query, max_results}
            responses: dict[str, list] = {}  # tool_call_id -> results
            for msg in ex.get("message_trace", []):
                for p in msg.get("parts", []):
                    t = p.get("type")
                    if t == "tool_call" and p.get("tool_name") == "search":
                        a = _arguments(p)
                        q = a.get("query")
                        if q:
                            calls[p.get("tool_call_id")] = {
                                "query": q,
                                "max_results": int(a.get("max_results", DEFAULT_MAX_RESULTS)),
                            }
                    elif t == "tool_call_response" and p.get("tool_name") == "search":
                        r = p.get("result")
                        if isinstance(r, list):
                            responses[p.get("tool_call_id")] = r
            for cid, c in calls.items():
                if cid in responses:
                    yield c["query"], c["max_results"], responses[cid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="results/**/*traces*.json",
                    help="glob (recursive) for trace JSONs")
    ap.add_argument("--cache-dir", default=None, help="cache dir (default: BRAVE_CACHE_DIR or .brave_cache)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing cache entries")
    args = ap.parse_args()

    svc = BraveSearchService(cache_dir=args.cache_dir, use_cache=True)
    files = glob.glob(args.traces, recursive=True)
    print(f"Scanning {len(files)} trace files -> cache dir {svc.cache_dir}/")

    # Cache is keyed on query only and stores the full result list, so for each
    # query keep the richest (longest) snapshot seen across all traces.
    best: dict[str, list] = {}
    n_pairs = n_empty = 0
    for query, _max_results, results in iter_search_pairs(files):
        n_pairs += 1
        if not results:                       # never cache empty/failed responses
            n_empty += 1
            continue
        if len(results) > len(best.get(query, [])):
            best[query] = results

    n_written = n_exist = 0
    for query, results in best.items():
        path = svc._cache_path(query)
        if os.path.exists(path) and not args.overwrite:
            n_exist += 1
            continue
        svc._cache_store(path, results)
        n_written += 1

    print(f"\nSearch pairs found:        {n_pairs}")
    print(f"  empty (skipped):         {n_empty}")
    print(f"  unique queries:          {len(best)}")
    print(f"  already cached (skipped):{n_exist}")
    print(f"  written:                 {n_written}")
    print(f"Unique cached queries now: {len([f for f in os.listdir(svc.cache_dir) if f.endswith('.json')])}")


if __name__ == "__main__":
    main()
