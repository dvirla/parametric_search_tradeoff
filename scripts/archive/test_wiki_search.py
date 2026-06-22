"""
Smoke test for WikipediaSearchService (the free, live MediaWiki search backend).

Runs real queries against the live Wikipedia API, prints ranked {title, snippet}
results, and asserts the service honours the BraveSearchService contract so it can be
used as a drop-in search tool. Also verifies the disk cache is written and reused.

Usage:
    uv run python scripts/test_wiki_search.py
    uv run python scripts/test_wiki_search.py --query "Bill Cockcroft chemist" --max-results 5
    uv run python scripts/test_wiki_search.py --no-cache
"""

import os
import sys
import re
import argparse
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.wiki_search import WikipediaSearchService

_TAG_RE = re.compile(r"<[^>]+>")

DEFAULT_QUERIES = [
    "Bill Cockcroft chemist",
    "most prevalent religion in Jordan",
]


def setup_args():
    parser = argparse.ArgumentParser(description="Smoke test for WikipediaSearchService.")
    parser.add_argument(
        "--query", action="append", default=None,
        help="Query to run (repeatable). Defaults to a built-in set.",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Max results per query.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the disk cache.")
    return parser.parse_args()


def check(condition: bool, label: str, failures: list):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        failures.append(label)


def assert_contract(results, max_results, failures, require_results=True):
    """Assert a result list honours the BraveSearchService contract."""
    check(isinstance(results, list), "returns a list", failures)
    if not isinstance(results, list):
        return
    check(len(results) <= max_results, f"len(results) <= max_results ({len(results)} <= {max_results})", failures)
    if require_results:
        check(len(results) >= 1, "returns >= 1 result for a known query", failures)
    for i, item in enumerate(results):
        ok_keys = isinstance(item, dict) and "title" in item and "snippet" in item
        check(ok_keys, f"item[{i}] has 'title' and 'snippet' keys", failures)
        if not ok_keys:
            continue
        check(isinstance(item["title"], str) and isinstance(item["snippet"], str),
              f"item[{i}] title/snippet are strings", failures)
        check(not _TAG_RE.search(item.get("snippet") or ""),
              f"item[{i}] snippet has no leftover HTML tags", failures)


def main():
    args = setup_args()
    queries = args.query if args.query else DEFAULT_QUERIES
    use_cache = not args.no_cache
    failures = []

    # Use a throwaway cache dir so the test is self-contained and repeatable.
    cache_dir = None
    tmp_ctx = None
    if use_cache:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="wiki_cache_test_")
        cache_dir = tmp_ctx.name

    service = WikipediaSearchService(cache_dir=cache_dir, use_cache=use_cache)

    print(f"\n=== WikipediaSearchService smoke test (cache={'on' if use_cache else 'off'}) ===\n")

    for q in queries:
        print(f"Query: {q!r}  (max_results={args.max_results})")
        results = service.search(q, args.max_results)
        for r in results:
            snippet = (r.get("snippet") or "")
            preview = snippet if len(snippet) <= 160 else snippet[:157] + "..."
            print(f"  - {r.get('title')}: {preview}")
        if not results:
            print("  (no results)")
        assert_contract(results, args.max_results, failures)
        print()

    # Cache behaviour: same query twice -> file written, second call matches.
    if use_cache:
        print("Cache check:")
        q = queries[0]
        first = service.search(q, args.max_results)
        key = service._cache_path(q)
        check(os.path.exists(key), "cache file created after first call", failures)
        # Disable network to prove the second call is served from disk.
        original_api = service._search_api
        service._search_api = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("network hit on a cached query")
        )
        try:
            second = service.search(q, args.max_results)
            check(second == first, "second call served from cache (no network)", failures)
        except AssertionError as e:
            check(False, f"second call served from cache (no network) [{e}]", failures)
        finally:
            service._search_api = original_api
        print()

    if tmp_ctx is not None:
        tmp_ctx.cleanup()

    print("=== Summary ===")
    if failures:
        print(f"FAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks PASSED.")


if __name__ == "__main__":
    main()
