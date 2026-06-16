import requests
import os
import json
import hashlib
import tempfile
import html
import re
from dotenv import load_dotenv
import time

load_dotenv()

# Strip MediaWiki search-highlight markup (e.g. <span class="searchmatch">term</span>)
# and any other stray tags out of snippets so the agent sees clean prose.
_TAG_RE = re.compile(r"<[^>]+>")


class WikipediaSearchService:
    """Free, drop-in replacement for BraveSearchService backed by the live
    MediaWiki search API (https://www.mediawiki.org/wiki/API:Search).

    No API key, no per-call cost. Returns the same ``[{"title", "snippet"}, ...]``
    shape as BraveSearchService so it can be swapped in anywhere a search tool is
    expected. Results are cached on disk (query-keyed) for reproducibility and to
    stay friendly to Wikimedia's rate limits across repeated runs/phrasings.
    """

    def __init__(self, cache_dir: str = None, use_cache: bool = True, lang: str = "en"):
        self.lang = lang
        self._endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        # Wikimedia etiquette: identify the client with a descriptive User-Agent.
        self._headers = {
            "User-Agent": "parametric-search-tradeoff/0.1 (research; https://github.com/dvirla)",
            "Accept": "application/json",
        }
        # Persistent disk cache keyed on the query string. Identical queries issued
        # across phrasing runs (or repeated runs) are served from disk instead of
        # hitting the live API again. Disable with WIKI_CACHE=0.
        self.use_cache = use_cache and os.getenv("WIKI_CACHE", "1") != "0"
        self.cache_dir = cache_dir or os.getenv("WIKI_CACHE_DIR", ".wiki_cache")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, query: str) -> str:
        # Keyed on the query string only (not max_results): the model varies
        # max_results call-to-call, so query-only keying maximises cache hits
        # across runs/phrasings. We store the full returned list and slice on read.
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    @staticmethod
    def _cache_store(path: str, results: list) -> None:
        """Atomic write (temp file + rename) so concurrent runs never read a half-written file."""
        try:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(results, f)
            os.replace(tmp, path)
        except OSError:
            pass  # caching is best-effort; never fail a search over a cache write

    def search(self, query: str, max_results: int = 10):
        """
        Search Wikipedia for a query and return relevant article snippets.

        Args:
            query: The search query string
            max_results: Maximum number of results to retrieve

        Returns:
            A list of {"title": str, "snippet": str} dicts, ranked by relevance.
        """
        if self.use_cache:
            path = self._cache_path(query)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)[:max_results]  # slice the cached full list
                except (json.JSONDecodeError, OSError):
                    pass  # corrupt/partial cache entry — fall through and re-fetch
        results = self._search_api(query, max_results)
        if self.use_cache and results:  # don't cache empty/failed responses
            self._cache_store(self._cache_path(query), results)
        return results

    def _search_api(self, query: str, max_results: int = 10):
        """Uncached MediaWiki search API call with exponential backoff."""
        # MediaWiki caps srlimit at 500; we stay well under that.
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max(1, min(max_results, 500)),
            "srprop": "snippet",
            "format": "json",
        }

        max_retries = 4
        backoff = 1.0
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self._endpoint, headers=self._headers, params=params, timeout=15
                )
                response.raise_for_status()
                data = response.json()
                return self._format_results(data)[:max_results]
            except requests.exceptions.RequestException as e:
                # Retry on transient failures (timeouts, 429, 5xx); give up cleanly
                # otherwise so the agent loop never crashes over a search call.
                print(
                    f"Error: Wikipedia API request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    return []
                time.sleep(backoff)
                backoff *= 2
        return []

    def _format_results(self, data):
        search_results_data = data.get("query", {}).get("search", [])
        formatted_results = []
        for result in search_results_data:
            formatted_results.append({
                "title": result.get("title"),
                "snippet": self._clean_snippet(result.get("snippet", "")),
            })
        return formatted_results

    @staticmethod
    def _clean_snippet(snippet: str) -> str:
        """Strip MediaWiki highlight HTML and unescape entities to plain text."""
        return html.unescape(_TAG_RE.sub("", snippet or "")).strip()


if __name__ == "__main__":
    service = WikipediaSearchService()
    search_results = service.search("what is the capital of israel?", 5)
    print(json.dumps(search_results, indent=2))
