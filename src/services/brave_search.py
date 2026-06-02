import requests
import os
import json
import hashlib
import tempfile
from dotenv import load_dotenv
import time

load_dotenv()

class BraveSearchService:
    _BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, cache_dir: str = None, use_cache: bool = True):
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY environment variable not set.")
        # Persistent disk cache keyed on (query, max_results). Identical queries
        # issued across phrasing runs (or repeated runs) are served from disk
        # instead of paying for another Brave API call. Disable with BRAVE_CACHE=0.
        self.use_cache = use_cache and os.getenv("BRAVE_CACHE", "1") != "0"
        self.cache_dir = cache_dir or os.getenv("BRAVE_CACHE_DIR", ".brave_cache")
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
        Search using Brave Search API with automatic pagination.
        Args:
            query: The search query string
            max_results: Total number of results to retrieve (up to 100)

        Returns:
            List of formatted search results
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
        """Uncached Brave API call with pagination."""
        all_results = []
        
        # If max_results <= 20, make a single request
        if max_results <= 20:
            count = max_results
            pages_needed = 1
        else:
            # For max_results > 20, use 20 results per page and calculate pages needed
            count = 20
            pages_needed = min((max_results + 19) // 20, 5)  # Max 5 pages
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        
        for offset in range(pages_needed):
            params = {"q": query, "count": count, "offset": offset}
            
            try:
                response = requests.get(self._BRAVE_API_ENDPOINT, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                results = self._format_results(data)
                all_results.extend(results)
                
                # Stop if we have enough results
                if len(all_results) >= max_results:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error: Brave API request failed at offset {offset}: {e}")
                break
            time.sleep(1)  # Add a delay between requests to avoid rate limiting
        
        return all_results[:max_results]

    def _format_results(self, data):
        search_results_data = data.get("web", {}).get("results", [])
        formatted_results = []
        for result in search_results_data:
            formatted_results.append({
                "title": result.get("title"),
                "snippet": result.get("description"),
            })
        return formatted_results

if __name__ == '__main__':
    import json
    service = BraveSearchService()
    search_results = service.search("what is the capital of israel?", 20)
    print(json.dumps(search_results, indent=2))
