import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()

class BraveSearchService:
    _BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY environment variable not set.")

    def search(self, query: str, max_results: int = 10):
        """
        Search using Brave Search API with automatic pagination.
        
        Args:
            query: The search query string
            max_results: Total number of results to retrieve (up to 100)
        
        Returns:
            List of formatted search results
        """
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
