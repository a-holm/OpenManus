from typing import List
import requests
from googlesearch import search
from app.config import config
from app.exceptions import ToolError
from app.tool.search.base import SearchItem, WebSearchEngine


class GoogleSearchEngine(WebSearchEngine):
    """Google search implementation supporting both API and scraping."""

    def __init__(self):
        google_config = config.search_config.google if config.search_config else None
        self.api_enabled = google_config.use_api if google_config else False
        self.api_key = google_config.googlesearch_api_key if google_config else None
        self.cx = google_config.cx if google_config else None

        if self.api_enabled and (not self.api_key or not self.cx):
            raise ToolError(
                "Google Search API requires both api_key and cx to be configured"
            )

    def _api_search(self, query: str, num_results: int) -> List[SearchItem]:
        """Search using Google Custom Search JSON API.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of SearchItem objects with title, url, and description
            
        Raises:
            ToolError: If API request fails and API is enabled
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10)  # API max is 10 results per request
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            
            return [
                SearchItem(
                    title=item.get("title", f"Result {i+1}"),
                    url=item["link"],
                    description=item.get("snippet", "")
                )
                for i, item in enumerate(results.get("items", []))
            ]
            
        except requests.exceptions.RequestException as e:
            if not self.api_enabled:
                return []
            raise ToolError(f"Google API request failed: {str(e)}") from e

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """Performs Google search and returns results as SearchItem objects.
        
        Uses API if enabled, otherwise falls back to scraping.
        """
        if self.api_enabled:
            return self._api_search(query, num_results)

        try:
            raw_results = search(query, num_results=num_results, advanced=True)
            return [
                SearchItem(
                    title=item.title if hasattr(item, "title") else f"Result {i+1}",
                    url=item.url if hasattr(item, "url") else item,
                    description=getattr(item, "description", "")
                )
                for i, item in enumerate(raw_results)
            ]
        except Exception as e:
            if self.api_enabled:
                raise ToolError(
                    "Google search failed. Please check your API configuration."
                ) from e
            return []