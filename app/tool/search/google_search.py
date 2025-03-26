from typing import Any, Dict, List

import requests
from googlesearch import search

from app.config import config
from app.exceptions import ToolError
from app.tool.search.base import WebSearchEngine


class GoogleSearchEngine(WebSearchEngine):
    """Google search implementation using either official API or web scraping."""

    def __init__(self):
        google_config = config.search_config.google if config.search_config else None
        self.api_enabled = google_config.use_api if google_config else False
        self.api_key = google_config.googlesearch_api_key if google_config else None
        self.cx = google_config.cx if google_config else None

        if self.api_enabled and (not self.api_key or not self.cx):
            raise ToolError(
                "Google Search API requires both api_key and cx to be configured"
            )

    def _api_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search JSON API."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(num_results, 10),  # 10 is max on the API
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            return [
                {"title": item["title"], "link": item["link"]}
                for item in results.get("items", [])
            ]
        except requests.exceptions.RequestException as e:
            if not self.api_enabled:
                return []
            raise ToolError(f"Google API request failed: {str(e)}") from e

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[str]:
        """Returns unique URLs from Google search results.

        Uses API if enabled, otherwise falls back to scraping with built-in deduplication.
        """
        if self.api_enabled:
            api_results = self._api_search(query, num_results)
            return list({result["link"]: None for result in api_results}.keys())

        try:
            return list(search(query, num_results=num_results, unique=True))
        except Exception as e:
            if self.api_enabled:
                raise ToolError(
                    "Google search failed. Please check your API configuration."
                ) from e
            return []
