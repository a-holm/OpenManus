from duckduckgo_search import DDGS

from app.tool.search.base import WebSearchEngine


class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=100, *args, **kwargs):
        """DuckDuckGo search engine."""
        return DDGS.text(query, max_results=num_results)
