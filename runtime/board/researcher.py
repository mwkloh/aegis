"""Brave Search pre-fetch for `/board --research`.

`BraveSearchClient` — thin async httpx wrapper around the Brave Web Search API.
`BoardResearcher` — orchestrates fetch and formats the context block for prompt injection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchError(Exception):
    """Raised by `BraveSearchClient` on any non-200 or network failure."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str


@dataclass(frozen=True)
class ResearchContext:
    query: str
    results: tuple[SearchResult, ...]
    elapsed_ms: int


class BoardResearcher:
    """Stub — full implementation comes in Task 3."""


class BraveSearchClient:
    def __init__(
        self,
        api_key: str,
        *,
        top_k: int = 5,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._top_k = top_k
        self._timeout_s = timeout_s

    async def search(self, query: str) -> list[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        params = {"q": query, "count": self._top_k}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            try:
                resp = await client.get(_BRAVE_SEARCH_URL, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                raise BraveSearchError("timeout") from exc
            except httpx.HTTPError as exc:
                raise BraveSearchError(f"http error: {exc}") from exc
            if resp.status_code >= 400:
                raise BraveSearchError(f"api error {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                )
            )
        return results
