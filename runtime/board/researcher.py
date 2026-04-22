"""Brave Search pre-fetch for `/board --research`.

`BraveSearchClient` — thin async httpx wrapper around the Brave Web Search API.
`BoardResearcher` — orchestrates fetch and formats the context block for prompt injection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

import httpx

_HTTP_ERROR_MIN: Final[int] = 400

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
    """Fetch Brave Search snippets and format them for prompt injection."""

    def __init__(self, client: BraveSearchClient) -> None:
        self._client = client

    async def fetch(self, question: str) -> ResearchContext | None:
        started = time.perf_counter()
        try:
            results = await self._client.search(question)
        except BraveSearchError:
            logger.warning("board.researcher.fetch_failed", exc_info=True)
            return None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ResearchContext(
            query=question,
            results=tuple(results),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def format_context(ctx: ResearchContext) -> str:
        lines = ["[Research context — Brave Search]"]
        for i, r in enumerate(ctx.results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   {r.url}")
            lines.append(f"   {r.description}")
        lines.append("---")
        return "\n".join(lines)


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
            if resp.status_code >= _HTTP_ERROR_MIN:
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


__all__ = [
    "BoardResearcher",
    "BraveSearchClient",
    "BraveSearchError",
    "ResearchContext",
    "SearchResult",
]
