"""Unit tests for BraveSearchClient and BoardResearcher."""
from __future__ import annotations

import pytest
import respx
import httpx

from runtime.board.researcher import (
    BoardResearcher,
    BraveSearchClient,
    BraveSearchError,
    ResearchContext,
    SearchResult,
)

pytestmark = pytest.mark.unit

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_SAMPLE_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Local LLMs in 2025",
                "url": "https://example.com/llms",
                "description": "A roundup of local LLM options.",
            },
            {
                "title": "Ollama Guide",
                "url": "https://example.com/ollama",
                "description": "How to run models with Ollama.",
            },
        ]
    }
}


@respx.mock
@pytest.mark.asyncio
async def test_client_parses_200_into_search_results() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json=_SAMPLE_RESPONSE))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    results = await client.search("local LLMs")
    assert len(results) == 2
    assert results[0] == SearchResult(
        title="Local LLMs in 2025",
        url="https://example.com/llms",
        description="A roundup of local LLM options.",
    )


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_401() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    client = BraveSearchClient("bad-key", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_500() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(500, json={}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_timeout() -> None:
    respx.get(_BRAVE_URL).mock(side_effect=httpx.TimeoutException("timed out"))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_returns_empty_list_when_no_results() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": []}}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    results = await client.search("obscure query")
    assert results == []
