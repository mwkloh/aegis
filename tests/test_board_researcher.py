"""Unit tests for BraveSearchClient and BoardResearcher."""
from __future__ import annotations

import httpx
import pytest
import respx

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
    with pytest.raises(BraveSearchError, match="401"):
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


@respx.mock
@pytest.mark.asyncio
async def test_researcher_fetch_returns_context_on_success() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json=_SAMPLE_RESPONSE))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    researcher = BoardResearcher(client)
    ctx = await researcher.fetch("local LLMs")
    assert ctx is not None
    assert ctx.query == "local LLMs"
    assert len(ctx.results) == 2
    assert ctx.elapsed_ms >= 0


@respx.mock
@pytest.mark.asyncio
async def test_researcher_fetch_returns_none_on_api_failure() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(500, json={}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    researcher = BoardResearcher(client)
    ctx = await researcher.fetch("anything")
    assert ctx is None


def test_format_context_produces_numbered_block() -> None:
    ctx = ResearchContext(
        query="local LLMs",
        results=(
            SearchResult(
                title="Local LLMs in 2025",
                url="https://example.com/llms",
                description="A roundup.",
            ),
        ),
        elapsed_ms=42,
    )
    researcher = BoardResearcher(BraveSearchClient("k"))
    text = researcher.format_context(ctx)
    assert "[Research context — Brave Search]" in text
    assert "1. Local LLMs in 2025" in text
    assert "https://example.com/llms" in text
    assert "A roundup." in text
    assert "---" in text


def test_format_context_handles_multiple_results() -> None:
    results = tuple(
        SearchResult(title=f"T{i}", url=f"https://u{i}.com", description=f"D{i}")
        for i in range(5)
    )
    ctx = ResearchContext(query="q", results=results, elapsed_ms=0)
    researcher = BoardResearcher(BraveSearchClient("k"))
    text = researcher.format_context(ctx)
    for i in range(5):
        assert f"{i + 1}. T{i}" in text
