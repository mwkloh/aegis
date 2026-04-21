"""Offline tests for the Ollama client.

Network is mocked with respx — zero real I/O. Verifies localhost-only host
allowlist, retry-on-5xx, no-retry-on-4xx, and JSON parsing.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from runtime.config import AegisConfig, ProviderConfig, get_config
from runtime.model_router.clients import ChatMessage, ChatRequest, OllamaClient
from runtime.model_router.clients.ollama_client import OllamaHostError


def _config_with_base(url: str) -> AegisConfig:
    base = get_config()
    return base.model_copy(update={"providers": ProviderConfig(ollama_base_url=url)})


def test_ollama_refuses_remote_host() -> None:
    cfg = _config_with_base("http://evil.example.com:11434")
    with pytest.raises(OllamaHostError):
        OllamaClient(cfg)


def test_ollama_accepts_loopback_variants() -> None:
    for url in (
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ):
        OllamaClient(_config_with_base(url))


@pytest.mark.asyncio
async def test_chat_parses_response() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock(assert_all_called=True) as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hello"},
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                },
            )
        )
        resp = await client.chat(request)

    assert resp.content == "hello"
    assert resp.model == "gemma4:e2b"
    assert resp.tokens_in == 4
    assert resp.tokens_out == 2


def test_ollama_refuses_non_http_scheme() -> None:
    cfg = _config_with_base("file:///tmp/ollama")
    with pytest.raises(OllamaHostError):
        OllamaClient(cfg)


@pytest.mark.asyncio
async def test_chat_json_mode_sets_format() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
        response_format="json",
    )

    captured: dict[str, object] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "{\"ok\":true}"}},
        )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        await client.chat(request)

    assert captured.get("format") == "json"


@pytest.mark.asyncio
async def test_chat_retries_on_read_timeout_then_succeeds() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:11434/api/chat").mock(
            side_effect=[
                httpx.ReadTimeout("slow"),
                httpx.Response(
                    200,
                    json={"message": {"role": "assistant", "content": "ok"}},
                ),
            ]
        )
        resp = await client.chat(request)

    assert route.call_count == 2
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_chat_retries_on_5xx_then_succeeds() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:11434/api/chat").mock(
            side_effect=[
                httpx.Response(503, json={"error": "warming up"}),
                httpx.Response(
                    200,
                    json={"message": {"role": "assistant", "content": "ok"}},
                ),
            ]
        )
        resp = await client.chat(request)

    assert route.call_count == 2
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_chat_does_not_retry_on_4xx() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(404, json={"error": "model not found"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat(request)
        assert route.call_count == 1


@pytest.mark.asyncio
async def test_health_returns_true_on_200() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        assert await client.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_on_connection_error() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        assert await client.health() is False


@pytest.mark.asyncio
async def test_list_models_returns_names() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(
                200,
                json={"models": [{"name": "gemma4:e2b"}, {"name": "gemma4:e4b"}]},
            )
        )
        names = await client.list_models()
    assert names == ["gemma4:e2b", "gemma4:e4b"]
