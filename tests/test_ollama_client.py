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
from runtime.llm.clients import ChatMessage, ChatRequest, OllamaClient
from runtime.llm.clients.ollama_client import OllamaHostError, _timeout_for_call
from runtime.llm.telemetry import PRODUCTION_READ_TIMEOUT_S, collect_calls
from runtime.llm.timeouts import read_timeout_override


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
async def test_chat_response_schema_sets_format_to_schema_dict() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    schema = {
        "type": "object",
        "required": ["intent"],
        "properties": {"intent": {"type": "string"}},
    }
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
        response_format="json",
        response_schema=schema,
    )

    captured: dict[str, object] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "{\"intent\":\"ask\"}"}},
        )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        await client.chat(request)

    assert captured.get("format") == schema


@pytest.mark.asyncio
async def test_chat_response_format_json_without_schema_sets_string() -> None:
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
    assert request.response_schema is None


@pytest.mark.asyncio
async def test_chat_omits_think_by_default() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    captured: dict[str, object] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}}
        )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        await client.chat(request)

    assert "think" not in captured


@pytest.mark.asyncio
async def test_chat_forwards_think_false() -> None:
    # A thinking-capable model can burn its entire max_tokens budget on a
    # hidden reasoning channel and never emit content -- confirmed live
    # against gemma4:e2b-mlx (2026-08-22). think=False must reach Ollama's
    # request body exactly, not just live as a no-op on ChatRequest.
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
        think=False,
    )

    captured: dict[str, object] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}}
        )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        await client.chat(request)

    assert captured.get("think") is False


@pytest.mark.asyncio
async def test_chat_falls_back_to_thinking_when_content_empty_and_json_requested() -> None:
    # qwen3-vl:4b (live-confirmed 2026-08-22): think=False doesn't stop this
    # model from routing its structured answer into `thinking` instead of
    # `content`. done_reason="stop" -- not a truncation, the model considers
    # itself finished, `content` is just genuinely empty.
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="qwen3-vl:4b",
        messages=[ChatMessage(role="user", content="hi")],
        response_format="json",
        think=False,
    )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": '{"intent": "list_files", "confidence": 0.95}',
                    },
                    "done_reason": "stop",
                },
            )
        )
        resp = await client.chat(request)

    assert resp.content == '{"intent": "list_files", "confidence": 0.95}'


@pytest.mark.asyncio
async def test_chat_does_not_fall_back_to_thinking_in_text_mode() -> None:
    # `thinking` is the model's private reasoning -- never surface it to a
    # plain-text chat reply just because content came back empty.
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="qwen3-vl:4b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "internal reasoning, not for the user",
                    },
                },
            )
        )
        resp = await client.chat(request)

    assert resp.content == ""


@pytest.mark.asyncio
async def test_chat_prefers_content_over_thinking_when_both_present() -> None:
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="qwen3-vl:4b",
        messages=[ChatMessage(role="user", content="hi")],
        response_format="json",
    )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": '{"intent": "ping", "confidence": 0.9}',
                        "thinking": "some reasoning trace",
                    },
                },
            )
        )
        resp = await client.chat(request)

    assert resp.content == '{"intent": "ping", "confidence": 0.9}'


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


# --- Per-call telemetry -------------------------------------------------
# The eval harness needs to tell "model chose wrong" from "harness cut it off".
# See docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md.


@pytest.mark.asyncio
async def test_chat_records_provider_timings_to_active_collector() -> None:
    """Ollama already reports load/eval timings; they must reach the collector."""
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hello"},
                    "prompt_eval_count": 4,
                    "eval_count": 2,
                    "load_duration": 23_482_571_375,
                    "prompt_eval_duration": 2_840_904_000,
                    "eval_duration": 1_428_180_000,
                    "done_reason": "stop",
                },
            )
        )
        await client.chat(request)

    assert len(calls) == 1
    t = calls[0]
    assert t.provider == "ollama"
    assert t.model == "gemma4:e2b"
    assert t.load_ms == 23_482  # the cold-load tax, in the measured window
    assert t.prompt_eval_ms == 2_840
    assert t.eval_ms == 1_428
    assert t.done_reason == "stop"
    assert t.attempts == 1
    assert not t.timed_out


@pytest.mark.asyncio
async def test_chat_records_thinking_tokens_when_content_is_empty() -> None:
    """qwen3-vl:4b spends its whole budget on `thinking` and emits no content."""
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="qwen3-vl:4b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "", "thinking": "x" * 40},
                    "eval_count": 512,
                    "done_reason": "length",
                },
            )
        )
        await client.chat(request)

    t = calls[0]
    assert t.tokens_out == 512
    assert t.thinking_tokens > 0
    assert t.truncated_by_budget


@pytest.mark.asyncio
async def test_retry_exhausted_timeout_is_recorded_as_timed_out() -> None:
    """Three read timeouts must surface as one timed-out record, not silence.

    This is the failure that currently reads as 'expected call never found',
    identical to a model that answered fast and wrongly.
    """
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="qwen3-vl:4b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            side_effect=httpx.ReadTimeout("read timeout")
        )
        with pytest.raises(httpx.ReadTimeout):
            await client.chat(request)

    assert len(calls) == 1
    t = calls[0]
    assert t.timed_out
    assert t.attempts == 3
    assert t.model == "qwen3-vl:4b"


@pytest.mark.asyncio
async def test_chat_without_collector_still_returns_normally() -> None:
    """Instrumentation must never be load-bearing for the runtime path."""
    cfg = _config_with_base("http://127.0.0.1:11434")
    client = OllamaClient(cfg)
    request = ChatRequest(
        model="gemma4:e2b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"role": "assistant", "content": "hello"}}
            )
        )
        resp = await client.chat(request)

    assert resp.content == "hello"


# --- Eval-local read-timeout override -----------------------------------
# The eval harness measures capability under a generous budget; the runtime
# keeps its shipped 30s. Production behaviour must not move.


def test_default_read_timeout_is_the_shipped_production_value() -> None:
    assert _timeout_for_call().read == PRODUCTION_READ_TIMEOUT_S


def test_override_widens_only_the_read_timeout() -> None:
    base = _timeout_for_call()
    with read_timeout_override(600.0):
        widened = _timeout_for_call()
        assert widened.read == 600.0
        # connect/write/pool are not measurement knobs -- they must not drift.
        assert widened.connect == base.connect
        assert widened.write == base.write
        assert widened.pool == base.pool


def test_override_is_reverted_on_exit() -> None:
    with read_timeout_override(600.0):
        pass
    assert _timeout_for_call().read == PRODUCTION_READ_TIMEOUT_S


def test_override_is_reverted_even_if_the_body_raises() -> None:
    with pytest.raises(RuntimeError), read_timeout_override(600.0):
        raise RuntimeError("boom")
    assert _timeout_for_call().read == PRODUCTION_READ_TIMEOUT_S
