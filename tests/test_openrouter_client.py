"""Offline tests for the OpenRouter client.

Mocks all network with respx. Verifies TLS-only refusal of `http://`, missing
key behaviour, header injection (Authorization, HTTP-Referer, X-Title), 401
classification, and 5xx retry.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from runtime.config import AegisConfig, ProviderConfig, get_config
from runtime.llm.clients import ChatMessage, ChatRequest, OpenRouterClient
from runtime.llm.clients import openrouter_client as _orc
from runtime.llm.clients.openrouter_client import (
    OpenRouterAuthError,
    OpenRouterConfigError,
)
from runtime.llm.telemetry import collect_calls


def _config(api_key: str | None, base_url: str = "https://openrouter.ai/api/v1") -> AegisConfig:
    base = get_config()
    return base.model_copy(
        update={
            "providers": ProviderConfig(
                openrouter_base_url=base_url,
                openrouter_api_key=api_key,
            )
        }
    )


def test_openrouter_requires_api_key() -> None:
    with pytest.raises(OpenRouterConfigError):
        OpenRouterClient(_config(api_key=None))


def test_openrouter_refuses_http_scheme() -> None:
    with pytest.raises(OpenRouterConfigError):
        OpenRouterClient(_config(api_key="sk-test", base_url="http://openrouter.ai/api/v1"))


@pytest.mark.asyncio
async def test_chat_injects_attribution_headers() -> None:
    client = OpenRouterClient(_config(api_key="sk-test"))
    request = ChatRequest(
        model="minimax/minimax-m2.7",
        messages=[ChatMessage(role="user", content="hi")],
    )

    captured: dict[str, str] = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured.update(dict(req.headers))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=_capture)
        resp = await client.chat(request)

    assert resp.content == "hello"
    assert resp.tokens_in == 5
    assert resp.tokens_out == 3
    assert captured["authorization"] == "Bearer sk-test"
    assert captured["http-referer"].startswith("https://")
    assert captured["x-title"] == "AEGIS"


@pytest.mark.asyncio
async def test_chat_classifies_401_as_auth_error() -> None:
    client = OpenRouterClient(_config(api_key="sk-bad"))
    request = ChatRequest(
        model="minimax/minimax-m2.7",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        route = mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "invalid key"})
        )
        with pytest.raises(OpenRouterAuthError):
            await client.chat(request)
        assert route.call_count == 1  # auth errors do not retry


@pytest.mark.asyncio
async def test_chat_retries_on_5xx_then_succeeds() -> None:
    client = OpenRouterClient(_config(api_key="sk-test"))
    request = ChatRequest(
        model="minimax/minimax-m2.7",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock:
        route = mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(502, json={"error": "bad gateway"}),
                httpx.Response(
                    200,
                    json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                ),
            ]
        )
        resp = await client.chat(request)

    assert route.call_count == 2
    assert resp.content == "ok"


def test_repr_does_not_leak_api_key() -> None:
    cfg = _config(api_key="sk-secret-12345")
    # ProviderConfig has repr=False on the key — verify the secret is not surfaced.
    assert "sk-secret-12345" not in repr(cfg.providers)


@pytest.mark.asyncio
async def test_chat_strips_invalid_openrouter_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``openrouter/x-ai/grok-4.1-fast`` is a config-side mistake (returns 400).

    The client should silently fix the wire payload to ``x-ai/grok-4.1-fast``
    so the user's run still completes, AND log a warning so the canonical
    config gets corrected.
    """
    # Reset the once-per-process warning set so this test is order-independent.
    _orc._WARNED_PREFIXES.clear()

    client = OpenRouterClient(_config(api_key="sk-test"))
    request = ChatRequest(
        model="openrouter/x-ai/grok-4.1-fast",
        messages=[ChatMessage(role="user", content="hi")],
    )

    captured_payload: dict[str, object] = {}

    def _capture(req: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    with (
        caplog.at_level("WARNING", logger="runtime.llm.clients.openrouter_client"),
        respx.mock(assert_all_called=True) as mock,
    ):
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=_capture
        )
        resp = await client.chat(request)

    assert captured_payload["model"] == "x-ai/grok-4.1-fast"
    assert resp.model == "x-ai/grok-4.1-fast"
    assert any("openrouter/" in r.message for r in caplog.records)


# --- Per-call telemetry -------------------------------------------------
# Without this, OpenRouter runs land in the eval JSON with telemetry=None,
# which `tgc_within_budget` treats as "within budget" -- silently scoring
# every OpenRouter pass as fast. See
# docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md.


@pytest.mark.asyncio
async def test_chat_records_usage_and_finish_reason() -> None:
    cfg = _config("sk-test")
    client = OpenRouterClient(cfg)
    request = ChatRequest(
        model="qwen/qwen3.5-9b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            )
        )
        await client.chat(request)

    assert len(calls) == 1
    t = calls[0]
    assert t.provider == "openrouter"
    assert t.model == "qwen/qwen3.5-9b"
    assert t.tokens_in == 7
    assert t.tokens_out == 3
    assert t.done_reason == "stop"
    assert not t.timed_out


@pytest.mark.asyncio
async def test_length_finish_reason_marks_the_call_truncated() -> None:
    """Parity with Ollama: budget truncation must be visible to the taxonomy."""
    cfg = _config("sk-test")
    client = OpenRouterClient(cfg)
    request = ChatRequest(
        model="qwen/qwen3.5-9b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 512},
                },
            )
        )
        await client.chat(request)

    assert calls[0].truncated_by_budget


@pytest.mark.asyncio
async def test_retry_exhausted_timeout_is_recorded() -> None:
    cfg = _config("sk-test")
    client = OpenRouterClient(cfg)
    request = ChatRequest(
        model="qwen/qwen3.5-9b",
        messages=[ChatMessage(role="user", content="hi")],
    )

    with respx.mock() as mock, collect_calls() as calls:
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout("read timeout")
        )
        with pytest.raises(httpx.ReadTimeout):
            await client.chat(request)

    assert len(calls) == 1
    assert calls[0].timed_out
    assert calls[0].attempts == 3
