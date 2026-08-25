"""Async Ollama client.

Hard-wired to local hosts only. Any non-loopback host raises ValueError before
a single byte leaves the process — defence-in-depth against SSRF + accidental
remote-Ollama leakage.
"""
from __future__ import annotations

import logging
import time
from typing import Final
from urllib.parse import urlparse

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from runtime.config import AegisConfig
from runtime.llm.telemetry import (
    PRODUCTION_READ_TIMEOUT_S,
    CallTelemetry,
    record_call,
)
from runtime.llm.timeouts import resolve_read_timeout

from .base import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=2.0, read=PRODUCTION_READ_TIMEOUT_S, write=10.0, pool=5.0
)
_HTTP_OK: Final[int] = 200
_HTTP_5XX_MIN: Final[int] = 500
_HTTP_6XX_MIN: Final[int] = 600


def _timeout_for_call() -> httpx.Timeout:
    """Shipped timeout, unless the eval harness has widened `read`.

    Only `read` moves: `connect`/`write`/`pool` guard a wedged socket rather
    than a slow model, and widening those would hide real breakage.
    """
    return httpx.Timeout(
        connect=_TIMEOUT.connect,
        read=resolve_read_timeout(PRODUCTION_READ_TIMEOUT_S),
        write=_TIMEOUT.write,
        pool=_TIMEOUT.pool,
    )


class OllamaHostError(ValueError):
    """Raised when an Ollama base URL points at a non-loopback host."""


def _validate_local(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise OllamaHostError(f"ollama base_url must be http/https, got {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise OllamaHostError(
            f"ollama base_url host {host!r} is not loopback; refusing to connect"
        )
    return base_url.rstrip("/")


class OllamaClient:
    """Local Ollama HTTP client. Localhost-only; no retries on 4xx."""

    def __init__(self, config: AegisConfig) -> None:
        self._base_url = _validate_local(config.providers.ollama_base_url)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        elif request.response_format == "json":
            payload["format"] = "json"
        if request.think is not None:
            payload["think"] = request.think

        started = time.perf_counter()
        attempts: list[int] = [1]
        try:
            data = await self._post_with_retry("/api/chat", payload, attempts)
        except httpx.ReadTimeout:
            # A retry-exhausted timeout is the failure that otherwise reaches the
            # eval JSON as a bare "expected call never found" -- identical text to
            # a model that answered quickly and wrongly. Record it before
            # re-raising so the two stay distinguishable.
            record_call(
                CallTelemetry(
                    provider="ollama",
                    model=request.model,
                    wall_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempts[0],
                    timed_out=True,
                )
            )
            raise
        latency_ms = int((time.perf_counter() - started) * 1000)

        message_raw = data.get("message", {})
        content = message_raw.get("content", "") if isinstance(message_raw, dict) else ""
        wants_structured = (
            request.response_schema is not None or request.response_format == "json"
        )
        if not content and wants_structured and isinstance(message_raw, dict):
            # Some models/templates route structured output into `thinking`
            # instead of `content` even with think=False in the request --
            # confirmed live on qwen3-vl:4b (2026-08-22): done_reason="stop",
            # content empty, thinking holds a valid JSON reply matching the
            # requested schema. Scoped to JSON/schema requests only -- for a
            # free-text chat reply, `thinking` is the model's private
            # reasoning, not something to surface to the user, so this must
            # never fall back there.
            thinking = message_raw.get("thinking")
            if thinking:
                logger.info(
                    "ollama_client.content_from_thinking_fallback",
                    extra={"model": request.model},
                )
                content = thinking
        tokens_in = _coerce_int(data.get("prompt_eval_count", 0))
        tokens_out = _coerce_int(data.get("eval_count", 0))

        record_call(
            CallTelemetry(
                provider="ollama",
                model=request.model,
                wall_ms=latency_ms,
                load_ms=_ns_to_ms(data.get("load_duration")),
                prompt_eval_ms=_ns_to_ms(data.get("prompt_eval_duration")),
                eval_ms=_ns_to_ms(data.get("eval_duration")),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                thinking_tokens=_thinking_tokens(message_raw, tokens_out),
                done_reason=_coerce_done_reason(data.get("done_reason")),
                attempts=attempts[0],
            )
        )

        return ChatResponse(
            content=str(content),
            model=request.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=_timeout_for_call(), follow_redirects=False
            ) as client:
                resp = await client.get("/api/tags")
                return resp.status_code == _HTTP_OK
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=_timeout_for_call(), follow_redirects=False
        ) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = data.get("models", []) if isinstance(data, dict) else []
        return [str(m.get("name", "")) for m in models if isinstance(m, dict)]

    async def _post_with_retry(
        self, path: str, payload: dict[str, object], attempts: list[int] | None = None
    ) -> dict[str, object]:
        """POST with bounded retries.

        `attempts` is an optional out-parameter: when given, its single element
        is kept up to date with the number of attempts consumed, so a caller
        can still report the retry count on the path where tenacity re-raises.
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.ReadTimeout, _RetryableStatus)
            ),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                if attempts is not None:
                    attempts[0] = attempt.retry_state.attempt_number
                async with httpx.AsyncClient(
                    base_url=self._base_url, timeout=_timeout_for_call(), follow_redirects=False
                ) as client:
                    resp = await client.post(path, json=payload)
                    if _HTTP_5XX_MIN <= resp.status_code < _HTTP_6XX_MIN:
                        raise _RetryableStatus(resp.status_code)
                    resp.raise_for_status()
                    body = resp.json()
                    if not isinstance(body, dict):
                        raise ValueError(f"ollama returned non-object body: {type(body).__name__}")
                    return body
        raise RuntimeError("unreachable: tenacity exhausted without raising")


class _RetryableStatus(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"retryable status {status}")
        self.status = status


def _coerce_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _ns_to_ms(value: object) -> int:
    """Ollama reports durations in nanoseconds; telemetry stores milliseconds."""
    return _coerce_int(value) // 1_000_000


def _coerce_done_reason(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _thinking_tokens(message: object, tokens_out: int) -> int:
    """Approximate how many generated tokens went to the hidden reasoning channel.

    Ollama's `eval_count` covers thinking and content together and it does not
    break them out, so this splits `tokens_out` by the character ratio between
    the two fields. It is an estimate, and only ever used to answer a coarse
    question -- "did this model spend its whole budget thinking?" -- which is
    exactly the shape of the qwen3-vl:4b finding (512/512 tokens on `thinking`,
    empty content). Do not read it as an exact token count.
    """
    if not isinstance(message, dict) or tokens_out <= 0:
        return 0
    thinking = message.get("thinking")
    if not isinstance(thinking, str) or not thinking:
        return 0
    content = message.get("content")
    content_len = len(content) if isinstance(content, str) else 0
    total_len = len(thinking) + content_len
    if total_len <= 0:
        return 0
    return round(tokens_out * len(thinking) / total_len)
