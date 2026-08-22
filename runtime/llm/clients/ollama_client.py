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

from .base import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1"})
_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=5.0)
_HTTP_OK: Final[int] = 200
_HTTP_5XX_MIN: Final[int] = 500
_HTTP_6XX_MIN: Final[int] = 600


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
        data = await self._post_with_retry("/api/chat", payload)
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
                base_url=self._base_url, timeout=_TIMEOUT, follow_redirects=False
            ) as client:
                resp = await client.get("/api/tags")
                return resp.status_code == _HTTP_OK
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=_TIMEOUT, follow_redirects=False
        ) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        models = data.get("models", []) if isinstance(data, dict) else []
        return [str(m.get("name", "")) for m in models if isinstance(m, dict)]

    async def _post_with_retry(self, path: str, payload: dict[str, object]) -> dict[str, object]:
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
                async with httpx.AsyncClient(
                    base_url=self._base_url, timeout=_TIMEOUT, follow_redirects=False
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
