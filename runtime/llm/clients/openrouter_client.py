"""Async OpenRouter client.

TLS-only. Refuses any base URL whose scheme is not `https`. Key is read once
from `AegisConfig` and never logged. Honours OpenRouter's `HTTP-Referer` and
`X-Title` attribution headers.
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

_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(connect=2.0, read=30.0, write=10.0, pool=5.0)
_REFERER: Final[str] = "https://github.com/aegis-local/aegis"
_TITLE: Final[str] = "AEGIS"
_HTTP_OK: Final[int] = 200
_HTTP_5XX_MIN: Final[int] = 500
_HTTP_6XX_MIN: Final[int] = 600
_AUTH_FAIL_CODES: Final[frozenset[int]] = frozenset({401, 403})
_BAD_PREFIX: Final[str] = "openrouter/"

_log = logging.getLogger(__name__)
_WARNED_PREFIXES: set[str] = set()


def _normalise_model_id(model: str) -> str:
    """Strip a leading ``openrouter/`` segment and warn once per offending id.

    OpenRouter's catalog uses ``<vendor>/<model>`` ids (e.g. ``x-ai/grok-4.1-fast``).
    A leading ``openrouter/`` is a config-side mistake that returns 400 from
    ``/chat/completions``. Defence-in-depth: silently fix the request, but log
    once so the canonical config still gets corrected.
    """
    if not model.startswith(_BAD_PREFIX):
        return model
    fixed = model[len(_BAD_PREFIX):]
    if model not in _WARNED_PREFIXES:
        _WARNED_PREFIXES.add(model)
        _log.warning(
            "openrouter: stripping invalid 'openrouter/' prefix from model id "
            "%r -> %r (fix MODEL_CODING/MODEL_SMART in .env to silence this)",
            model, fixed,
        )
    return fixed


class OpenRouterConfigError(ValueError):
    """Raised when OpenRouter cannot be configured (no key, bad URL)."""


class OpenRouterAuthError(RuntimeError):
    """Raised when OpenRouter rejects the API key."""


def _validate_https(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme != "https":
        raise OpenRouterConfigError(
            f"openrouter base_url must use https, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise OpenRouterConfigError("openrouter base_url is missing a host")
    return base_url.rstrip("/")


class OpenRouterClient:
    """Frontier-model client. TLS-only, header-attributed, key-redacted."""

    def __init__(self, config: AegisConfig) -> None:
        if not config.providers.openrouter_api_key:
            raise OpenRouterConfigError("OPENROUTER_API_KEY is not set")
        self._base_url = _validate_https(config.providers.openrouter_base_url)
        self._api_key = config.providers.openrouter_api_key

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        wire_model = _normalise_model_id(request.model)
        payload: dict[str, object] = {
            "model": wire_model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data = await self._post_with_retry("/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)

        choices_raw = data.get("choices", [])
        first: dict[str, object] = {}
        if isinstance(choices_raw, list) and choices_raw and isinstance(choices_raw[0], dict):
            first = choices_raw[0]
        message_raw = first.get("message", {})
        content = message_raw.get("content", "") if isinstance(message_raw, dict) else ""
        usage_raw = data.get("usage", {})
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        tokens_in = _coerce_int(usage.get("prompt_tokens", 0))
        tokens_out = _coerce_int(usage.get("completion_tokens", 0))

        return ChatResponse(
            content=str(content),
            model=wire_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=_TIMEOUT,
                follow_redirects=False,
                headers=self._headers(),
            ) as client:
                resp = await client.get("/models")
                return resp.status_code == _HTTP_OK
        except httpx.HTTPError:
            return False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": _REFERER,
            "X-Title": _TITLE,
            "Content-Type": "application/json",
        }

    async def _post_with_retry(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.4, min=0.4, max=4.0),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.ReadTimeout, _RetryableStatus)
            ),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    timeout=_TIMEOUT,
                    follow_redirects=False,
                    headers=self._headers(),
                ) as client:
                    resp = await client.post(path, json=payload)
                    if resp.status_code in _AUTH_FAIL_CODES:
                        raise OpenRouterAuthError(
                            f"openrouter rejected credentials (status {resp.status_code})"
                        )
                    if _HTTP_5XX_MIN <= resp.status_code < _HTTP_6XX_MIN:
                        raise _RetryableStatus(resp.status_code)
                    resp.raise_for_status()
                    body = resp.json()
                    if not isinstance(body, dict):
                        raise ValueError(
                            f"openrouter returned non-object body: {type(body).__name__}"
                        )
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
