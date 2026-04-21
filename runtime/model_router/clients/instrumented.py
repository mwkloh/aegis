"""Instrumented wrapper around any `ModelClient`.

Emits `model.call.start` / `model.call.end` events via an `EventStream` so the
Reflection plane can see latency, model, tier, and token counts. Never logs
the prompt body — only the model name and tier.
"""
from __future__ import annotations

import httpx

from runtime.events import EventStream, EventType

from .base import ChatRequest, ChatResponse, ModelClient


class InstrumentedModelClient:
    """Shim that records a start/end event around every `chat()` call."""

    def __init__(
        self, inner: ModelClient, events: EventStream, tier: str, provider: str
    ) -> None:
        self._inner = inner
        self._events = events
        self._tier = tier
        self._provider = provider

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self._events.append(
            EventType.MODEL_CALL_START,
            {
                "tier": self._tier,
                "provider": self._provider,
                "model": request.model,
                "prompt_tokens_estimate": _estimate_prompt_tokens(request),
            },
        )
        try:
            response = await self._inner.chat(request)
        except httpx.HTTPError as exc:
            self._events.append(
                EventType.MODEL_CALL_END,
                {
                    "tier": self._tier,
                    "provider": self._provider,
                    "model": request.model,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                },
            )
            raise

        self._events.append(
            EventType.MODEL_CALL_END,
            {
                "tier": self._tier,
                "provider": self._provider,
                "model": response.model,
                "status": "ok",
                "latency_ms": response.latency_ms,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
            },
        )
        return response

    async def health(self) -> bool:
        return bool(await self._inner.health())


def _estimate_prompt_tokens(request: ChatRequest) -> int:
    """Rough char/4 estimate — we never see a tokenizer on the hot path."""
    total_chars = sum(len(m.content) for m in request.messages)
    return max(1, total_chars // 4)
