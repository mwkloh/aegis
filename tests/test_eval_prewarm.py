"""Pre-warming the model so a cold load is not inside the measured window.

Measured 2026-08-24: a cold `load_duration` of 23.48 s against a 30 s read
timeout -- 78% of the budget spent before a single token. `/api/ps` showed no
resident models, and running configs back-to-back forces evictions, so this
tax lands unpredictably inside task durations.
"""
from __future__ import annotations

import pytest

from runtime.eval.prewarm import prewarm
from runtime.llm.clients.base import ChatRequest, ChatResponse
from runtime.llm.telemetry import CallTelemetry, collect_calls, record_call

pytestmark = pytest.mark.unit


class _Client:
    def __init__(self, *, load_ms: int = 23_482, fails: bool = False) -> None:
        self._load_ms = load_ms
        self._fails = fails
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if self._fails:
            raise RuntimeError("model not available")
        record_call(
            CallTelemetry(
                provider="ollama",
                model=request.model,
                wall_ms=self._load_ms + 100,
                load_ms=self._load_ms,
            )
        )
        return ChatResponse(content="ok", model=request.model)

    async def health(self) -> bool:  # pragma: no cover - protocol completeness
        return True


async def test_prewarm_reports_the_cold_load_it_absorbed() -> None:
    client = _Client(load_ms=23_482)
    observed = await prewarm(client, "qwen3-vl:4b")

    assert observed is not None
    assert observed.load_ms == 23_482


async def test_prewarm_asks_for_a_trivial_generation() -> None:
    """It exists to make the model resident, not to produce output."""
    client = _Client()
    await prewarm(client, "gemma4:e2b-mlx")

    assert len(client.requests) == 1
    assert client.requests[0].model == "gemma4:e2b-mlx"
    assert client.requests[0].max_tokens == 1


async def test_prewarm_never_raises_when_the_model_is_unavailable() -> None:
    """A failed warm-up must not abort a benchmark run."""
    assert await prewarm(_Client(fails=True), "nope:1b") is None


async def test_prewarm_cost_is_excluded_from_an_enclosing_measurement() -> None:
    """The whole point: the cold load must not land inside a measured variant."""
    client = _Client(load_ms=23_482)
    with collect_calls() as measured:
        await prewarm(client, "qwen3-vl:4b")
        record_call(
            CallTelemetry(provider="ollama", model="qwen3-vl:4b", wall_ms=1_200)
        )

    assert [c.wall_ms for c in measured] == [1_200]
