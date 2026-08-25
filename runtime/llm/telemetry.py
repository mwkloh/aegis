"""Per-call model telemetry, collected context-locally and opt-in.

Ollama's non-streaming `/api/chat` response already reports `load_duration`,
`eval_count`, `eval_duration` and `done_reason`; the runtime has been throwing
all of it away. Without those numbers an eval failure reads only as "expected
call never found", which is the same text whether the model engaged and chose
wrong or never got a response at all.

Measured on 2026-08-24 (see
`docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md`): a cold load
costs 23.5 s against a 30 s read timeout, and three retries of that timeout
produce a ~90.6 s "failure" indistinguishable in the result JSON from a fast
wrong answer.

Collection is deliberately out-of-band. Threading a telemetry object back
through the reasoner and dispatcher layers would mean changing signatures the
runtime path shares with production callers; a `ContextVar` keeps the
instrumentation entirely inside the client and the eval harness. With no
collector active `record_call` is a no-op, so nothing changes for the runtime.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

_TRUNCATING_DONE_REASONS: Final[frozenset[str]] = frozenset({"length"})

PRODUCTION_READ_TIMEOUT_S: Final[float] = 30.0
"""The read timeout the runtime actually ships with.

Single source of truth, deliberately: `ollama_client` builds its `_TIMEOUT`
from it, and the eval report measures "would this have fit the real budget?"
against it. Two copies of the number would let the benchmark quietly grade
against a budget the product no longer has.
"""

_active: ContextVar[list[CallTelemetry] | None] = ContextVar(
    "llm_call_telemetry", default=None
)


class CallTelemetry(BaseModel):
    """One model call's cost and termination, as reported by the provider.

    Durations are milliseconds. `wall_ms` is measured by the client and spans
    every retry attempt; the provider-reported `load_ms`/`eval_ms` describe the
    final successful attempt only, so `wall_ms` is the one to compare against a
    timeout budget.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    wall_ms: int = Field(ge=0)

    load_ms: int = Field(default=0, ge=0)
    prompt_eval_ms: int = Field(default=0, ge=0)
    eval_ms: int = Field(default=0, ge=0)

    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    thinking_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "Output tokens spent on a hidden reasoning channel rather than "
            "content. A thinking-capable model can spend its entire budget "
            "here and emit no content at all."
        ),
    )

    done_reason: str | None = Field(
        default=None,
        description="Provider's termination reason; 'length' means budget-truncated.",
    )
    attempts: int = Field(
        default=1, ge=1, description="Retry attempts consumed, including the first."
    )
    timed_out: bool = Field(
        default=False,
        description="True when every attempt failed on a read timeout.",
    )

    @property
    def thinking_token_share(self) -> float:
        """Fraction of generated tokens spent thinking. 0.0 when nothing generated."""
        if self.tokens_out <= 0:
            return 0.0
        return min(self.thinking_tokens / self.tokens_out, 1.0)

    @property
    def truncated_by_budget(self) -> bool:
        """The response was cut off mid-generation rather than finishing."""
        return self.done_reason in _TRUNCATING_DONE_REASONS


def record_call(telemetry: CallTelemetry) -> None:
    """Append to the active collector, or do nothing if there isn't one."""
    sink = _active.get()
    if sink is not None:
        sink.append(telemetry)


@contextmanager
def collect_calls() -> Iterator[list[CallTelemetry]]:
    """Collect every `record_call` made inside this block.

    The returned list is populated in place. Nested and concurrent collectors
    stay independent -- each `ContextVar` token is reset on exit, and asyncio
    tasks copy the context at creation, so sibling variants never share a sink.
    """
    sink: list[CallTelemetry] = []
    token = _active.set(sink)
    try:
        yield sink
    finally:
        _active.reset(token)


__all__ = ["CallTelemetry", "collect_calls", "record_call"]
