"""Per-call model telemetry collection.

The eval harness needs to tell "the model could not do it" apart from "the
harness cut it off" (docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md).
That distinction lives in timings the Ollama response already returns and the
runtime currently discards -- load_duration, eval_count, done_reason -- plus
whether the call exhausted its retries on a read timeout.

Collection is context-local and opt-in: with no active collector, recording is
a no-op, so the production path is unchanged.
"""
from __future__ import annotations

import asyncio

import pytest

from runtime.llm.telemetry import CallTelemetry, collect_calls, record_call

pytestmark = pytest.mark.unit


def _telemetry(**overrides: object) -> CallTelemetry:
    base: dict[str, object] = {
        "provider": "ollama",
        "model": "gemma4:e2b-mlx",
        "wall_ms": 1200,
    }
    base.update(overrides)
    return CallTelemetry(**base)  # type: ignore[arg-type]


def test_record_without_active_collector_is_a_noop() -> None:
    """The production path must not need a collector -- recording just drops."""
    record_call(_telemetry())  # must not raise


def test_collector_captures_calls_made_inside_the_block() -> None:
    with collect_calls() as calls:
        record_call(_telemetry(model="a"))
        record_call(_telemetry(model="b"))

    assert [c.model for c in calls] == ["a", "b"]


def test_collector_does_not_capture_calls_made_after_the_block() -> None:
    with collect_calls() as calls:
        record_call(_telemetry(model="inside"))
    record_call(_telemetry(model="outside"))

    assert [c.model for c in calls] == ["inside"]


def test_nested_collectors_do_not_leak_into_each_other() -> None:
    with collect_calls() as outer:
        record_call(_telemetry(model="outer-1"))
        with collect_calls() as inner:
            record_call(_telemetry(model="inner-1"))
        record_call(_telemetry(model="outer-2"))

    assert [c.model for c in inner] == ["inner-1"]
    assert [c.model for c in outer] == ["outer-1", "outer-2"]


async def test_concurrent_tasks_get_independent_collectors() -> None:
    """Variants may run concurrently; one variant's calls must not bleed into another's."""

    async def _run(tag: str, delay: float) -> list[str]:
        with collect_calls() as calls:
            record_call(_telemetry(model=f"{tag}-1"))
            await asyncio.sleep(delay)
            record_call(_telemetry(model=f"{tag}-2"))
        return [c.model for c in calls]

    left, right = await asyncio.gather(_run("L", 0.02), _run("R", 0.01))

    assert left == ["L-1", "L-2"]
    assert right == ["R-1", "R-2"]


def test_timed_out_call_is_distinguishable_from_a_completed_one() -> None:
    """A retry-exhausted timeout is the failure mode that currently masquerades
    as 'model engaged and chose wrong' in eval JSON."""
    completed = _telemetry(attempts=1, timed_out=False)
    exhausted = _telemetry(attempts=3, timed_out=True, wall_ms=90_600)

    assert not completed.timed_out
    assert exhausted.timed_out
    assert exhausted.attempts == 3


def test_thinking_token_share_reports_budget_spent_on_hidden_reasoning() -> None:
    """qwen3-vl:4b burned 512/512 tokens on `thinking` with empty content --
    the measurement that drives the deferred thinking-mode decision."""
    all_thinking = _telemetry(tokens_out=512, thinking_tokens=512)
    no_thinking = _telemetry(tokens_out=512, thinking_tokens=0)

    assert all_thinking.thinking_token_share == pytest.approx(1.0)
    assert no_thinking.thinking_token_share == pytest.approx(0.0)


def test_thinking_token_share_is_zero_when_nothing_was_generated() -> None:
    """No division-by-zero on a call that produced no output tokens."""
    assert _telemetry(tokens_out=0, thinking_tokens=0).thinking_token_share == 0.0


def test_truncated_by_budget_flags_a_response_cut_off_mid_generation() -> None:
    """done_reason='length' means the model never finished -- a budget artifact,
    not a model decision."""
    assert _telemetry(done_reason="length").truncated_by_budget
    assert not _telemetry(done_reason="stop").truncated_by_budget
    assert not _telemetry(done_reason=None).truncated_by_budget
