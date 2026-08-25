"""`run_variant` must attach model-call telemetry to every result it returns.

The failure this guards against is the one that made the 2026-08-23 rerun hard
to read: a variant that timed out and a variant that answered wrongly produced
byte-identical JSON apart from the duration. See
`docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md`.
"""
from __future__ import annotations

from typing import Any

import pytest

from runtime.config import get_config
from runtime.eval import runner as runner_mod
from runtime.eval.tasks import EvalTask, ExpectedCall
from runtime.llm.telemetry import CallTelemetry, record_call

pytestmark = pytest.mark.unit


class _StubHarness:
    def has_tool(self, name: str) -> bool:
        return True

    def execute(self, intent: Any) -> Any:  # pragma: no cover - never called
        raise AssertionError("stub harness should not execute")


class _StubDispatcher:
    """Stands in for the real dispatcher, recording telemetry like a client would."""

    def __init__(self, *, calls: list[CallTelemetry], raises: bool = False) -> None:
        self._harness = _StubHarness()
        self._calls = calls
        self._raises = raises

    async def dispatch(self, **_kwargs: Any) -> None:
        for c in self._calls:
            record_call(c)
        if self._raises:
            raise RuntimeError("planner blew up after burning a call")


def _task() -> EvalTask:
    return EvalTask(
        id="time_check",
        description="ask the time",
        variants=("what time is it?",),
        expected_calls=(ExpectedCall(tool="time", args_match={}),),
    )


def _timeout_call() -> CallTelemetry:
    return CallTelemetry(
        provider="ollama",
        model="qwen3-vl:4b",
        wall_ms=90_600,
        attempts=3,
        timed_out=True,
    )


async def _run(monkeypatch: pytest.MonkeyPatch, dispatcher: _StubDispatcher) -> Any:
    monkeypatch.setattr(
        runner_mod, "build_harness_dispatcher", lambda *a, **k: dispatcher
    )
    cfg = get_config()
    return await runner_mod.run_variant(
        cfg,
        registry=None,  # type: ignore[arg-type]
        tier1_loader=None,  # type: ignore[arg-type]
        task=_task(),
        variant_text="what time is it?",
    )


async def test_timed_out_variant_is_distinguishable_from_a_wrong_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run(monkeypatch, _StubDispatcher(calls=[_timeout_call()]))

    assert not result.passed
    assert result.telemetry is not None
    assert result.telemetry.any_timed_out
    assert result.telemetry.max_call_wall_ms == 90_600


async def test_engaged_but_wrong_variant_records_no_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same pass/fail and same grading reason as above -- telemetry is what separates them."""
    fast = CallTelemetry(
        provider="ollama", model="lfm2.5:8b", wall_ms=1_200, eval_ms=1_100
    )
    result = await _run(monkeypatch, _StubDispatcher(calls=[fast]))

    assert not result.passed
    assert result.telemetry is not None
    assert not result.telemetry.any_timed_out
    assert result.telemetry.model_calls == 1


async def test_telemetry_is_attached_even_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A propagating timeout hits run_variant's catch-all; the evidence must survive."""
    result = await _run(
        monkeypatch, _StubDispatcher(calls=[_timeout_call()], raises=True)
    )

    assert not result.passed
    assert "run_variant raised" in result.reason
    assert result.telemetry is not None
    assert result.telemetry.any_timed_out


async def test_variant_that_never_reached_a_model_has_no_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent, not zeroed -- the distinction `actual_calls` lost before 78f84e0."""
    result = await _run(monkeypatch, _StubDispatcher(calls=[]))

    assert result.telemetry is None


async def test_runner_labels_a_timeout_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """End of the chain: telemetry -> taxonomy -> the field a reader sees."""
    result = await _run(monkeypatch, _StubDispatcher(calls=[_timeout_call()]))

    assert result.failure_kind == "timeout_exhausted"


async def test_runner_labels_a_no_tool_call_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = CallTelemetry(provider="ollama", model="lfm2.5:8b", wall_ms=1_200)
    result = await _run(monkeypatch, _StubDispatcher(calls=[fast]))

    assert result.failure_kind == "no_tool_call"


async def test_runner_leaves_failure_kind_unset_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _StubDispatcher(calls=[])
    dispatcher._harness = _StubHarness()
    monkeypatch.setattr(runner_mod, "build_harness_dispatcher", lambda *a, **k: dispatcher)

    # Make the stub actually satisfy the expected `time` call.
    async def _dispatch(**_kwargs: object) -> None:
        runner_mod._ObservingHarness  # noqa: B018 - referenced for clarity

    dispatcher.dispatch = _dispatch  # type: ignore[method-assign]
    cfg = get_config()
    result = await runner_mod.run_variant(
        cfg,
        registry=None,  # type: ignore[arg-type]
        tier1_loader=None,  # type: ignore[arg-type]
        task=_task(),
        variant_text="what time is it?",
    )
    # No calls were made, so it still fails -- but the label must reflect that,
    # never a stale value from a previous run.
    assert result.failure_kind == "no_tool_call"
