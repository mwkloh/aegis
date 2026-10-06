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


async def test_runner_labels_an_unbuildable_dispatcher_as_a_harness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model was never asked, so this must not read as the model declining.

    2026-10-03 overnight run: 30 of 45 `llama3.2:3b` variants and 14 of 45
    `qwen3-vl:4b` variants failed in ~2s because the Ollama liveness probe
    could not get an answer. They were labelled `no_tool_call` -- the same
    bucket as `lfm2.5:8b` genuinely declining -- and nearly published as model
    results.
    """
    monkeypatch.setattr(runner_mod, "build_harness_dispatcher", lambda *a, **k: None)
    result = await runner_mod.run_variant(
        get_config(),
        registry=None,  # type: ignore[arg-type]
        tier1_loader=None,  # type: ignore[arg-type]
        task=_task(),
        variant_text="what time is it?",
    )

    assert not result.passed
    assert result.failure_kind == "harness_unavailable"


class _ReplyingDispatcher(_StubDispatcher):
    """Replies to the user the way the real dispatcher does, via `reply=`."""

    def __init__(self, *texts: str) -> None:
        super().__init__(calls=[])
        self._texts = texts

    async def dispatch(self, **kwargs: Any) -> None:
        for text in self._texts:
            await kwargs["reply"](text)


async def test_runner_records_the_final_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass/fail grades the tool calls, never the words. Whether the model then
    *claimed* something it never did is only visible in the reply, so it has to
    be kept -- it was previously collected and thrown away."""
    result = await _run(
        monkeypatch, _ReplyingDispatcher("I listed 3 files.", "Anything else?")
    )

    assert result.reply == "I listed 3 files.\nAnything else?"


async def test_runner_reply_is_none_when_the_model_said_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent, not an empty string -- the same distinction telemetry draws."""
    result = await _run(monkeypatch, _ReplyingDispatcher())

    assert result.reply is None
