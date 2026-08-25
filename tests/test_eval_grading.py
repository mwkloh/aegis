"""Ordered subsequence grading of observed tool calls against expected_calls."""
from __future__ import annotations

import pytest

from runtime.eval.grading import FailureKind, classify_failure, grade_calls
from runtime.eval.report import VariantTelemetry
from runtime.eval.tasks import ExpectedCall

pytestmark = pytest.mark.unit


def test_exact_match_passes() -> None:
    expected = (ExpectedCall(tool="files_search", args_match={"glob": "*CT-001*"}),)
    actual = [("files_search", {"glob": "*CT-001*", "root": "."}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is True


def test_missing_call_fails() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_search", {"glob": "*.md"}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is False
    assert "files_read" in result.reason


def test_wrong_args_fails() -> None:
    expected = (ExpectedCall(tool="files_search", args_match={"glob": "*CT-001*"}),)
    actual = [("files_search", {"glob": "*other*"}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_error_status_does_not_count() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_read", {"path": "/tmp/x"}, "error")]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_order_matters() -> None:
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    # files_read happens BEFORE files_search -- wrong order.
    actual = [
        ("files_read", {"path": "/tmp/x"}, "ok"),
        ("files_search", {"glob": "*.md"}, "ok"),
    ]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_incidental_extra_calls_are_tolerated() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [
        ("time", {}, "ok"),  # incidental, unrelated call
        ("files_read", {"path": "/tmp/x"}, "ok"),
    ]
    result = grade_calls(expected, actual)
    assert result.passed is True


def test_empty_args_match_matches_any_args() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_read", {"path": "/tmp/anything"}, "ok")]
    assert grade_calls(expected, actual).passed is True


def test_non_string_args_match_by_equality() -> None:
    expected = (ExpectedCall(tool="tier2_compress", args_match={"limit": 5}),)
    assert grade_calls(expected, [("tier2_compress", {"limit": 5}, "ok")]).passed is True
    assert grade_calls(expected, [("tier2_compress", {"limit": 6}, "ok")]).passed is False


# --- Failure taxonomy ---------------------------------------------------
# One "0% TGC" hides several unrelated mechanisms. See
# docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md.


def _telemetry(*, timed_out: bool = False, calls: int = 1) -> VariantTelemetry:
    return VariantTelemetry(
        model_calls=calls,
        load_ms_total=0,
        eval_ms_total=0,
        max_call_wall_ms=90_600 if timed_out else 1_000,
        timed_out_calls=1 if timed_out else 0,
        max_thinking_token_share=0.0,
    )


def test_passing_run_has_no_failure_kind() -> None:
    expected = (ExpectedCall(tool="time", args_match={}),)
    actual = [("time", {}, "ok")]
    assert classify_failure(expected, actual, _telemetry()) is None


def test_retry_exhausted_timeout_outranks_the_absence_of_tool_calls() -> None:
    """qwen3-vl:4b: no calls, but only because nothing ever came back.

    Must not be reported as a decision not to call a tool.
    """
    expected = (ExpectedCall(tool="time", args_match={}),)
    kind = classify_failure(expected, [], _telemetry(timed_out=True))
    assert kind is FailureKind.TIMEOUT_EXHAUSTED


def test_responded_without_calling_a_tool() -> None:
    """lfm2.5:8b: healthy, fast, well-formed -- and declines to use a tool."""
    expected = (ExpectedCall(tool="time", args_match={}),)
    kind = classify_failure(expected, [], _telemetry())
    assert kind is FailureKind.NO_TOOL_CALL


def test_called_something_else_entirely() -> None:
    expected = (ExpectedCall(tool="time", args_match={}),)
    actual = [("files_list", {"path": "/tmp"}, "ok")]
    assert classify_failure(expected, actual, _telemetry()) is FailureKind.WRONG_TOOL


def test_right_tool_but_every_attempt_errored() -> None:
    """Distinct from wrong_tool: the plan was right, execution failed."""
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_read", {"path": "/nope"}, "error")]
    assert classify_failure(expected, actual, _telemetry()) is FailureKind.TOOL_ERRORED


def test_first_step_landed_but_the_chain_stopped() -> None:
    """Gemma's stop-after-step-one shape."""
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    actual = [("files_search", {"directory": "/tmp"}, "ok")]
    kind = classify_failure(expected, actual, _telemetry())
    assert kind is FailureKind.INCOMPLETE_CHAIN


def test_repeating_a_successful_step_is_its_own_shape() -> None:
    """llama-3.1-8b-instruct, 2026-08-23: found the file, then re-searched five
    times and never read it. Success-repetition, not failure-blindness --
    reported separately from a chain that simply stopped."""
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    same = ("files_search", {"directory": "/tmp", "pattern": "CT-001"}, "ok")
    actual = [same, same, same, same, same]
    kind = classify_failure(expected, actual, _telemetry())
    assert kind is FailureKind.REPEATED_STEP


def test_two_repeats_is_not_yet_repetition() -> None:
    """Guards the threshold: a single retry is normal, not a pathology."""
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    same = ("files_search", {"directory": "/tmp"}, "ok")
    kind = classify_failure(expected, [same, same], _telemetry())
    assert kind is FailureKind.INCOMPLETE_CHAIN


def test_missing_telemetry_still_classifies_from_observed_calls() -> None:
    """Historical results have no telemetry; they must still bucket sensibly."""
    expected = (ExpectedCall(tool="time", args_match={}),)
    assert classify_failure(expected, [], None) is FailureKind.NO_TOOL_CALL


def test_truncated_thinking_is_not_the_same_as_declining_a_tool() -> None:
    """qwen3-vl:4b vs lfm2.5:8b, measured 2026-08-24 with the timeout lifted.

    Both make zero tool calls and neither times out, but the mechanisms differ:
    qwen3-vl thinks until it hits the token ceiling (truncated, 161s of eval,
    no content ever emitted); lfm2.5 finishes cleanly in 15s and still chooses
    not to call a tool. Only the second is a decision.
    """
    expected = (ExpectedCall(tool="time", args_match={}),)
    exhausted = VariantTelemetry(
        model_calls=3,
        load_ms_total=24_332,
        eval_ms_total=161_467,
        max_call_wall_ms=130_593,
        timed_out_calls=0,
        max_thinking_token_share=1.0,
        truncated_calls=2,
    )
    declined = VariantTelemetry(
        model_calls=3,
        load_ms_total=7_925,
        eval_ms_total=1_794,
        max_call_wall_ms=6_985,
        timed_out_calls=0,
        max_thinking_token_share=1.0,
        truncated_calls=0,
    )

    assert (
        classify_failure(expected, [], exhausted)
        is FailureKind.THINKING_BUDGET_EXHAUSTED
    )
    assert classify_failure(expected, [], declined) is FailureKind.NO_TOOL_CALL


def test_a_timeout_still_outranks_budget_exhaustion() -> None:
    expected = (ExpectedCall(tool="time", args_match={}),)
    both = VariantTelemetry(
        model_calls=1,
        load_ms_total=0,
        eval_ms_total=0,
        max_call_wall_ms=90_600,
        timed_out_calls=1,
        max_thinking_token_share=1.0,
        truncated_calls=1,
    )
    assert classify_failure(expected, [], both) is FailureKind.TIMEOUT_EXHAUSTED


def test_truncation_alongside_real_tool_calls_is_not_budget_exhaustion() -> None:
    """Truncation only explains the failure when nothing was called at all."""
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    tel = VariantTelemetry(
        model_calls=2,
        load_ms_total=0,
        eval_ms_total=0,
        max_call_wall_ms=5_000,
        timed_out_calls=0,
        max_thinking_token_share=0.5,
        truncated_calls=1,
    )
    kind = classify_failure(expected, [("files_search", {}, "ok")], tel)
    assert kind is FailureKind.INCOMPLETE_CHAIN
