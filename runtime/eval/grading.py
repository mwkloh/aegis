"""Ordered subsequence grading — did the run make the expected tool calls, in order?

No LLM-judge, no reply-text parsing. Matches this codebase's existing
philosophy of trusting structural evidence over self-reported text (see
`HarnessDispatcher._gate_completion`).
"""
from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from runtime.eval.report import VariantTelemetry
from runtime.eval.tasks import ExpectedCall

CallRecord = tuple[str, dict[str, Any], str]  # (tool, args, status)


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reason: str = ""


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected in actual
    return bool(expected == actual)


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(
        key in actual and _value_matches(value, actual[key])
        for key, value in expected.items()
    )


def _matched_prefix(
    expected_calls: tuple[ExpectedCall, ...], actual_calls: list[CallRecord]
) -> int:
    """How many leading expected calls were satisfied, in order.

    Shared by `grade_calls` and `classify_failure` so a run can never be graded
    by one rule and explained by another.
    """
    cursor = 0
    matched = 0
    for expected in expected_calls:
        found = False
        while cursor < len(actual_calls):
            tool, args, status = actual_calls[cursor]
            cursor += 1
            if (
                tool == expected.tool
                and status != "error"
                and _args_match(expected.args_match, args)
            ):
                found = True
                break
        if not found:
            break
        matched += 1
    return matched


def grade_calls(
    expected_calls: tuple[ExpectedCall, ...], actual_calls: list[CallRecord]
) -> GradeResult:
    """Ordered subsequence match: each expected call must match some actual
    call at or after the position of the previous match. Incidental extra
    calls between matches are tolerated. A call with status "error" never
    counts as satisfying an expected call.
    """
    matched = _matched_prefix(expected_calls, actual_calls)
    if matched == len(expected_calls):
        return GradeResult(passed=True)
    missing = expected_calls[matched]
    return GradeResult(
        passed=False,
        reason=(
            f"expected call to {missing.tool!r} with args matching "
            f"{missing.args_match!r} never found"
        ),
    )


class FailureKind(StrEnum):
    """Why a variant failed, as far as the recorded evidence can say.

    `reason` alone cannot answer this: a model that timed out and a model that
    answered fast and wrongly both produce "expected call ... never found".

    Deliberately absent: a split between "declined to call a tool" and "claimed
    done without engaging". Both reach this function as *no observed calls and
    no timeout* -- the planner's own `kind: "respond"` decision is not visible
    here, so the two collapse into `NO_TOOL_CALL`. Splitting them would need
    planner-level instrumentation; inventing the distinction from this evidence
    would be guessing.
    """

    TIMEOUT_EXHAUSTED = "timeout_exhausted"
    THINKING_BUDGET_EXHAUSTED = "thinking_budget_exhausted"
    NO_TOOL_CALL = "no_tool_call"
    TOOL_ERRORED = "tool_errored"
    WRONG_TOOL = "wrong_tool"
    REPEATED_STEP = "repeated_step"
    INCOMPLETE_CHAIN = "incomplete_chain"


_REPETITION_THRESHOLD: Final[int] = 3


def _fingerprint(tool: str, args: dict[str, Any]) -> tuple[str, str]:
    return tool, repr(sorted(args.items()))


def classify_failure(
    expected_calls: tuple[ExpectedCall, ...],
    actual_calls: list[CallRecord],
    telemetry: VariantTelemetry | None,
) -> FailureKind | None:
    """Bucket a failing variant. Returns `None` when the variant passed.

    Order matters: a retry-exhausted timeout is checked first, because it also
    presents as "no tool calls" and would otherwise be misread as a decision
    the model made.
    """
    if _matched_prefix(expected_calls, actual_calls) == len(expected_calls):
        return None

    if telemetry is not None and telemetry.any_timed_out:
        return FailureKind.TIMEOUT_EXHAUSTED

    if not actual_calls:
        # Truncation only explains a failure when nothing was called at all.
        # Measured 2026-08-24 with the eval timeout lifted: `qwen3-vl:4b` runs
        # 161s of generation, spends every token on `thinking`, hits the token
        # ceiling, and emits no content -- so no tool call is ever decoded.
        # `lfm2.5:8b` finishes cleanly in 15s and still declines. Same empty
        # call list, different mechanism; only the second is a decision.
        if telemetry is not None and telemetry.truncated_calls > 0:
            return FailureKind.THINKING_BUDGET_EXHAUSTED
        return FailureKind.NO_TOOL_CALL

    matched = _matched_prefix(expected_calls, actual_calls)
    if matched == 0:
        next_tool = expected_calls[0].tool if expected_calls else None
        attempted = any(
            tool == next_tool and status == "error" for tool, _args, status in actual_calls
        )
        return FailureKind.TOOL_ERRORED if attempted else FailureKind.WRONG_TOOL

    counts = Counter(_fingerprint(tool, args) for tool, args, _s in actual_calls)
    if counts and max(counts.values()) >= _REPETITION_THRESHOLD:
        return FailureKind.REPEATED_STEP
    return FailureKind.INCOMPLETE_CHAIN


__all__ = [
    "CallRecord",
    "FailureKind",
    "GradeResult",
    "classify_failure",
    "grade_calls",
]
