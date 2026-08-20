"""Ordered subsequence grading — did the run make the expected tool calls, in order?

No LLM-judge, no reply-text parsing. Matches this codebase's existing
philosophy of trusting structural evidence over self-reported text (see
`HarnessDispatcher._gate_completion`).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

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


def grade_calls(
    expected_calls: tuple[ExpectedCall, ...], actual_calls: list[CallRecord]
) -> GradeResult:
    """Ordered subsequence match: each expected call must match some actual
    call at or after the position of the previous match. Incidental extra
    calls between matches are tolerated. A call with status "error" never
    counts as satisfying an expected call.
    """
    cursor = 0
    for expected in expected_calls:
        found = False
        while cursor < len(actual_calls):
            tool, args, status = actual_calls[cursor]
            cursor += 1
            if tool == expected.tool and status != "error" and _args_match(
                expected.args_match, args
            ):
                found = True
                break
        if not found:
            return GradeResult(
                passed=False,
                reason=(
                    f"expected call to {expected.tool!r} with args matching "
                    f"{expected.args_match!r} never found"
                ),
            )
    return GradeResult(passed=True)


__all__ = ["CallRecord", "GradeResult", "grade_calls"]
