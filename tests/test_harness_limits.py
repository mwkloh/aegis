"""How much of a tool result the model gets to see.

The planner used to clip every result to 1,024 characters of its *JSON-encoded*
form, and the chain-synthesis prompt carried its own duplicate constant. Measured
2026-10-04: no rationale or measurement stood behind the number, the encoding cost
~12% of it (a newline is two characters, plus a `{"result": "` wrapper), the cut
landed mid-JSON, and prefill cost differs ~12x between models so one constant fits
none of them. See docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md
section 14.
"""
from __future__ import annotations

import pytest

from runtime.harness.contract import ToolResult
from runtime.harness.limits import (
    DEFAULT_TOOL_RESULT_CHARS,
    DEFAULT_TURN_RESULT_BUDGET_CHARS,
    MIN_RESULT_STUB_CHARS,
    allocate_result_limits,
    clip_text,
    result_text,
)

pytestmark = pytest.mark.unit


# --- result_text: what the model reads, decoded ---------------------------


def test_a_plain_result_is_the_text_not_its_json_encoding() -> None:
    """Encoding it would turn every newline into two characters and wrap it in
    `{"result": "..."}`, spending the clip on punctuation."""
    result = ToolResult(status="ok", payload={"result": "line one\nline two"})
    assert result_text(result) == "line one\nline two"


def test_a_structured_payload_is_still_json() -> None:
    result = ToolResult(status="ok", payload={"b": 2, "a": 1})
    assert result_text(result) == '{"a": 1, "b": 2}'


def test_an_empty_payload_reads_as_empty() -> None:
    assert result_text(ToolResult(status="ok", payload={})) == "(empty)"


def test_an_error_is_prefixed_and_never_empty() -> None:
    assert result_text(ToolResult(status="error", error="boom")) == "ERROR: boom"
    assert result_text(ToolResult(status="error")) == "ERROR: (no detail)"


# --- clip_text: say how much was cut, not just that something was ----------


def test_text_within_the_limit_is_untouched() -> None:
    assert clip_text("short", 100) == "short"
    assert clip_text("x" * 100, 100) == "x" * 100


def test_a_clipped_result_says_how_much_it_showed_and_how_much_there_was() -> None:
    clipped = clip_text("x" * 5000, 1000)

    assert clipped.startswith("x" * 1000)
    assert "showing 1,000 of 5,000 chars" in clipped


# --- allocate_result_limits: a per-turn budget, newest first --------------


def test_small_results_are_never_clipped() -> None:
    assert allocate_result_limits([50, 60], per_result=3500, total=8000) == [50, 60]


def test_each_result_is_capped_at_the_per_result_limit() -> None:
    assert allocate_result_limits([9000], per_result=3500, total=8000) == [3500]


def test_the_newest_result_gets_the_budget_first() -> None:
    """Three 5,000-char results against an 8,000-char budget: the newest two fit at
    3,500 and the oldest gets what is left."""
    assert allocate_result_limits([5000, 5000, 5000], per_result=3500, total=8000) == [
        1000,
        3500,
        3500,
    ]


def test_an_exhausted_budget_leaves_a_stub_not_nothing() -> None:
    limits = allocate_result_limits([5000] * 5, per_result=3500, total=8000)

    assert limits[-1] == 3500
    assert limits[0] == MIN_RESULT_STUB_CHARS
    assert all(limit >= MIN_RESULT_STUB_CHARS for limit in limits)


def test_no_results_allocates_nothing() -> None:
    assert allocate_result_limits([], per_result=3500, total=8000) == []


def test_a_short_result_does_not_consume_budget_it_does_not_use() -> None:
    assert allocate_result_limits([5000, 40], per_result=3500, total=3600) == [3500, 40]


def test_defaults_are_consistent() -> None:
    """The per-result default must fit inside the turn budget, or one result would
    starve every other."""
    assert DEFAULT_TOOL_RESULT_CHARS < DEFAULT_TURN_RESULT_BUDGET_CHARS
