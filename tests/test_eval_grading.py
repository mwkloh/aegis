"""Ordered subsequence grading of observed tool calls against expected_calls."""
from __future__ import annotations

import pytest

from runtime.eval.grading import grade_calls
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
