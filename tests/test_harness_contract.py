from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.harness import HarnessAdapter, ToolIntent

pytestmark = pytest.mark.unit


def test_intent_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ToolIntent(tool="echo", args={}, skill_id="echo", surprise=1)  # type: ignore[call-arg]


def test_intent_requires_non_empty_tool() -> None:
    with pytest.raises(ValidationError):
        ToolIntent(tool="", args={}, skill_id="echo")


def test_harness_dispatches_to_echo() -> None:
    intent = ToolIntent(tool="echo", args={"message": "hi"}, skill_id="echo")
    result = HarnessAdapter().execute(intent)
    assert result.status == "ok"
    assert result.payload == {"echoed": "hi", "length": 2}


def test_harness_returns_error_for_unknown_tool() -> None:
    intent = ToolIntent(tool="nope", args={}, skill_id="echo")
    result = HarnessAdapter().execute(intent)
    assert result.status == "error"
    assert "unknown tool" in (result.error or "")


def test_harness_traps_exception_from_tool() -> None:
    intent = ToolIntent(tool="echo", args={"message": 123}, skill_id="echo")  # type: ignore[dict-item]
    result = HarnessAdapter().execute(intent)
    assert result.status == "error"
    assert "TypeError" in (result.error or "")


def test_has_tool_returns_true_for_registered() -> None:
    assert HarnessAdapter().has_tool("echo") is True


def test_has_tool_returns_false_for_unknown() -> None:
    assert HarnessAdapter().has_tool("nonexistent") is False
