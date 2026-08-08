from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.config import CommandsConfig
from runtime.harness import DEFAULT_TOOLS, HarnessAdapter, ToolIntent
from runtime.harness.tools.command_tool import make_command_tool

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


# --- run_command wiring shape (Phase 11 D3) ----------------------------------
# Mirrors the merge in runtime/chat/telegram/bot.py's build_harness_dispatcher:
# HarnessAdapter(tools={**DEFAULT_TOOLS, "run_command": make_command_tool(cfg.commands)}).


def test_harness_adapter_with_merged_run_command_has_tool(tmp_path: Path) -> None:
    tools = {**DEFAULT_TOOLS, "run_command": make_command_tool(CommandsConfig())}
    harness = HarnessAdapter(tools=tools)

    assert harness.has_tool("run_command") is True
    assert harness.has_tool("echo") is True  # merge kept the defaults


def test_harness_adapter_executes_run_command_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    tools = {**DEFAULT_TOOLS, "run_command": make_command_tool(CommandsConfig())}
    harness = HarnessAdapter(tools=tools)
    intent = ToolIntent(
        tool="run_command",
        args={"argv": ["ls", str(tmp_path)]},
        skill_id="run_command",
    )

    result = harness.execute(intent)

    assert result.status == "ok"
    assert result.payload["verdict"] == "verified"
    assert "hello.txt" in result.payload["stdout_tail"]
