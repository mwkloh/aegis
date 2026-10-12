"""One native-mode turn (spec: 2026-10-06-harness-ablation-design.md, section 4)."""
from __future__ import annotations

from typing import Any

import pytest

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.skills.registry import SkillDescriptor

pytestmark = pytest.mark.unit


def _desc(tool: str) -> SkillDescriptor:
    return SkillDescriptor(
        id=tool,
        description=f"{tool} tool",
        intents=[tool],
        tool=tool,
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )


DESCRIPTORS = [
    _desc("files_list"),
    _desc("files_search"),
    _desc("files_read"),
    _desc("files_delete"),
]


class _Harness:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools
        self.calls: list[ToolIntent] = []

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, intent: ToolIntent) -> ToolResult:
        self.calls.append(intent)
        return ToolResult(status="ok", payload=self._tools[intent.tool](intent.args))


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
    }


def _answer(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


class _Script:
    """Fake native chat: returns the scripted messages in order, records what it saw.
    Once the script runs out it keeps asking for `files_list` (to exercise the cap)."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        return self._replies.pop(0) if self._replies else _call("files_list", path="/tmp")


async def _turn(chat: _Script, harness: _Harness, **kw: Any):
    from runtime.chat.telegram.native_turn import run_native_turn

    args: dict[str, Any] = {
        "descriptors": DESCRIPTORS,
        "user_text": "do it",
        "skill_id": "native",
        "max_steps": 5,
        "autofill": True,
        "tool_result_chars": 3500,
    }
    args.update(kw)
    return await run_native_turn(chat=chat, harness=harness, **args)


async def test_runs_requested_tools_until_the_model_answers() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "a.md\nb.md"}})
    turn = await _turn(_Script(_call("files_list", path="/tmp"), _answer("Two files.")), harness)
    assert turn.final_text == "Two files."
    assert [i.tool for i, _ in turn.history] == ["files_list"]
    assert turn.hit_cap is False


async def test_system_prompt_defaults_to_the_bare_benchmark_prompt() -> None:
    """The eval's bare arm shares NATIVE_SYSTEM_PROMPT; production passes its own."""
    from runtime.harness.native_loop import NATIVE_SYSTEM_PROMPT

    script = _Script(_answer("hi"))
    await _turn(script, _Harness({}))
    assert script.seen[0][0] == {"role": "system", "content": NATIVE_SYSTEM_PROMPT}

    script = _Script(_answer("hi"))
    await _turn(script, _Harness({}), system_prompt="You are Eva.")
    assert script.seen[0][0] == {"role": "system", "content": "You are Eva."}


def _search_read_harness() -> _Harness:
    return _Harness(
        {
            "files_search": lambda a: {"result": "/tmp/n/CT-001-notes.md"},
            "files_read": lambda a: {"result": "notes"},
        }
    )


def _search_then_guess() -> _Script:
    return _Script(
        _call("files_search", path="/tmp/n"),
        _call("files_read", path="/tmp/n/ct001.md"),
        _answer("ok"),
    )


async def test_autofill_corrects_a_guessed_read_path() -> None:
    harness = _search_read_harness()
    await _turn(_search_then_guess(), harness)
    assert harness.calls[1].args["path"] == "/tmp/n/CT-001-notes.md"


async def test_autofill_off_keeps_the_guess() -> None:
    harness = _search_read_harness()
    await _turn(_search_then_guess(), harness, autofill=False)
    assert harness.calls[1].args["path"] == "/tmp/n/ct001.md"


async def test_step_cap_ends_with_no_text_and_keeps_the_history() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    turn = await _turn(_Script(), harness, max_steps=2)
    assert turn.hit_cap is True
    assert turn.final_text is None
    assert len(turn.history) == 2


async def test_a_destructive_tool_after_the_first_call_is_intercepted() -> None:
    harness = _Harness(
        {"files_list": lambda a: {"result": "x"}, "files_delete": lambda a: {"result": "gone"}}
    )
    chat = _Script(_call("files_list", path="/tmp"), _call("files_delete", path="/tmp/x"))
    turn = await _turn(chat, harness)
    assert turn.guarded_intent is not None
    assert turn.guarded_intent.tool == "files_delete"
    assert [c.tool for c in harness.calls] == ["files_list"]  # delete never ran


async def test_an_unknown_tool_returns_an_error_to_the_model() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    chat = _Script(_call("files_teleport", path="/tmp"), _answer("sorry"))
    turn = await _turn(chat, harness)
    assert turn.history[0][1].status == "error"
    tool_msgs = [m for m in chat.seen[1] if m.get("role") == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]


async def test_tool_results_are_clipped_to_the_per_model_limit() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "Z" * 20000}})
    chat = _Script(_call("files_list", path="/tmp"), _answer("done"))
    await _turn(chat, harness, tool_result_chars=1000)
    tool_msg = next(m for m in chat.seen[1] if m.get("role") == "tool")
    assert len(tool_msg["content"]) < 1200
    assert "truncated" in tool_msg["content"]


async def test_every_executed_call_is_reported_for_the_ledger() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    seen: list[str] = []
    await _turn(
        _Script(_call("files_list", path="/tmp"), _answer("ok")),
        harness,
        on_executed=lambda intent, result: seen.append(intent.tool),
    )
    assert seen == ["files_list"]
