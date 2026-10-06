"""The "bare" arm: a typical native-tool-calling loop with none of AEGIS's guardrails.

Why it exists (docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md):
every result so far came from the AEGIS planner protocol -- a forced JSON schema
-- plus plan-lock, completion gate and claim annotation. That confounds the
protocol with the guardrails. This arm drives the same models through the
Ollama-native `tools` interface the models were trained on, with the same
sandboxed tools, tasks and grader, so full-vs-bare isolates what AEGIS adds.

Network is mocked with respx; no live model is called.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from runtime.config import ModelProfile, ProviderConfig, get_config
from runtime.eval import bare as bare_mod
from runtime.eval.bare import (
    BARE_SYSTEM_PROMPT,
    build_tool_specs,
    native_chat,
    parse_tool_calls,
    run_native_loop,
)
from runtime.eval.tasks import EvalTask, ExpectedCall
from runtime.llm.clients import OllamaClient
from runtime.llm.telemetry import collect_calls
from runtime.skills.registry import SkillDescriptor

pytestmark = pytest.mark.unit

_PATH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


def _descriptor(skill_id: str, tool: str, description: str = "does a thing") -> SkillDescriptor:
    return SkillDescriptor(
        id=skill_id, description=description, tool=tool, args_schema=_PATH_SCHEMA
    )


# --- build_tool_specs ----------------------------------------------------


def test_tool_specs_use_the_ollama_function_shape() -> None:
    specs = build_tool_specs(
        [_descriptor("list_files", "files_list", "List files")], has_tool=lambda _t: True
    )

    assert specs == [
        {
            "type": "function",
            "function": {
                "name": "files_list",
                "description": "List files",
                "parameters": _PATH_SCHEMA,
            },
        }
    ]


def test_tool_specs_leave_out_respond_and_tools_the_harness_lacks() -> None:
    """`respond` is the planner's way of answering in prose; a native-tools model
    just answers. And a skill whose tool is not wired must not be offered."""
    specs = build_tool_specs(
        [
            _descriptor("list_files", "files_list"),
            _descriptor("respond", "respond"),
            _descriptor("ghost", "ghost_tool"),
        ],
        has_tool=lambda t: t != "ghost_tool",
    )

    assert [s["function"]["name"] for s in specs] == ["files_list"]


# --- parse_tool_calls ----------------------------------------------------


def _call(name: str, arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def test_parse_tool_calls_reads_dict_arguments() -> None:
    message = {"tool_calls": [_call("files_list", {"path": "/x"})]}
    assert parse_tool_calls(message) == [("files_list", {"path": "/x"})]


def test_parse_tool_calls_accepts_json_string_arguments() -> None:
    """Some templates return arguments as a JSON string, not an object."""
    message = {"tool_calls": [_call("files_list", '{"path": "/x"}')]}
    assert parse_tool_calls(message) == [("files_list", {"path": "/x"})]


def test_parse_tool_calls_survives_malformed_arguments() -> None:
    """A garbled argument blob is the model's mistake to be graded, not a crash."""
    message = {"tool_calls": [_call("files_list", "{not json")]}
    assert parse_tool_calls(message) == [("files_list", {})]


def test_parse_tool_calls_is_empty_without_calls() -> None:
    assert parse_tool_calls({"content": "hello"}) == []
    assert parse_tool_calls({"tool_calls": None}) == []


# --- run_native_loop -----------------------------------------------------


class _ScriptedChat:
    """Replays assistant messages (the last one repeats) and remembers what it was sent."""

    def __init__(self, *messages: dict[str, Any]) -> None:
        self._messages = list(messages)
        self.sent: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.sent.append([dict(m) for m in messages])
        return self._messages.pop(0) if len(self._messages) > 1 else self._messages[0]


async def test_loop_executes_a_tool_call_then_returns_the_final_answer() -> None:
    chat = _ScriptedChat(
        {"role": "assistant", "content": "", "tool_calls": [_call("files_list", {"path": "/x"})]},
        {"role": "assistant", "content": "Found 2 files."},
    )
    executed: list[tuple[str, dict[str, Any]]] = []

    def execute(tool: str, args: dict[str, Any]) -> str:
        executed.append((tool, args))
        return '{"entries": ["a", "b"]}'

    result = await run_native_loop(
        chat=chat,
        execute=execute,
        messages=[{"role": "user", "content": "list /x"}],
        max_steps=5,
    )

    assert executed == [("files_list", {"path": "/x"})]
    assert result.reply == "Found 2 files."
    assert result.steps == 1
    assert not result.hit_cap
    # The second request carries the assistant's call and the tool's result back.
    second = chat.sent[1]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"]
    assert second[-1] == {
        "role": "tool",
        "tool_name": "files_list",
        "content": '{"entries": ["a", "b"]}',
    }


async def test_loop_returns_immediately_when_the_model_answers_without_a_tool() -> None:
    chat = _ScriptedChat({"role": "assistant", "content": "I cannot do that."})

    result = await run_native_loop(
        chat=chat, execute=lambda *_a: "", messages=[{"role": "user", "content": "hi"}], max_steps=5
    )

    assert result.reply == "I cannot do that."
    assert result.steps == 0


async def test_loop_stops_at_the_step_cap() -> None:
    """A model that keeps calling tools must not run forever -- the same hard cap
    the full harness applies (`harness.max_steps`)."""
    chat = _ScriptedChat(
        {"role": "assistant", "content": "", "tool_calls": [_call("files_list", {"path": "/x"})]}
    )
    calls: list[str] = []

    result = await run_native_loop(
        chat=chat,
        execute=lambda tool, _a: calls.append(tool) or "ok",
        messages=[{"role": "user", "content": "list /x"}],
        max_steps=3,
    )

    assert len(calls) == 3
    assert result.hit_cap
    assert result.reply is None


async def test_loop_reply_is_none_when_the_final_message_is_empty() -> None:
    chat = _ScriptedChat({"role": "assistant", "content": ""})
    result = await run_native_loop(
        chat=chat, execute=lambda *_a: "", messages=[{"role": "user", "content": "hi"}], max_steps=5
    )
    assert result.reply is None


# --- native_chat: the HTTP boundary -------------------------------------


def _cfg(**update: Any) -> Any:
    return get_config().model_copy(
        update={"providers": ProviderConfig(ollama_base_url="http://127.0.0.1:11434"), **update}
    )


def _ollama_reply(message: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": message,
            "prompt_eval_count": 40,
            "eval_count": 12,
            "load_duration": 3_000_000,
            "eval_duration": 5_000_000,
            "done_reason": "stop",
        },
    )


async def test_native_chat_sends_tools_and_the_profile_settings() -> None:
    """num_ctx and think must come from the same profiles the full arm uses, or
    the two arms would differ in more than the harness."""
    cfg = _cfg(model_profiles={"qwen3-vl:4b": ModelProfile(think=False, num_ctx=32768)})
    tools = [{"type": "function", "function": {"name": "files_list", "parameters": {}}}]
    captured: dict[str, Any] = {}

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return _ollama_reply({"role": "assistant", "content": "ok"})

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(side_effect=_handler)
        message = await native_chat(
            OllamaClient(cfg),
            cfg,
            model="qwen3-vl:4b",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            max_tokens=2048,
        )

    assert message["content"] == "ok"
    assert captured["tools"] == tools
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["options"]["num_ctx"] == 32768
    assert captured["options"]["num_predict"] == 2048


async def test_native_chat_records_telemetry_like_the_full_arm() -> None:
    """Without it the bare arm's timeouts and token use would be invisible, and
    the two arms could not be compared on the budget metric."""
    cfg = _cfg()
    with respx.mock() as mock, collect_calls() as calls:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=_ollama_reply({"role": "assistant", "content": "ok"})
        )
        await native_chat(
            OllamaClient(cfg),
            cfg,
            model="gemma4:e2b",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_tokens=512,
        )

    assert len(calls) == 1
    assert calls[0].model == "gemma4:e2b"
    assert calls[0].tokens_out == 12
    assert calls[0].eval_ms == 5


# --- run_variant_bare: end to end with a scripted model -----------------


class _Registry:
    def __init__(self, *descriptors: SkillDescriptor) -> None:
        self._descriptors = list(descriptors)

    def all(self) -> list[SkillDescriptor]:
        return self._descriptors


def _list_task() -> EvalTask:
    return EvalTask(
        id="list_downloads",
        description="list a folder",
        variants=("list files in {sandbox}",),
        expected_calls=(ExpectedCall(tool="files_list", args_match={"path": "{sandbox}"}),),
    )


def _registry() -> _Registry:
    return _Registry(_descriptor("list_files", "files_list", "List files in a folder."))


async def _run_bare() -> Any:
    task = _list_task()
    return await bare_mod.run_variant_bare(
        get_config(), _registry(), None, task, task.variants[0]  # type: ignore[arg-type]
    )


async def test_bare_variant_passes_when_the_model_calls_the_right_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def _fake_chat(_client: Any, _cfg: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        if len(seen) == 1:
            # The sandbox path only exists once the run has made it; it is the
            # last word of the user message.
            path = kwargs["messages"][-1]["content"].rsplit(" ", 1)[-1]
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("files_list", {"path": path})],
            }
        return {"role": "assistant", "content": "There are two files."}

    monkeypatch.setattr(bare_mod, "native_chat", _fake_chat)
    result = await _run_bare()

    assert result.passed, result.reason
    assert result.reply == "There are two files."
    assert [c.tool for c in result.actual_calls] == ["files_list"]
    assert result.failure_kind is None
    # The model was offered the registry's tool in native form, plus a minimal prompt.
    assert seen[0]["tools"][0]["function"]["name"] == "files_list"
    assert seen[0]["messages"][0] == {"role": "system", "content": BARE_SYSTEM_PROMPT}


async def test_bare_variant_that_never_calls_a_tool_is_labelled_no_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reply is kept: this is exactly where a false 'I listed your files'
    claim would show up."""

    async def _fake_chat(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "I listed the files for you."}

    monkeypatch.setattr(bare_mod, "native_chat", _fake_chat)
    result = await _run_bare()

    assert not result.passed
    assert result.failure_kind == "no_tool_call"
    assert result.reply == "I listed the files for you."


async def test_bare_variant_survives_the_model_call_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(bare_mod, "native_chat", _boom)
    result = await _run_bare()

    assert not result.passed
    assert "run_variant_bare raised" in result.reason


# --- the CLI chooses the arm --------------------------------------------


def test_cli_arm_defaults_to_the_full_harness() -> None:
    from runtime.eval.cli import _parse_args

    assert _parse_args([]).arm == "full"


def test_cli_accepts_the_bare_arm() -> None:
    from runtime.eval.cli import _parse_args

    assert _parse_args(["--arm", "bare"]).arm == "bare"


def test_cli_rejects_an_unknown_arm() -> None:
    from runtime.eval.cli import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--arm", "turbo"])


def test_cli_bare_arm_refuses_a_hosted_provider_up_front() -> None:
    """Otherwise a hosted config would produce 45 identical 'ollama only' failures
    that look like a model result."""
    from runtime.eval.cli import _arm_unsupported_reason

    assert _arm_unsupported_reason("bare", "openrouter") is not None
    assert _arm_unsupported_reason("bare", "ollama") is None
    assert _arm_unsupported_reason("full", "openrouter") is None


def test_cli_picks_the_variant_runner_for_the_arm() -> None:
    from runtime.eval.cli import _variant_runner
    from runtime.eval.runner import run_variant

    assert _variant_runner("full") is run_variant
    assert _variant_runner("bare") is bare_mod.run_variant_bare


# --- run_task takes the variant runner ----------------------------------


async def test_run_task_uses_the_supplied_variant_runner() -> None:
    from runtime.eval.report import VariantResult
    from runtime.eval.runner import run_task

    seen: list[str] = []

    async def _fake(cfg: Any, registry: Any, loader: Any, task: Any, text: str) -> VariantResult:
        seen.append(text)
        return VariantResult(task_id=task.id, variant_text=text, passed=True, duration_s=0.0)

    task = _list_task()
    result = await run_task(
        get_config(), _registry(), None, task, repeat=2, run_fn=_fake  # type: ignore[arg-type]
    )

    assert seen == [task.variants[0]] * 2
    assert len(result.variants) == 2
