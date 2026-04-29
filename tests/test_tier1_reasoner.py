"""Tests for `Tier1Reasoner` and the skill runner's graceful degradation."""
from __future__ import annotations

import httpx
import pytest

from runtime.harness import ToolIntent
from runtime.llm.clients import ChatRequest, ChatResponse
from runtime.reasoning.skill_runner import SkillRunner
from runtime.reasoning.tier1_reasoner import Tier1Reasoner, Tier1ReasonerError
from runtime.skills import SkillDescriptor


class _StubClient:
    def __init__(self, *, content: str = "", raises: Exception | None = None) -> None:
        self._content = content
        self._raises = raises
        self.calls: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if self._raises is not None:
            raise self._raises
        return ChatResponse(content=self._content, model=request.model)

    async def health(self) -> bool:
        return True


def _ask_question() -> SkillDescriptor:
    return SkillDescriptor(
        id="ask_question",
        description="Answer a general question using a frontier model.",
        intents=["ask_question"],
        tool="respond",
        args_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        requires_tier1=True,
    )


@pytest.mark.asyncio
async def test_tier1_reasoner_produces_contract() -> None:
    stub = _StubClient(
        content='{"args": {"message": "Tokyo is UTC+9."}, "rationale": "asked for time"}'
    )
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")

    intent = await reasoner.reason(_ask_question(), "what time is it in Tokyo?")

    assert isinstance(intent, ToolIntent)
    assert intent.tool == "respond"
    assert intent.args == {"message": "Tokyo is UTC+9."}
    assert intent.skill_id == "ask_question"
    assert intent.rationale.startswith("asked")


@pytest.mark.asyncio
async def test_tier1_reasoner_includes_recent_turns_in_prompt() -> None:
    stub = _StubClient(
        content='{"args": {"message": "ok"}, "rationale": "r"}'
    )
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")

    await reasoner.reason(
        _ask_question(),
        "and what about tomorrow?",
        recent=[
            ("user", "what's the weather in Tokyo?"),
            ("bot", "22°C and sunny"),
        ],
    )

    assert len(stub.calls) == 1
    system_prompt = stub.calls[0].messages[0].content
    assert "USER: what's the weather in Tokyo?" in system_prompt
    assert "BOT: 22°C and sunny" in system_prompt


@pytest.mark.asyncio
async def test_tier1_reasoner_renders_empty_history_sentinel() -> None:
    stub = _StubClient(content='{"args": {"message": "ok"}, "rationale": "r"}')
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")

    await reasoner.reason(_ask_question(), "hi")

    system_prompt = stub.calls[0].messages[0].content
    assert "(no prior turns)" in system_prompt


@pytest.mark.asyncio
async def test_tier1_reasoner_rejects_extra_args() -> None:
    stub = _StubClient(content='{"args": {"message": "ok", "rogue": true}, "rationale": "x"}')
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")
    with pytest.raises(Tier1ReasonerError):
        await reasoner.reason(_ask_question(), "hello")


@pytest.mark.asyncio
async def test_tier1_reasoner_rejects_non_json() -> None:
    stub = _StubClient(content="sorry, I cannot comply")
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")
    with pytest.raises(Tier1ReasonerError):
        await reasoner.reason(_ask_question(), "hello")


@pytest.mark.asyncio
async def test_tier1_reasoner_wraps_transport_errors() -> None:
    stub = _StubClient(raises=httpx.ConnectError("refused"))
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")
    with pytest.raises(Tier1ReasonerError):
        await reasoner.reason(_ask_question(), "hello")


@pytest.mark.asyncio
async def test_skill_runner_degrades_when_no_reasoner() -> None:
    runner = SkillRunner(tier1=None)
    intent = await runner.build(_ask_question(), "anything")
    assert intent.tool == "respond"
    assert "tier1 unavailable" in intent.args["message"]


@pytest.mark.asyncio
async def test_skill_runner_degrades_when_reasoner_fails() -> None:
    stub = _StubClient(content="garbage")
    reasoner = Tier1Reasoner(client=stub, model="minimax/minimax-m2.7")
    runner = SkillRunner(tier1=reasoner)
    intent = await runner.build(_ask_question(), "anything")
    assert intent.tool == "respond"
    assert "tier1 unavailable" in intent.args["message"]


@pytest.mark.asyncio
async def test_skill_runner_tier0_path_unchanged() -> None:
    echo = SkillDescriptor(
        id="echo",
        description="Echoes text back to the user.",
        intents=["echo", "ping"],
        tool="echo",
        args_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        requires_tier1=False,
    )
    runner = SkillRunner(tier1=None)
    intent = await runner.build(echo, "echo hello")
    assert intent.tool == "echo"
    assert intent.args == {"message": "hello"}
