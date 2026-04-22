"""Unit tests for HarnessDispatcher — all stubs, no network, no filesystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from runtime.chat.memory.tier1 import Tier1Loader, Tier1Snapshot
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.telegram.harness_dispatcher import DispatchOutcome, HarnessDispatcher
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.intent.classifier import IntentClassification
from runtime.model_router.clients.base import ChatResponse
from runtime.reasoning.skill_runner import SkillRunner
from runtime.skills.registry import SkillDescriptor, SkillRegistry

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubClassifier:
    def __init__(self, intent: str, confidence: float) -> None:
        self._result = IntentClassification(intent=intent, confidence=confidence)

    async def classify(self, text: str) -> IntentClassification:
        return self._result


class _RaisingClassifier:
    async def classify(self, text: str) -> IntentClassification:
        raise RuntimeError("ollama down")


def _stub_descriptor(
    skill_id: str = "list_files",
    tool: str = "files_list",
    intents: list[str] | None = None,
) -> SkillDescriptor:
    return SkillDescriptor(
        id=skill_id,
        description="List files and directories at a local path.",
        tool=tool,
        intents=intents or [skill_id],
        args_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        requires_tier1=True,
    )


def _stub_registry(descriptor: SkillDescriptor | None = None) -> SkillRegistry:
    if descriptor is None:
        return SkillRegistry([])
    return SkillRegistry([descriptor])


class _StubRunner:
    def __init__(self, intent: ToolIntent) -> None:
        self._intent = intent

    async def build(self, descriptor: SkillDescriptor, user_text: str) -> ToolIntent:
        return self._intent


def _stub_harness(tool: str = "files_list") -> HarnessAdapter:
    return HarnessAdapter(
        tools={
            tool: lambda args: {"entries": ["file1.txt", "file2.txt"]},
        }
    )


class _StubSynthesizer:
    def __init__(self, reply: str = "Here are your files: file1.txt, file2.txt") -> None:
        self._reply = reply

    async def chat(self, request: Any) -> ChatResponse:
        return ChatResponse(content=self._reply, model="stub", tokens_in=0, tokens_out=0)


class _RaisingSynthesizer:
    async def chat(self, request: Any) -> ChatResponse:
        raise RuntimeError("openrouter down")


@dataclass
class _StubTier3:
    turns: list[tuple[str, str, str]] = field(default_factory=list)

    def append(self, chat_id: str, role: str, text: str) -> None:
        self.turns.append((chat_id, role, text))


@dataclass
class _StubTier1Loader:
    def load(self, chat_id: str) -> Tier1Snapshot:
        return Tier1Snapshot(
            identity="Eva, your personal assistant",
            user="",
            bytes_identity=30,
            bytes_user=0,
            bytes_prefs=0,
        )


@dataclass
class _FakeMessage:
    replies: list[str] = field(default_factory=list)
    text: str = ""

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


def _make_dispatcher(
    *,
    classifier: Any = None,
    descriptor: SkillDescriptor | None = None,
    runner_intent: ToolIntent | None = None,
    harness: HarnessAdapter | None = None,
    synthesizer: Any = None,
    tier3: _StubTier3 | None = None,
    tier1_loader: Any = None,
) -> HarnessDispatcher:
    if descriptor is None:
        descriptor = _stub_descriptor()
    registry = _stub_registry(descriptor)
    if runner_intent is None:
        runner_intent = ToolIntent(
            tool="files_list",
            args={"path": "~/Downloads"},
            skill_id="list_files",
        )
    return HarnessDispatcher(
        classifier=classifier or _StubClassifier("list_files", 0.9),
        registry=registry,
        runner=_StubRunner(runner_intent),
        harness=harness or _stub_harness(),
        synthesizer=synthesizer or _StubSynthesizer(),
        tier3=tier3 or _StubTier3(),
        tier1_loader=tier1_loader or _StubTier1Loader(),
        synthesis_model="stub-model",
    )


# ---------------------------------------------------------------------------
# PASS path tests
# ---------------------------------------------------------------------------


async def test_pass_on_non_tool_intent() -> None:
    descriptor = _stub_descriptor("list_files", "files_list", ["list_files"])
    registry = SkillRegistry([descriptor])
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("ask_question", 0.95),
        descriptor=descriptor,
    )
    dispatcher._registry = registry
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=123, user_text="what time is it?", message=message)
    assert outcome == DispatchOutcome.PASS
    assert message.replies == []


async def test_pass_on_unknown_intent() -> None:
    dispatcher = _make_dispatcher(classifier=_StubClassifier("unknown", 0.0))
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=123, user_text="hmm", message=message)
    assert outcome == DispatchOutcome.PASS
    assert message.replies == []


async def test_pass_when_tool_not_in_harness() -> None:
    harness = HarnessAdapter(tools={"echo": lambda args: {"echoed": "x", "length": 1}})
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        harness=harness,
    )
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=123, user_text="list my downloads", message=message)
    assert outcome == DispatchOutcome.PASS
    assert message.replies == []


async def test_pass_on_tier1_degrade() -> None:
    degrade_intent = ToolIntent(
        tool="respond",
        args={"message": "tier1 unavailable"},
        skill_id="list_files",
    )
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        runner_intent=degrade_intent,
    )
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=123, user_text="list my downloads", message=message)
    assert outcome == DispatchOutcome.PASS
    assert message.replies == []


async def test_classifier_exception_returns_pass() -> None:
    dispatcher = _make_dispatcher(classifier=_RaisingClassifier())
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=123, user_text="list my downloads", message=message)
    assert outcome == DispatchOutcome.PASS
    assert message.replies == []


# ---------------------------------------------------------------------------
# CLARIFY path tests
# ---------------------------------------------------------------------------


async def test_clarify_path() -> None:
    # Confidence below threshold → clarifying question
    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.5))
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list something", message=message
    )
    assert outcome == DispatchOutcome.CLARIFY
    assert len(message.replies) == 1
    assert "folder" in message.replies[0].lower()


async def test_tier3_written_on_clarify() -> None:
    tier3 = _StubTier3()
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.5),
        tier3=tier3,
    )
    await dispatcher.dispatch(
        chat_id=555, user_text="show me files", message=_FakeMessage()
    )
    assert len(tier3.turns) == 2
    chat_ids = {t[0] for t in tier3.turns}
    assert chat_ids == {"555"}
    roles = [t[1] for t in tier3.turns]
    assert roles == ["user", "bot"]
    assert tier3.turns[0][2] == "show me files"
