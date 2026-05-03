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
from runtime.llm.clients.base import ChatResponse
from runtime.reasoning.skill_runner import PlanStep, SkillRunner
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
        self.last_recent: tuple[tuple[str, str], ...] = ()

    async def build(
        self,
        descriptor: SkillDescriptor,
        user_text: str,
        *,
        recent: tuple[tuple[str, str], ...] = (),
    ) -> ToolIntent:
        self.last_recent = tuple(recent)
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
class _StubTier3Turn:
    role: str
    text: str


@dataclass
class _StubTier3:
    turns: list[tuple[str, str, str]] = field(default_factory=list)
    preload: dict[str, list[_StubTier3Turn]] = field(default_factory=dict)

    def append(self, chat_id: str, role: str, text: str) -> None:
        self.turns.append((chat_id, role, text))

    def recent(self, chat_id: str) -> tuple[_StubTier3Turn, ...]:
        return tuple(self.preload.get(chat_id, ()))


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


# ---------------------------------------------------------------------------
# FIRED path tests
# ---------------------------------------------------------------------------


async def test_fired_path() -> None:
    message = _FakeMessage()
    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.9))
    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads folder", message=message
    )
    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    assert message.replies[0]  # non-empty reply


async def test_tier3_written_on_fired() -> None:
    tier3 = _StubTier3()
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        tier3=tier3,
    )
    await dispatcher.dispatch(
        chat_id=999, user_text="list downloads", message=_FakeMessage()
    )
    assert len(tier3.turns) == 2
    assert tier3.turns[0] == ("999", "user", "list downloads")
    assert tier3.turns[1][1] == "bot"


async def test_recent_turns_threaded_to_runner() -> None:
    # Prior turns in the rolling window must flow into SkillRunner.build so the
    # Tier 1 reasoner can resolve references like "the same folder".
    tier3 = _StubTier3(
        preload={
            "777": [
                _StubTier3Turn("user", "show files in main Desktop folder"),
                _StubTier3Turn("bot", "Here are the files in your Desktop…"),
            ],
        }
    )
    runner_intent = ToolIntent(
        tool="files_read",
        args={"path": "~/Desktop/ava-selfie.png"},
        skill_id="read_file",
    )
    descriptor = _stub_descriptor("read_file", "files_read", ["read_file"])
    registry = _stub_registry(descriptor)
    stub_runner = _StubRunner(runner_intent)
    harness = HarnessAdapter(tools={"files_read": lambda args: {"content": "…"}})

    dispatcher = HarnessDispatcher(
        classifier=_StubClassifier("read_file", 0.9),
        registry=registry,
        runner=stub_runner,
        harness=harness,
        synthesizer=_StubSynthesizer(),
        tier3=tier3,
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
    )

    outcome = await dispatcher.dispatch(
        chat_id=777,
        user_text="open ava-selfie.png in the same folder",
        message=_FakeMessage(),
    )
    assert outcome == DispatchOutcome.FIRED
    assert stub_runner.last_recent == (
        ("user", "show files in main Desktop folder"),
        ("bot", "Here are the files in your Desktop…"),
    )


async def test_recent_turns_empty_when_tier3_lacks_recent() -> None:
    # Defensive path: any tier3 stub without .recent() still dispatches cleanly.
    @dataclass
    class _MinimalTier3:
        turns: list[tuple[str, str, str]] = field(default_factory=list)

        def append(self, chat_id: str, role: str, text: str) -> None:
            self.turns.append((chat_id, role, text))

    stub_runner = _StubRunner(
        ToolIntent(tool="files_list", args={"path": "~/Downloads"}, skill_id="list_files")
    )
    dispatcher = HarnessDispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        registry=_stub_registry(_stub_descriptor()),
        runner=stub_runner,
        harness=_stub_harness(),
        synthesizer=_StubSynthesizer(),
        tier3=_MinimalTier3(),
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
    )
    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="list downloads", message=_FakeMessage()
    )
    assert outcome == DispatchOutcome.FIRED
    assert stub_runner.last_recent == ()


async def test_error_result_synthesized() -> None:
    def _raise_perm(*_: object) -> dict:
        raise PermissionError("denied")

    error_harness = HarnessAdapter(tools={"files_list": _raise_perm})
    message = _FakeMessage()
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        harness=error_harness,
    )
    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=message
    )
    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1


async def test_synthesis_failure_falls_back_to_raw() -> None:
    message = _FakeMessage()
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        synthesizer=_RaisingSynthesizer(),
    )
    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=message
    )
    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    assert len(message.replies[0]) <= 3500


# ---------------------------------------------------------------------------
# Reply-callback routing (typing-indicator UX hook)
# ---------------------------------------------------------------------------


async def test_reply_callback_used_on_fired() -> None:
    # When the caller (Telegram route_chat) supplies a `reply` callable
    # — so it can tear down its "typing…" placeholder alongside the
    # reply — the dispatcher must send through the callback, NOT
    # message.reply_text (which would post a second, un-placeholdered
    # bubble).
    message = _FakeMessage()
    captured: list[str] = []

    async def _capture(text: str) -> None:
        captured.append(text)

    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.9))
    outcome = await dispatcher.dispatch(
        chat_id=123,
        user_text="list my downloads",
        message=message,
        reply=_capture,
    )
    assert outcome == DispatchOutcome.FIRED
    assert len(captured) == 1
    assert captured[0]
    assert message.replies == []  # never bypassed the callback


async def test_reply_callback_used_on_clarify() -> None:
    message = _FakeMessage()
    captured: list[str] = []

    async def _capture(text: str) -> None:
        captured.append(text)

    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.5))
    outcome = await dispatcher.dispatch(
        chat_id=123,
        user_text="list something",
        message=message,
        reply=_capture,
    )
    assert outcome == DispatchOutcome.CLARIFY
    assert len(captured) == 1
    assert "folder" in captured[0].lower()
    assert message.replies == []


# ---------------------------------------------------------------------------
# route_chat integration
# ---------------------------------------------------------------------------


async def test_route_chat_dispatcher_fires_before_pipeline() -> None:
    """When dispatcher returns FIRED, the pipeline is never called."""
    from types import SimpleNamespace

    from runtime.chat.telegram.bot import route_chat

    pipeline_called = False

    class _FakePipeline:
        async def turn(self, chat_id: str, text: str) -> str:
            nonlocal pipeline_called
            pipeline_called = True
            return "pipeline reply"

    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.9))
    message = _FakeMessage(text="list my downloads folder")
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=123),
        effective_message=message,
    )

    await route_chat(
        update,
        None,
        pipeline=_FakePipeline(),
        harness_dispatcher=dispatcher,
    )

    assert not pipeline_called
    assert len(message.replies) == 1


# ---------------------------------------------------------------------------
# Multi-step scaffolding (cfg.harness.multi_step) — Step 1 of
# docs/PLAN_MULTI_STEP_AGENT_LOOP.md. The full bounded loop lands in Step 2;
# Step 1 only verifies the flag routes through `runner.plan_next` and that the
# legacy single-shot `runner.build` path is preserved when the flag is off.
# ---------------------------------------------------------------------------


class _StubPlanRunner:
    """Runner stub that exposes both `build` and `plan_next` so the dispatcher
    can be exercised on either branch of the multi_step flag."""

    def __init__(
        self,
        *,
        plan_step: PlanStep,
        build_intent: ToolIntent | None = None,
    ) -> None:
        self._plan_step = plan_step
        self._build_intent = build_intent
        self.build_calls = 0
        self.plan_next_calls: list[dict[str, Any]] = []

    async def build(
        self,
        descriptor: SkillDescriptor,
        user_text: str,
        *,
        recent: tuple[tuple[str, str], ...] = (),
    ) -> ToolIntent:
        self.build_calls += 1
        if self._build_intent is None:
            raise AssertionError("build() should not be called when multi_step=True")
        return self._build_intent

    async def plan_next(
        self,
        *,
        user_text: str,
        available_skills: Any,
        history: tuple = (),
        recent: tuple[tuple[str, str], ...] = (),
    ) -> PlanStep:
        self.plan_next_calls.append(
            {
                "user_text": user_text,
                "available_skills": list(available_skills),
                "history": tuple(history),
                "recent": tuple(recent),
            }
        )
        return self._plan_step


def _make_multi_step_dispatcher(
    *,
    plan_step: PlanStep,
    descriptor: SkillDescriptor | None = None,
    runner: _StubPlanRunner | None = None,
    multi_step: bool = True,
) -> tuple[HarnessDispatcher, _StubPlanRunner]:
    descriptor = descriptor or _stub_descriptor()
    registry = _stub_registry(descriptor)
    runner = runner or _StubPlanRunner(plan_step=plan_step)
    dispatcher = HarnessDispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        registry=registry,
        runner=runner,
        harness=_stub_harness(),
        synthesizer=_StubSynthesizer(),
        tier3=_StubTier3(),
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
        multi_step=multi_step,
    )
    return dispatcher, runner


async def test_multi_step_off_by_default_uses_build() -> None:
    """Default constructor (no multi_step) must hit `runner.build`."""
    descriptor = _stub_descriptor()
    runner = _StubPlanRunner(
        plan_step=PlanStep(kind="respond"),
        build_intent=ToolIntent(
            tool="files_list", args={"path": "~/Downloads"}, skill_id="list_files"
        ),
    )
    dispatcher = HarnessDispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        registry=_stub_registry(descriptor),
        runner=runner,
        harness=_stub_harness(),
        synthesizer=_StubSynthesizer(),
        tier3=_StubTier3(),
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
    )

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="list downloads", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    assert runner.build_calls == 1
    assert runner.plan_next_calls == []


async def test_multi_step_true_routes_to_plan_next_and_fires() -> None:
    plan_step = PlanStep(
        kind="tool_call", tool="files_list", args={"path": "~/Downloads"}
    )
    dispatcher, runner = _make_multi_step_dispatcher(plan_step=plan_step)

    message = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="list downloads", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert runner.build_calls == 0
    assert len(runner.plan_next_calls) == 1
    call = runner.plan_next_calls[0]
    assert call["user_text"] == "list downloads"
    assert call["history"] == ()  # Step 1 always passes empty history
    assert len(call["available_skills"]) == 1
    assert len(message.replies) == 1


async def test_multi_step_respond_kind_returns_pass() -> None:
    dispatcher, runner = _make_multi_step_dispatcher(
        plan_step=PlanStep(kind="respond")
    )

    message = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="thanks", message=message
    )

    assert outcome == DispatchOutcome.PASS
    assert message.replies == []
    assert len(runner.plan_next_calls) == 1


async def test_multi_step_threads_recent_turns_into_plan_next() -> None:
    """The rolling tier3 window must reach plan_next for anaphora resolution."""
    descriptor = _stub_descriptor()
    runner = _StubPlanRunner(
        plan_step=PlanStep(kind="tool_call", tool="files_list", args={"path": "~/Downloads"}),
    )
    tier3 = _StubTier3(
        preload={
            "777": [
                _StubTier3Turn("user", "show files in main Desktop folder"),
                _StubTier3Turn("bot", "Here are the files in your Desktop…"),
            ],
        }
    )
    dispatcher = HarnessDispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        registry=_stub_registry(descriptor),
        runner=runner,
        harness=_stub_harness(),
        synthesizer=_StubSynthesizer(),
        tier3=tier3,
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
        multi_step=True,
    )

    await dispatcher.dispatch(
        chat_id=777, user_text="and the same folder again?", message=_FakeMessage()
    )

    assert len(runner.plan_next_calls) == 1
    assert runner.plan_next_calls[0]["recent"] == (
        ("user", "show files in main Desktop folder"),
        ("bot", "Here are the files in your Desktop…"),
    )
