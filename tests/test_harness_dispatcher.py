"""Unit tests for HarnessDispatcher — all stubs, no network, no filesystem."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from runtime.chat.memory.tier1 import Tier1Loader, Tier1Snapshot
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.reply_verdict import UNVERIFIED_BANNER
from runtime.chat.telegram.harness_dispatcher import DispatchOutcome, HarnessDispatcher
from runtime.events import EventStream
from runtime.files.client import FilesClient
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.harness.tools.files_tool import make_files_tools
from runtime.intent.classifier import IntentClassification
from runtime.llm.clients.base import ChatResponse
from runtime.reasoning.skill_runner import PlanStep, SkillRunner
from runtime.skills.registry import SkillDescriptor, SkillRegistry
from runtime.tools.record import load_tool_calls

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubClassifier:
    def __init__(self, intent: str, confidence: float) -> None:
        self._result = IntentClassification(intent=intent, confidence=confidence)
        self.calls: list[str] = []

    async def classify(self, text: str) -> IntentClassification:
        self.calls.append(text)
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


class _RecordingSynthesizer:
    """Synthesizer stub that counts calls — used to prove the completion
    gate skips chain synthesis entirely rather than just ignoring output."""

    def __init__(self, reply: str = "should never be used") -> None:
        self._reply = reply
        self.calls = 0

    async def chat(self, request: Any) -> ChatResponse:
        self.calls += 1
        return ChatResponse(content=self._reply, model="stub", tokens_in=0, tokens_out=0)


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


@dataclass
class _MutableClock:
    """Injectable clock for TTL tests — advance `.now` between dispatches."""

    now: datetime

    def __call__(self) -> datetime:
        return self.now


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
    """Runner stub that exposes both `build` and `plan_next`.

    `plan_next` returns from a queue of `PlanStep`s in order. After the queue
    is drained the stub keeps returning the LAST step (so a stub seeded with
    a single `tool_call` step exercises the step-cap path without needing
    `max_steps + 1` queue entries). Pass `plan_step=` for the legacy
    single-element queue API.
    """

    def __init__(
        self,
        *,
        plan_step: PlanStep | None = None,
        plan_steps: list[PlanStep] | None = None,
        build_intent: ToolIntent | None = None,
    ) -> None:
        if plan_steps is None:
            assert plan_step is not None, "supply plan_step or plan_steps"
            plan_steps = [plan_step]
        self._plan_steps = list(plan_steps)
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
        # Pop from the queue; once drained, repeat the last step so callers
        # that only seed one step exercise the step-cap path naturally.
        if len(self._plan_steps) > 1:
            return self._plan_steps.pop(0)
        return self._plan_steps[0]


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
    descriptor = _stub_descriptor()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_list", args={"path": "~/Downloads"}),
            PlanStep(kind="respond"),
        ],
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
        multi_step=True,
    )

    message = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="list downloads", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert runner.build_calls == 0
    assert len(runner.plan_next_calls) == 2  # tool_call + respond
    first = runner.plan_next_calls[0]
    assert first["user_text"] == "list downloads"
    assert first["history"] == ()  # first call always sees empty history
    assert len(first["available_skills"]) == 1
    # Second call sees the one tool result threaded into history.
    assert len(runner.plan_next_calls[1]["history"]) == 1
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
    """The rolling tier3 window must reach EVERY plan_next call (anaphora)."""
    descriptor = _stub_descriptor()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_list", args={"path": "~/Downloads"}),
            PlanStep(kind="respond"),
        ],
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

    assert len(runner.plan_next_calls) >= 1
    expected_recent = (
        ("user", "show files in main Desktop folder"),
        ("bot", "Here are the files in your Desktop…"),
    )
    for call in runner.plan_next_calls:
        assert call["recent"] == expected_recent


# ---------------------------------------------------------------------------
# Multi-step bounded LOOP body — Step 2 of docs/PLAN_MULTI_STEP_AGENT_LOOP.md.
# ---------------------------------------------------------------------------


class _RecordingHarness:
    """HarnessAdapter-shaped stub that records each execute() call.

    Tools is a dict[tool_name → callable]; the callable returns either a
    payload dict or raises (which the real adapter wraps into an error
    ToolResult — we mirror that here).
    """

    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools
        self.calls: list[ToolIntent] = []

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, intent: ToolIntent) -> ToolResult:
        self.calls.append(intent)
        fn = self._tools.get(intent.tool)
        if fn is None:
            return ToolResult(status="error", error=f"unknown: {intent.tool}")
        try:
            payload = fn(intent.args)
        except Exception as exc:
            return ToolResult(status="error", error=f"{type(exc).__name__}: {exc}")
        return ToolResult(status="ok", payload=payload)


class _MultiSkillRegistry:
    """Registry stub that exposes multiple skills via .all() while still
    routing one classified intent through .for_intent()."""

    def __init__(self, descriptors: list[SkillDescriptor], primary_intent: str) -> None:
        self._descriptors = descriptors
        self._primary = primary_intent

    def for_intent(self, intent: str) -> SkillDescriptor | None:
        for d in self._descriptors:
            if intent in d.intents:
                return d
        return None

    def all(self) -> list[SkillDescriptor]:
        return list(self._descriptors)


def _two_skill_setup() -> tuple[_MultiSkillRegistry, _RecordingHarness]:
    search = SkillDescriptor(
        id="search_files",
        description="Search for files matching a glob.",
        intents=["search_files"],
        tool="files_search",
        args_schema={"type": "object", "properties": {"glob": {"type": "string"}}},
        requires_tier1=True,
    )
    read = SkillDescriptor(
        id="read_file",
        description="Read a file's contents.",
        intents=["read_file"],
        tool="files_read",
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )
    registry = _MultiSkillRegistry([search, read], primary_intent="search_files")
    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": ["/tmp/a.md", "/tmp/b.md"]},
            "files_read": lambda args: {"content": "hello"},
        }
    )
    return registry, harness


def _make_loop_dispatcher(
    *,
    runner: _StubPlanRunner,
    registry: _MultiSkillRegistry | None = None,
    harness: _RecordingHarness | None = None,
    synthesizer: Any = None,
    tier3: _StubTier3 | None = None,
    max_steps: int = 5,
    events: EventStream | None = None,
    clock: Callable[[], datetime] | None = None,
    classifier: Any = None,
) -> HarnessDispatcher:
    if registry is None or harness is None:
        registry, harness = _two_skill_setup()
    return HarnessDispatcher(
        classifier=classifier or _StubClassifier("search_files", 0.9),
        registry=registry,
        runner=runner,
        harness=harness,
        synthesizer=synthesizer or _StubSynthesizer(reply="chain reply"),
        tier3=tier3 or _StubTier3(),
        tier1_loader=_StubTier1Loader(),
        synthesis_model="stub-model",
        multi_step=True,
        max_steps=max_steps,
        events=events,
        clock=clock,
    )


async def test_multi_step_happy_two_step_chain() -> None:
    """Planner: search → read → respond. Both tools fire; synthesis gets the chain."""
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )
    tier3 = _StubTier3()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, tier3=tier3
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and read", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(runner.plan_next_calls) == 3
    # First call has empty history; subsequent calls thread accumulating history.
    assert runner.plan_next_calls[0]["history"] == ()
    assert len(runner.plan_next_calls[1]["history"]) == 1
    assert len(runner.plan_next_calls[2]["history"]) == 2
    # Both tools executed, in order.
    assert [c.tool for c in harness.calls] == ["files_search", "files_read"]
    assert len(message.replies) == 1
    # tier3 logged user + bot turn.
    assert [t[1] for t in tier3.turns] == ["user", "bot"]


async def test_multi_step_single_tool_then_respond() -> None:
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="respond"),
        ],
    )
    registry, harness = _two_skill_setup()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness
    )

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(harness.calls) == 1
    # Synthesis got exactly the one call's history.
    assert len(runner.plan_next_calls[1]["history"]) == 1


async def test_multi_step_immediate_respond_returns_pass() -> None:
    runner = _StubPlanRunner(plan_steps=[PlanStep(kind="respond")])
    registry, harness = _two_skill_setup()
    tier3 = _StubTier3()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, tier3=tier3
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="thanks", message=message
    )

    assert outcome == DispatchOutcome.PASS
    assert harness.calls == []
    assert message.replies == []
    assert tier3.turns == []  # no synthesis ⇒ no tier3 write


async def test_multi_step_step_cap_forces_termination() -> None:
    """Planner that always returns tool_call must stop at max_steps."""
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
        ],
    )
    registry, harness = _two_skill_setup()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, max_steps=3
    )

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="loop please", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(harness.calls) == 3  # exactly max_steps
    assert len(runner.plan_next_calls) == 3  # no extra plan_next after cap


async def test_multi_step_mid_chain_error_thread_into_history() -> None:
    """Tool raise → harness wraps as error ToolResult → planner sees it → respond."""
    registry, _ = _two_skill_setup()

    def _boom(args: dict) -> dict:
        raise PermissionError("denied")

    harness = _RecordingHarness(
        tools={
            "files_search": _boom,
            "files_read": lambda args: {"content": "x"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find x", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # The respond-call saw the error result threaded into history.
    second_call_history = runner.plan_next_calls[1]["history"]
    assert len(second_call_history) == 1
    _, err_result = second_call_history[0]
    assert err_result.status == "error"
    assert "denied" in (err_result.error or "")


async def test_multi_step_synthesis_failure_falls_back_to_last_payload() -> None:
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="respond"),
        ],
    )
    registry, harness = _two_skill_setup()
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        synthesizer=_RaisingSynthesizer(),
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # Fallback uses last payload — the matches dict from files_search.
    assert "matches" in message.replies[0]
    assert len(message.replies[0]) <= 3500


async def test_multi_step_chain_synthesis_runs_verdict_gate() -> None:
    """Synthesizer claims an unran action — the verdict gate must annotate."""
    from runtime.chat.reply_verdict import UNVERIFIED_BANNER

    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )
    registry, harness = _two_skill_setup()
    rogue = _StubSynthesizer(reply="Done — I deleted the file you asked about.")
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=rogue
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and read a.md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # Chain ran files_search + files_read; reply claims a delete → flag.
    assert message.replies[0].startswith(UNVERIFIED_BANNER)
    assert "deleted the file" in message.replies[0]


async def test_multi_step_respond_path_verified_tools_excludes_failed_tool() -> None:
    """The respond-path verdict gate (non-task_complete chain) must exclude
    a tool that FAILED this turn from its verified-tools set — otherwise a
    model can dodge `_gate_completion`'s stricter scrutiny by emitting
    `respond` instead of `task_complete` and still get a false claim about
    the failed tool trusted (Phase 11 whole-branch review, I3). Mirrors
    `test_multi_step_chain_synthesis_runs_verdict_gate` above but the
    claimed tool actually errored, rather than never having run at all."""
    registry, _ = _two_skill_setup()

    def _boom(args: dict[str, Any]) -> dict[str, Any]:
        raise PermissionError("denied")

    harness = _RecordingHarness(
        tools={
            "files_search": _boom,
            "files_read": lambda args: {"content": "hello"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="respond"),
        ],
    )
    rogue = _StubSynthesizer(reply="Done — I searched the files for you.")
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=rogue
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # files_search errored this turn — a status-blind verified set (the
    # pre-fix bug) would still count it "verified", letting "I searched"
    # through unflagged.
    assert message.replies[0].startswith(UNVERIFIED_BANNER)
    assert "searched the files" in message.replies[0]


async def test_multi_step_respond_path_verified_tools_excludes_soft_failed_run_command() -> None:
    """A tool can also "fail" without raising: `run_command` never raises on
    a non-zero exit — it returns `status="ok"` with its OWN
    `payload["verdict"] == "exit_nonzero"`. `_history_verified_tools` must
    defer to `verdict_for_result` (which already understands the payload
    verdict via C4) rather than a bare `res.status == "ok"` check, or this
    soft-failure shape slips past the respond-path gate the same way a
    raised exception used to (Phase 11 review follow-up to C4/I3)."""
    registry, _ = _two_skill_setup()

    def _soft_fail_search(args: dict[str, Any]) -> dict[str, Any]:
        # run_command-shaped: the harness sees status="ok" (it ran to
        # completion, nothing raised) but the tool's own verdict says it
        # failed.
        return {
            "argv": ["grep", "nope", "/tmp/haystack.txt"],
            "exit_code": 1,
            "stdout_tail": "",
            "verdict": "exit_nonzero",
        }

    harness = _RecordingHarness(
        tools={
            "files_search": _soft_fail_search,
            "files_read": lambda args: {"content": "hello"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="respond"),
        ],
    )
    rogue = _StubSynthesizer(reply="Done — I searched the files for you.")
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=rogue
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # files_search "succeeded" (status="ok") but its own verdict says
    # exit_nonzero — a status-only check would still count it "verified"
    # and let "I searched" through unflagged.
    assert message.replies[0].startswith(UNVERIFIED_BANNER)
    assert "searched the files" in message.replies[0]


# ---------------------------------------------------------------------------
# Destructive-tool guard — Step 4 of docs/PLAN_MULTI_STEP_AGENT_LOOP.md.
# Allowed at step 1 (operator's explicit opening request); intercepted at
# step 2+ with a deterministic confirmation prompt (no LLM).
# ---------------------------------------------------------------------------


def _destructive_setup(
    destructive_tool: str = "files_delete",
) -> tuple[_MultiSkillRegistry, _RecordingHarness]:
    """Registry that exposes both files_search (non-destructive, executes) and
    a destructive tool descriptor. Harness has both stubs; the test asserts
    the destructive stub is NEVER invoked when intercepted."""
    search = SkillDescriptor(
        id="search_files",
        description="Search for files matching a glob.",
        intents=["search_files"],
        tool="files_search",
        args_schema={"type": "object", "properties": {"glob": {"type": "string"}}},
        requires_tier1=True,
    )
    destructive = SkillDescriptor(
        id=destructive_tool,
        description=f"Destructive: {destructive_tool}.",
        intents=[destructive_tool],
        tool=destructive_tool,
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )
    registry = _MultiSkillRegistry(
        [search, destructive], primary_intent="search_files"
    )
    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": ["/tmp/a.md"]},
            destructive_tool: lambda args: {"removed": args.get("path")},
        }
    )
    return registry, harness


async def test_destructive_at_step_1_is_allowed() -> None:
    """Operator's opening request CAN be destructive — no interception."""
    registry, harness = _destructive_setup("files_delete")
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_delete", args={"path": "/tmp/x"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, registry=registry, harness=harness)
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="delete /tmp/x", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_delete"]
    assert len(message.replies) == 1
    # Synthesizer ran (the stub returns "chain reply"); guard did NOT trip.
    assert "I'd like to run" not in message.replies[0]


async def test_destructive_at_step_2_is_intercepted() -> None:
    """search → delete: only search executes; deterministic confirm message ships."""
    registry, harness = _destructive_setup("files_delete")
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_delete", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )

    class _ExplodingSynth:
        """If the synthesizer is ever called on the guarded path, fail loudly."""

        async def chat(self, request: Any) -> Any:
            raise AssertionError("synthesizer must not be called on destructive guard")

    tier3 = _StubTier3()
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        synthesizer=_ExplodingSynth(),
        tier3=tier3,
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search"]
    assert len(message.replies) == 1
    reply = message.replies[0]
    assert "files_delete" in reply
    assert "/tmp/a.md" in reply
    # Tier3 records both turns even on guard path (mirrors CLARIFY semantics).
    assert tier3.turns == [
        ("42", "user", "find and delete a.md"),
        ("42", "bot", reply),
    ]


async def test_destructive_confirmation_message_is_deterministic() -> None:
    """Confirmation message must contain literal tool id + args, no filler."""
    registry, harness = _destructive_setup("files_move")
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="tool_call", tool="files_move", args={"src": "/a", "dst": "/b"}),
            PlanStep(kind="respond"),
        ],
    )

    class _ExplodingSynth:
        async def chat(self, request: Any) -> Any:
            raise AssertionError("synthesizer must not be called")

    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=_ExplodingSynth(),
    )
    message = _FakeMessage()

    await dispatcher.dispatch(
        chat_id=1, user_text="move things", message=message
    )

    reply = message.replies[0]
    # Deterministic prefix from _format_destructive_confirmation:
    assert reply.startswith("⚠️ I'd like to run `files_move`")
    assert "/a" in reply and "/b" in reply
    assert "please confirm" in reply.lower()


@pytest.mark.parametrize("dtool", ["files_delete", "files_move", "files_write"])
async def test_destructive_guard_covers_all_destructive_tools(dtool: str) -> None:
    registry, harness = _destructive_setup(dtool)
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="tool_call", tool=dtool, args={"path": "/tmp/x"}),
            PlanStep(kind="respond"),
        ],
    )

    class _ExplodingSynth:
        async def chat(self, request: Any) -> Any:
            raise AssertionError(f"synthesizer must not run for guarded {dtool}")

    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=_ExplodingSynth(),
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text=f"chain into {dtool}", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search"]
    assert dtool in message.replies[0]


async def test_destructive_guard_composes_with_real_files_write(tmp_path: Path) -> None:
    """D1 (confirmation guard) + D2 (files_write) integration.

    The guard holds a REAL `files_write` step proposed at step 2 (no stub
    standing in for the tool); confirming with "yes" then executes the
    actual wrapped `FilesClient.write_file` against a tmp root and the file
    lands on disk. Proves the full destructive-tool composition end to end,
    not guard-only or tool-only in isolation.
    """
    root = tmp_path / "root"
    root.mkdir()
    files_client = FilesClient(allowed_roots=[root])
    file_tools = make_files_tools(files_client)

    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": []},
            "files_write": file_tools["files_write"],  # the REAL wrapper
        }
    )

    search_skill = SkillDescriptor(
        id="search_files",
        description="Search for files matching a glob.",
        intents=["search_files"],
        tool="files_search",
        args_schema={"type": "object", "properties": {"glob": {"type": "string"}}},
        requires_tier1=True,
    )
    write_skill = SkillDescriptor(
        id="write_file",
        description="Write text content to a file.",
        intents=["write_file"],
        tool="files_write",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        },
        requires_tier1=True,
    )
    registry = _MultiSkillRegistry([search_skill, write_skill], primary_intent="search_files")

    target = root / "notes.txt"
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(
                kind="tool_call",
                tool="files_write",
                args={"path": str(target), "content": "hello world"},
            ),
            PlanStep(kind="respond"),
        ],
    )

    class _ExplodingSynth:
        async def chat(self, request: Any) -> Any:
            raise AssertionError("synthesizer must not run before confirmation")

    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=_ExplodingSynth(),
    )

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and write notes.txt", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search"]  # write NOT run yet
    assert not target.exists()

    message2 = _FakeMessage()
    outcome2 = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert outcome2 == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search", "files_write"]
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world"


# ---------------------------------------------------------------------------
# Evidence ledger wiring (B2) — real EventStream backed by tmp_path.
# `aegis_sandbox` (conftest, autouse) already points AEGIS_HOME/AEGIS_ROOT at
# tmp_path; we build the EventStream against its own tmp_path subdir directly
# rather than relying on any AEGIS-config-derived sessions dir.
# ---------------------------------------------------------------------------


async def test_multi_step_records_one_tool_call_per_step_same_turn_id(
    tmp_path: Path,
) -> None:
    events = EventStream(tmp_path / "sessions")
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and read", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    records = load_tool_calls(events)
    assert len(records) == 2
    assert [r.tool for r in records] == ["files_search", "files_read"]
    # Every record from this dispatch shares one turn_id (imp_id).
    imp_ids = {r.imp_id for r in records}
    assert len(imp_ids) == 1
    (imp_id,) = imp_ids
    assert imp_id.startswith("turn-42-")


async def test_multi_step_records_every_step_including_errors(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    registry, _ = _two_skill_setup()

    def _boom(args: dict) -> dict:
        raise PermissionError("denied")

    harness = _RecordingHarness(
        tools={
            "files_search": _boom,
            "files_read": lambda args: {"content": "x"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )

    await dispatcher.dispatch(chat_id=1, user_text="find x", message=_FakeMessage())

    records = load_tool_calls(events)
    assert len(records) == 1
    assert records[0].tool == "files_search"
    assert records[0].verdict == "tool_error"


async def test_single_shot_records_verified_on_success(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
    )
    dispatcher._events = events

    await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=_FakeMessage()
    )

    records = load_tool_calls(events)
    assert len(records) == 1
    assert records[0].tool == "files_list"
    assert records[0].verdict == "verified"


async def test_single_shot_records_tool_error_on_failure(tmp_path: Path) -> None:
    def _raise_perm(*_: object) -> dict:
        raise PermissionError("denied")

    error_harness = HarnessAdapter(tools={"files_list": _raise_perm})
    events = EventStream(tmp_path / "sessions")
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        harness=error_harness,
    )
    dispatcher._events = events

    await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=_FakeMessage()
    )

    records = load_tool_calls(events)
    assert len(records) == 1
    assert records[0].tool == "files_list"
    assert records[0].verdict == "tool_error"


async def test_non_json_serializable_payload_does_not_break_dispatch_or_ledger(
    tmp_path: Path,
) -> None:
    # A tool payload containing a value json.dumps can't natively serialize
    # (a set here) must never raise out of dispatch() — recording is
    # telemetry, not the product. The ledger still gets a record, with a
    # verdict derived from the (successful) ToolResult and a positive
    # outcome_bytes from the `default=str`-serialized fallback.
    odd_harness = HarnessAdapter(tools={"files_list": lambda args: {"paths": {1, 2}}})
    events = EventStream(tmp_path / "sessions")
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
        harness=odd_harness,
    )
    dispatcher._events = events

    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=_FakeMessage()
    )

    assert outcome == DispatchOutcome.FIRED
    records = load_tool_calls(events)
    assert len(records) == 1
    assert records[0].tool == "files_list"
    assert records[0].verdict == "verified"
    assert records[0].outcome_bytes > 0


async def test_two_dispatches_produce_two_turn_ids(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    dispatcher = _make_dispatcher(
        classifier=_StubClassifier("list_files", 0.9),
    )
    dispatcher._events = events

    await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=_FakeMessage()
    )
    await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads again", message=_FakeMessage()
    )

    records = load_tool_calls(events)
    assert len(records) == 2
    imp_ids = {r.imp_id for r in records}
    assert len(imp_ids) == 2


async def test_events_none_by_default_does_not_record() -> None:
    # Default constructor path (no `events=`) must dispatch exactly as before —
    # no EventStream means no ledger writes and no behavior change.
    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.9))
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=123, user_text="list my downloads", message=message
    )
    assert outcome == DispatchOutcome.FIRED
    assert dispatcher._events is None


# ---------------------------------------------------------------------------
# task_complete completion gate (Track C, C2).
#
# `task_complete` no longer falls through to chain synthesis: `_gate_completion`
# checks the claimed summary against this turn's evidence (ledger when an
# EventStream is wired, else in-memory history) and the gated text becomes
# the reply directly. See docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md Track C.
# ---------------------------------------------------------------------------


async def test_multi_step_task_complete_gates_and_skips_chain_synthesis() -> None:
    """Planner: search → task_complete. C2 supersedes the C1 pin: the gated
    summary becomes the reply directly and chain synthesis never runs."""
    recorder = _RecordingSynthesizer()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete", summary="Found the markdown files."),
        ],
    )
    registry, harness = _two_skill_setup()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, synthesizer=recorder
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(harness.calls) == 1
    # Loop broke on task_complete — no third plan_next call was made.
    assert len(runner.plan_next_calls) == 2
    # Chain synthesis (the stub synthesizer) was never invoked.
    assert recorder.calls == 0
    # Reply is the gated summary verbatim — no unverified claim, no failed
    # tools, so no annotation is added.
    assert message.replies == ["Found the markdown files."]


async def test_task_complete_empty_history_returns_pass() -> None:
    """task_complete with nothing run yet behaves exactly like respond with
    no history: PASS, no reply, no tier3 write."""
    runner = _StubPlanRunner(
        plan_steps=[PlanStep(kind="task_complete", summary="Nothing to report.")],
    )
    registry, harness = _two_skill_setup()
    tier3 = _StubTier3()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, tier3=tier3
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="thanks", message=message
    )

    assert outcome == DispatchOutcome.PASS
    assert harness.calls == []
    assert message.replies == []
    assert tier3.turns == []


async def test_task_complete_failed_tool_gets_warning_and_emits_gated_event(
    tmp_path: Path,
) -> None:
    """A failed tool in history must annotate the summary with the ⚠️ note
    AND emit HARNESS_COMPLETION_GATED — verified via a real EventStream."""
    events = EventStream(tmp_path / "sessions")
    registry, _ = _two_skill_setup()

    def _boom(args: dict) -> dict:
        raise PermissionError("denied")

    harness = _RecordingHarness(
        tools={
            "files_search": _boom,
            "files_read": lambda args: {"content": "x"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="task_complete", summary="Searched for the files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find x", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    reply = message.replies[0]
    assert "Searched for the files." in reply
    assert "⚠️ Note: files_search did not complete successfully" in reply

    raw_events = [
        json.loads(line)
        for line in events.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gated = [e for e in raw_events if e["type"] == "harness.completion_gated"]
    assert len(gated) == 1
    assert gated[0]["payload"]["failed_tools"] == ["files_search"]


async def test_task_complete_soft_failed_run_command_gets_warning_and_gated_event(
    tmp_path: Path,
) -> None:
    """The `failed` set must also defer to `verdict_for_result`, not a bare
    status check. A `run_command` soft-failure (status="ok" + payload
    verdict="exit_nonzero") must appear in the ⚠️ warning and the
    HARNESS_COMPLETION_GATED `failed_tools` payload — otherwise it drops out
    of BOTH `verified` and `failed`, getting only the generic unverified
    banner and never the specific 'that command failed' note (Phase 11
    review follow-up: symmetric to the _history_verified_tools fix)."""
    events = EventStream(tmp_path / "sessions")
    registry, _ = _two_skill_setup()

    def _soft_fail_search(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "argv": ["grep", "nope", "/tmp/haystack.txt"],
            "exit_code": 1,
            "stdout_tail": "",
            "verdict": "exit_nonzero",
        }

    harness = _RecordingHarness(
        tools={
            "files_search": _soft_fail_search,
            "files_read": lambda args: {"content": "x"},
        }
    )
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="task_complete", summary="Searched for the files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find x", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    reply = message.replies[0]
    assert "⚠️ Note: files_search did not complete successfully" in reply

    raw_events = [
        json.loads(line)
        for line in events.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gated = [e for e in raw_events if e["type"] == "harness.completion_gated"]
    assert len(gated) == 1
    assert gated[0]["payload"]["failed_tools"] == ["files_search"]


async def test_task_complete_ledger_backed_verified_set_flags_unran_tool_claim(
    tmp_path: Path,
) -> None:
    """With events wired, the gate's verified set comes from this turn's
    ledger records: a summary claiming an unexecuted tool (files_delete,
    which never ran) gets the unverified-claim annotation."""
    events = EventStream(tmp_path / "sessions")
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(
                kind="task_complete",
                summary="Done — I deleted the file you asked about.",
            ),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find and delete a.md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    assert message.replies[0].startswith(UNVERIFIED_BANNER)
    assert "deleted the file" in message.replies[0]


async def test_task_complete_events_none_falls_back_to_history() -> None:
    """No EventStream wired — the gate's verified/failed sets must come from
    in-memory history, and the failed-tool annotation still fires."""

    def _boom(args: dict) -> dict:
        raise PermissionError("denied")

    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": ["/tmp/a.md"]},
            "files_read": _boom,
        }
    )
    registry, _ = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="task_complete", summary="Searched and read the file."),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, registry=registry, harness=harness)
    assert dispatcher._events is None
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find and read", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    reply = message.replies[0]
    assert "Searched and read the file." in reply
    assert "⚠️ Note: files_read did not complete successfully" in reply


async def test_task_complete_ledger_read_failure_falls_back_to_history() -> None:
    """`_gate_completion`'s ledger read must be guarded like every other
    ledger read in this module (Phase 11 whole-branch review, C3) — a disk
    error on a task_complete turn must not escape `dispatch()` (bot.py
    re-raises; there is no PTB error handler). Falls back to the in-memory
    history-derived verified set, same as the events=None path exercised
    above. Mirrors `chat/pipeline.py::_gate_reply`'s guarded read."""

    class _RaisingLoadEventStream:
        session_id = "raising-session"

        @property
        def path(self) -> Path:
            raise OSError("disk full (simulated)")

        def append(self, event_type: Any, payload: dict[str, Any]) -> None:
            raise OSError("disk full (simulated)")

    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete", summary="Searched for the files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        events=_RaisingLoadEventStream(),  # type: ignore[arg-type]
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    # files_search ran and succeeded — the history fallback finds it
    # verified, so the plain summary ships with no unverified-claim banner
    # and no exception ever escaped the ledger read.
    assert message.replies[0] == "Searched for the files."


async def test_task_complete_identical_args_retry_still_warns_oq7(tmp_path: Path) -> None:
    """OQ7 pin (Phase 11 whole-branch review, I2) — do NOT change
    record.py's idempotency key to make this test pass differently; that
    fix is tracked separately for a later phase.

    `_gate_completion`'s ledger-backed verified set is scoped by
    `record_tool_call`'s composite key `(session, imp_id, skill, tool,
    argv_hash)`. When a tool fails then succeeds with IDENTICAL args in
    the same turn, the successful retry's argv_hash matches the
    already-recorded failed attempt's key, so `record_tool_call` returns
    `"skipped_idempotent"` and writes nothing — the ledger ends up with
    ONLY the failed record. The gate's `failed - verified` compensation
    only works when a retry uses DIFFERENT args; for identical args it
    still warns, even though the retry actually succeeded. This is a
    known, conservative-direction over-warning (spurious, not a
    false-negative) — pinning current behaviour, not endorsing it."""
    events = EventStream(tmp_path / "sessions")
    registry, _ = _two_skill_setup()

    call_count = {"n": 0}

    def _flaky_search(args: dict[str, Any]) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise PermissionError("denied")
        return {"matches": ["/tmp/a.md"]}

    harness = _RecordingHarness(tools={"files_search": _flaky_search})
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete", summary="Found the markdown files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert call_count["n"] == 2  # failed once, then succeeded, identical args
    records = load_tool_calls(events)
    assert len(records) == 1  # the successful retry was skipped-idempotent
    assert records[0].verdict == "tool_error"
    reply = message.replies[0]
    assert "⚠️ Note: files_search did not complete successfully" in reply


# ---------------------------------------------------------------------------
# Completion gate review follow-ups (quality review of ca7c5f9).
# ---------------------------------------------------------------------------


async def test_task_complete_blank_summary_defaults_to_done() -> None:
    """A 2B planner can emit task_complete with NO summary at all after a
    successful tool call. `plan.summary or ""` alone would yield an empty
    reply text, which crashes the send path (python-telegram-bot rejects
    empty message text). The gate must default to a minimal non-empty
    claim instead."""
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete"),  # summary=None
        ],
    )
    registry, harness = _two_skill_setup()
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    assert message.replies[0]  # non-empty — no crash
    assert "Done." in message.replies[0]


async def test_task_complete_recovered_tool_not_branded_failure(
    tmp_path: Path,
) -> None:
    """A tool that fails on step 1 and succeeds on a later retry (varied
    args, so the ledger records the success under a different argv_hash
    rather than being deduped by the failed call's idempotency key) must
    NOT be reported as a failure: its later verification supersedes the
    earlier error. No ⚠️ warning, no HARNESS_COMPLETION_GATED event, and
    the tool is NOT treated as unverified by the claim-annotation gate."""
    events = EventStream(tmp_path / "sessions")
    registry, _ = _two_skill_setup()

    def _search(args: dict) -> dict:
        if args.get("glob") == "*":
            raise PermissionError("denied")
        return {"matches": ["/tmp/a.md"]}

    harness = _RecordingHarness(tools={"files_search": _search})
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete", summary="Found the markdown files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(message.replies) == 1
    reply = message.replies[0]
    assert reply == "Found the markdown files."
    assert "⚠️" not in reply

    raw_events = [
        json.loads(line)
        for line in events.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gated = [e for e in raw_events if e["type"] == "harness.completion_gated"]
    assert gated == []


async def test_task_complete_verified_set_scoped_to_current_turn_id(
    tmp_path: Path,
) -> None:
    """The completion gate's ledger query must scope to THIS turn's
    turn_id. A tool verified on a PRIOR turn must not count as evidence
    for a later turn's task_complete claim — proves the exclusion the
    whole gate hinges on."""
    events = EventStream(tmp_path / "sessions")
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            # Turn 1: search runs, then respond — records files_search
            # under turn 1's turn_id.
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="respond"),
            # Turn 2: only files_read runs, then task_complete claims a
            # files_search action — that tool never ran on THIS turn.
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(
                kind="task_complete",
                summary="Done — I searched for the file you wanted.",
            ),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner, registry=registry, harness=harness, events=events
    )

    turn1_outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=_FakeMessage()
    )
    assert turn1_outcome == DispatchOutcome.FIRED

    message2 = _FakeMessage()
    turn2_outcome = await dispatcher.dispatch(
        chat_id=1, user_text="now read it", message=message2
    )

    assert turn2_outcome == DispatchOutcome.FIRED
    assert len(message2.replies) == 1
    assert message2.replies[0].startswith(UNVERIFIED_BANNER)
    assert "searched" in message2.replies[0].lower()

    # Sanity: two distinct turn_ids were recorded across the two dispatches.
    records = load_tool_calls(events)
    imp_ids = {r.imp_id for r in records}
    assert len(imp_ids) == 2


# ---------------------------------------------------------------------------
# Pending-confirmation state for the destructive guard — Phase 11 Track D,
# D1. A guarded intent is held (in-memory, per chat_id) rather than dropped;
# an explicit "yes" within the TTL executes it without re-planning. See
# docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md Track D.
# ---------------------------------------------------------------------------


def _raw_events(events: EventStream) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in events.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _guard_trip_runner() -> _StubPlanRunner:
    """search (step 1) → files_delete (step 2, guarded) → respond."""
    return _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="tool_call", tool="files_delete", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )


async def test_confirmation_guard_stores_pending_and_emits_requested_event(
    tmp_path: Path,
) -> None:
    events = EventStream(tmp_path / "sessions")
    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(), registry=registry, harness=harness, events=events
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search"]  # delete NOT run yet
    assert len(message.replies) == 1
    reply = message.replies[0]
    assert "files_delete" in reply

    pending = dispatcher._pending.get(42)
    assert pending is not None
    assert pending.intent.tool == "files_delete"
    assert pending.intent.args == {"path": "/tmp/a.md"}
    # skill_id is the descriptor CLASSIFIED for this turn ("search_files" —
    # the same skill_id `_run_multi_step` used for every call in the chain),
    # not the guarded tool's own id.
    assert pending.skill_id == "search_files"
    assert pending.user_text == "find and delete a.md"

    requested = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_requested"
    ]
    assert len(requested) == 1
    assert requested[0]["payload"] == {"tool": "files_delete", "skill_id": "search_files"}


async def test_confirmation_yes_within_ttl_executes_once(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )
    assert [c.tool for c in harness.calls] == ["files_search"]

    clock.now = clock.now + timedelta(seconds=5)
    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]
    assert 42 not in dispatcher._pending
    assert len(message2.replies) == 1

    records = load_tool_calls(events)
    delete_records = [r for r in records if r.tool == "files_delete"]
    assert len(delete_records) == 1

    accepted = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_accepted"
    ]
    assert len(accepted) == 1
    assert accepted[0]["payload"] == {"tool": "files_delete", "skill_id": "search_files"}


@pytest.mark.parametrize("affirmative_text", ["  YES  ", "  yes\n", "Yes"])
async def test_confirmation_whitespace_case_variants_accepted(
    tmp_path: Path, affirmative_text: str
) -> None:
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )

    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(
        chat_id=42, user_text=affirmative_text, message=message2
    )

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]


async def test_confirmation_declined_falls_through_to_normal_dispatch(
    tmp_path: Path,
) -> None:
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    classifier = _StubClassifier("search_files", 0.9)
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
        classifier=classifier,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )
    assert classifier.calls == ["find and delete a.md"]

    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="no thanks", message=message2)

    # Declined — falls through to NORMAL dispatch: the classifier is
    # consulted again for the follow-up text itself.
    assert classifier.calls == ["find and delete a.md", "no thanks"]
    assert 42 not in dispatcher._pending
    assert [c.tool for c in harness.calls] == ["files_search"]  # delete never ran
    assert outcome == DispatchOutcome.PASS  # remaining queued step is "respond"

    declined = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_declined"
    ]
    assert len(declined) == 1
    assert declined[0]["payload"] == {"tool": "files_delete", "expired": False}


async def test_confirmation_expired_falls_through_and_declines(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    classifier = _StubClassifier("search_files", 0.9)
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
        classifier=classifier,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )

    clock.now = clock.now + timedelta(seconds=121)  # just past the 120s TTL
    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert 42 not in dispatcher._pending
    assert [c.tool for c in harness.calls] == ["files_search"]  # delete NEVER executed
    assert outcome == DispatchOutcome.PASS
    # Falls through to normal dispatch — classifier consulted for "yes" too.
    assert classifier.calls == ["find and delete a.md", "yes"]

    declined = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_declined"
    ]
    assert len(declined) == 1
    assert declined[0]["payload"] == {"tool": "files_delete", "expired": True}


async def test_confirmation_consumed_once_second_yes_does_not_reexecute(
    tmp_path: Path,
) -> None:
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    classifier = _StubClassifier("search_files", 0.9)
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
        classifier=classifier,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )
    await dispatcher.dispatch(chat_id=42, user_text="yes", message=_FakeMessage())
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]
    assert classifier.calls == ["find and delete a.md"]  # accepted path skips classify

    message3 = _FakeMessage()
    await dispatcher.dispatch(chat_id=42, user_text="yes", message=message3)

    # Pending was already consumed by the first "yes" — this second "yes" is
    # just an ordinary message: classifier is consulted, and files_delete is
    # NOT executed again.
    assert classifier.calls == ["find and delete a.md", "yes"]
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]

    accepted = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_accepted"
    ]
    assert len(accepted) == 1  # not two


async def test_confirmation_ttl_boundary_at_exactly_120s_is_accepted(
    tmp_path: Path,
) -> None:
    """`age <= _CONFIRMATION_TTL_S` is inclusive — exactly 120s must still
    accept, not just anything strictly under."""
    events = EventStream(tmp_path / "sessions")
    clock = _MutableClock(datetime(2026, 1, 1, tzinfo=UTC))
    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        clock=clock,
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )

    clock.now = clock.now + timedelta(seconds=120)  # exactly at the TTL boundary
    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]


async def test_confirmation_execute_synthesis_failure_falls_back_to_raw_payload(
    tmp_path: Path,
) -> None:
    """`_execute_confirmed` must inherit the same synthesis-failure fallback
    as the single-shot path (`_synthesize`'s own try/except): a raising
    synthesizer still yields a reply (the clipped raw tool payload) rather
    than losing the turn, and the ledger record + ACCEPTED event are
    unaffected by the synthesis failure. Pins the code-reuse — a future
    rewrite of `_execute_confirmed` that stops calling `_synthesize` would
    silently lose this fallback."""
    events = EventStream(tmp_path / "sessions")
    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        synthesizer=_RaisingSynthesizer(),
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )

    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert outcome == DispatchOutcome.FIRED
    assert len(message2.replies) == 1
    # Fallback is the raw tool payload (files_delete's stub returns
    # {"removed": <path>}) — not a synthesized sentence.
    assert "removed" in message2.replies[0]
    assert len(message2.replies[0]) <= 3500

    records = load_tool_calls(events)
    delete_records = [r for r in records if r.tool == "files_delete"]
    assert len(delete_records) == 1

    accepted = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_accepted"
    ]
    assert len(accepted) == 1


async def test_confirmation_guard_send_failure_does_not_arm_pending(
    tmp_path: Path,
) -> None:
    """If sending the confirmation prompt itself raises (e.g. a Telegram API
    error), no pending intent may be armed. Otherwise a later UNRELATED
    message that happens to match an affirmative ("yes", "go ahead" are
    common words) would silently execute a destructive tool the operator
    was never shown a prompt for."""
    events = EventStream(tmp_path / "sessions")
    registry, harness = _destructive_setup("files_delete")
    classifier = _StubClassifier("search_files", 0.9)
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=events,
        classifier=classifier,
    )

    async def _raising_reply(text: str) -> None:
        raise RuntimeError("telegram api error")

    with pytest.raises(RuntimeError, match="telegram api error"):
        await dispatcher.dispatch(
            chat_id=42,
            user_text="find and delete a.md",
            message=_FakeMessage(),
            reply=_raising_reply,
        )

    assert 42 not in dispatcher._pending
    requested = [
        e for e in _raw_events(events) if e["type"] == "harness.confirmation_requested"
    ]
    assert requested == []  # never armed — send never succeeded

    # A following "yes" is just an ordinary message: classifier consulted,
    # destructive tool never executes.
    message2 = _FakeMessage()
    await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert classifier.calls[-1] == "yes"
    assert [c.tool for c in harness.calls] == ["files_search"]  # delete never ran


async def test_confirmation_accepted_survives_event_stream_append_failure() -> None:
    """`EventStream.append` does unguarded file I/O and can raise (disk
    full, permissions, ...). HARNESS_CONFIRMATION_ACCEPTED fires AFTER a
    destructive tool has already mutated disk — an unguarded raise there
    must not propagate out of `dispatch()`: the operator still needs a
    reply about an action that already happened."""

    class _RaisingEventStream:
        session_id = "raising-session"
        path = Path("/nonexistent/raising.jsonl")

        def append(self, event_type: Any, payload: dict[str, Any]) -> None:
            raise OSError("disk full (simulated)")

    registry, harness = _destructive_setup("files_delete")
    dispatcher = _make_loop_dispatcher(
        runner=_guard_trip_runner(),
        registry=registry,
        harness=harness,
        events=_RaisingEventStream(),  # type: ignore[arg-type]
    )
    await dispatcher.dispatch(
        chat_id=42, user_text="find and delete a.md", message=_FakeMessage()
    )
    # Pending is stored regardless of the (also-swallowed) REQUESTED event
    # append failure — arming happens before the event append, per Fix 1.
    assert 42 in dispatcher._pending

    message2 = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=42, user_text="yes", message=message2)

    assert outcome == DispatchOutcome.FIRED
    assert [c.tool for c in harness.calls] == ["files_search", "files_delete"]
    assert len(message2.replies) == 1


# ---------------------------------------------------------------------------
# Existing behavior untouched: no pending state ⇒ dispatch runs exactly as
# before (regression guard for the new entry check added at the top of
# `dispatch()`).
# ---------------------------------------------------------------------------


async def test_no_pending_confirmation_does_not_affect_normal_dispatch() -> None:
    dispatcher = _make_dispatcher(classifier=_StubClassifier("list_files", 0.9))
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=99, user_text="list my downloads", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert dispatcher._pending == {}
