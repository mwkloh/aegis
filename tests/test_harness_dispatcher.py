"""Unit tests for HarnessDispatcher — all stubs, no network, no filesystem."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from runtime.chat.memory.tier1 import Tier1Loader, Tier1Snapshot
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.telegram.harness_dispatcher import DispatchOutcome, HarnessDispatcher
from runtime.events import EventStream
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
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
) -> HarnessDispatcher:
    if registry is None or harness is None:
        registry, harness = _two_skill_setup()
    return HarnessDispatcher(
        classifier=_StubClassifier("search_files", 0.9),
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
