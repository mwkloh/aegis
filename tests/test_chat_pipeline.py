"""Phase 7 §6 step 13 — `ChatPipeline` composition tests.

Pins:

* Happy path: LLM called with [system, ...history, user]; reply
  returned; tier-3 gets both user+bot turns appended; `chat.turn.context`
  + `chat.turn.reply` events emitted with structural counts only.
* LLM failure → `STUB_REPLY` returned; turn still recorded in tier 3;
  events emit with `errored=True`.
* Empty-ish LLM reply (whitespace only) also counts as errored — the
  pipeline refuses to record a blank response as a real reply.
* Recall failure → pipeline catches and proceeds with zero lookups
  (never raises to the caller).
* Empty `user_text` (whitespace) returns `""` without touching the
  model, the stores, or the event stream — cheap short-circuit.
* System prompt renders tier-1 (identity + user + prefs) and the
  retrieved lookups into a single system message.

No network, no sqlite, no embedder — all fakes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from runtime.chat.memory.context_builder import ContextBuilder, Lookup
from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.pipeline import (
    DEFAULT_SYSTEM_PROMPT,
    STUB_REPLY,
    ChatPipeline,
)
from runtime.chat.reply_verdict import UNVERIFIED_BANNER
from runtime.events import EventStream
from runtime.llm.clients.base import (
    ChatRequest,
    ChatResponse,
)
from runtime.tools.record import record_tool_call

pytestmark = pytest.mark.unit


# --- fakes ------------------------------------------------------------


@dataclass
class _FakeRecall:
    canned: tuple[Lookup, ...] = ()
    calls: list[tuple[str, str]] = field(default_factory=list)
    raises: Exception | None = None

    def recall(self, chat_id: str, user_text: str) -> tuple[Lookup, ...]:
        self.calls.append((chat_id, user_text))
        if self.raises is not None:
            raise self.raises
        return self.canned


@dataclass
class _FakeModel:
    canned_reply: str = "hi operator"
    calls: list[ChatRequest] = field(default_factory=list)
    raises: Exception | None = None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if self.raises is not None:
            raise self.raises
        return ChatResponse(content=self.canned_reply, model=request.model)

    async def health(self) -> bool:
        return True


# --- helpers ----------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_pipeline(
    tmp_path: Path,
    *,
    recall: _FakeRecall | None = None,
    model: _FakeModel | None = None,
    events: EventStream | None = None,
    budget_bytes: int = 8 * 1024,
) -> tuple[ChatPipeline, _FakeRecall, _FakeModel, Tier3Store]:
    _write(tmp_path / "IDENTITY.md", "You are AEGIS.")
    _write(tmp_path / "USER.md", "Operator: Michael.")
    tier1 = Tier1Loader(tmp_path)
    tier3 = Tier3Store(keep_turns=6)
    builder = ContextBuilder(tier1, tier3, budget_bytes=budget_bytes)
    r = recall if recall is not None else _FakeRecall()
    m = model if model is not None else _FakeModel()
    pipe = ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=r,
        builder=builder,
        model=m,
        model_name="fake-model",
        provider="ollama",
        events=events,
    )
    return pipe, r, m, tier3


def _read_events(stream: EventStream) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in stream.path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# --- tests ------------------------------------------------------------


async def test_happy_path_assembles_prompt_and_records_turn(tmp_path: Path) -> None:
    pipe, recall, model, tier3 = _make_pipeline(
        tmp_path,
        recall=_FakeRecall(
            canned=(
                Lookup(
                    kind="episodic",
                    text="we talked about CT-001 last week",
                    score=0.9,
                    origin="episodic:chat1:2026-04-10",
                ),
            )
        ),
        model=_FakeModel(canned_reply="acknowledged"),
    )
    reply = await pipe.turn("chat1", "what did we decide about CT-001?")
    assert reply == "acknowledged"
    assert recall.calls == [("chat1", "what did we decide about CT-001?")]
    # Exactly one LLM call with system + user.
    assert len(model.calls) == 1
    req = model.calls[0]
    assert req.model == "fake-model"
    roles = [m.role for m in req.messages]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    sys_content = req.messages[0].content
    assert DEFAULT_SYSTEM_PROMPT in sys_content
    assert "You are AEGIS." in sys_content  # tier1 identity rendered
    assert "Operator: Michael." in sys_content  # tier1 user rendered
    assert "CT-001 last week" in sys_content  # lookup rendered
    assert req.messages[-1].content == "what did we decide about CT-001?"
    # Tier 3 now holds user+bot turns in order.
    turns = tier3.recent("chat1")
    assert [(t.role, t.text) for t in turns] == [
        ("user", "what did we decide about CT-001?"),
        ("bot", "acknowledged"),
    ]


async def test_history_replayed_in_chronological_order(tmp_path: Path) -> None:
    pipe, _, model, _ = _make_pipeline(tmp_path)
    await pipe.turn("chat1", "first")
    await pipe.turn("chat1", "second")
    # On turn 2, history should show [user:first, bot:canned, user:second].
    req = model.calls[1]
    roles = [m.role for m in req.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert req.messages[1].content == "first"
    assert req.messages[2].content == "hi operator"
    assert req.messages[3].content == "second"


async def test_llm_failure_returns_stub_and_records_turn(tmp_path: Path) -> None:
    pipe, _, _, tier3 = _make_pipeline(
        tmp_path, model=_FakeModel(raises=RuntimeError("model down"))
    )
    reply = await pipe.turn("chat1", "ping")
    assert reply == STUB_REPLY
    # Turn + stub still recorded so next turn's history is coherent.
    assert [t.text for t in tier3.recent("chat1")] == ["ping", STUB_REPLY]


async def test_empty_llm_reply_counts_as_errored(tmp_path: Path) -> None:
    pipe, _, _, tier3 = _make_pipeline(
        tmp_path, model=_FakeModel(canned_reply="   ")
    )
    reply = await pipe.turn("chat1", "ping")
    assert reply == STUB_REPLY
    assert [t.text for t in tier3.recent("chat1")] == ["ping", STUB_REPLY]


async def test_recall_failure_proceeds_with_zero_lookups(tmp_path: Path) -> None:
    pipe, _, model, _ = _make_pipeline(
        tmp_path,
        recall=_FakeRecall(raises=RuntimeError("tier2 down")),
    )
    reply = await pipe.turn("chat1", "what about the vault?")
    assert reply == "hi operator"
    # System message has no "Retrieved context" section.
    sys_content = model.calls[0].messages[0].content
    assert "Retrieved context" not in sys_content


async def test_empty_user_text_short_circuits(tmp_path: Path) -> None:
    pipe, recall, model, tier3 = _make_pipeline(tmp_path)
    reply = await pipe.turn("chat1", "   ")
    assert reply == ""
    assert recall.calls == []
    assert model.calls == []
    assert tier3.recent("chat1") == ()


async def test_empty_chat_id_short_circuits(tmp_path: Path) -> None:
    pipe, recall, model, _ = _make_pipeline(tmp_path)
    reply = await pipe.turn("", "hello")
    assert reply == ""
    assert recall.calls == []
    assert model.calls == []


async def test_events_emitted_on_success(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "sessions")
    pipe, _, _, _ = _make_pipeline(tmp_path, events=stream)
    await pipe.turn("chat1", "hi")
    events = _read_events(stream)
    types = [e["type"] for e in events]
    assert "chat.turn.context" in types
    assert "chat.turn.reply" in types
    ctx_evt = next(e for e in events if e["type"] == "chat.turn.context")
    assert ctx_evt["payload"]["chat_id"] == "chat1"
    assert ctx_evt["payload"]["errored"] is False
    # No message bodies — only byte counts.
    assert "text" not in ctx_evt["payload"]
    reply_evt = next(e for e in events if e["type"] == "chat.turn.reply")
    assert reply_evt["payload"]["errored"] is False
    assert reply_evt["payload"]["reply_bytes"] > 0
    assert "reply" not in reply_evt["payload"]  # body never persisted


async def test_events_mark_errored_on_llm_failure(tmp_path: Path) -> None:
    stream = EventStream(tmp_path / "sessions")
    pipe, _, _, _ = _make_pipeline(
        tmp_path,
        model=_FakeModel(raises=RuntimeError("boom")),
        events=stream,
    )
    await pipe.turn("chat1", "hello")
    events = _read_events(stream)
    ctx_evt = next(e for e in events if e["type"] == "chat.turn.context")
    reply_evt = next(e for e in events if e["type"] == "chat.turn.reply")
    assert ctx_evt["payload"]["errored"] is True
    assert reply_evt["payload"]["errored"] is True


async def test_lookups_dropped_under_budget(tmp_path: Path) -> None:
    # Tiny budget forces the builder to drop lookups.
    big = Lookup(
        kind="episodic",
        text="x" * 5000,
        score=0.9,
        origin="episodic:big",
    )
    pipe, _, model, _ = _make_pipeline(
        tmp_path,
        recall=_FakeRecall(canned=(big,)),
        budget_bytes=256,  # tier1 alone will exceed this
    )
    await pipe.turn("chat1", "ping")
    sys_content = model.calls[0].messages[0].content
    # Lookup text is > 5000 bytes; budget is 256; must NOT be in system prompt.
    assert "x" * 5000 not in sys_content


async def test_verdict_gate_flags_unverified_claim(tmp_path: Path) -> None:
    # Reply claims an action; no tool.invoked records on the shard.
    stream = EventStream(tmp_path / "sessions")
    pipe, _, _, tier3 = _make_pipeline(
        tmp_path,
        model=_FakeModel(canned_reply="I ran the cleanup tool."),
        events=stream,
    )
    reply = await pipe.turn("chat1", "clean up")
    assert reply.startswith(UNVERIFIED_BANNER)
    assert "I ran the cleanup tool." in reply
    # Tier 3 stores the annotated reply so history reflects what was shown.
    bot_turn = next(t for t in tier3.recent("chat1") if t.role == "bot")
    assert bot_turn.text.startswith(UNVERIFIED_BANNER)


async def test_verdict_gate_passes_claim_when_verified_on_record(
    tmp_path: Path,
) -> None:
    stream = EventStream(tmp_path / "sessions")
    # Seed a verified tool.invoked on the same shard.
    record_tool_call(
        stream,
        imp_id="IMP-0000aaaa",
        skill="cleanup",
        tool="run",
        argv_hash="aaaa",
        verdict="verified",
        outcome_bytes=42,
    )
    pipe, _, _, _ = _make_pipeline(
        tmp_path,
        model=_FakeModel(canned_reply="I ran the cleanup tool."),
        events=stream,
    )
    reply = await pipe.turn("chat1", "clean up")
    assert reply == "I ran the cleanup tool."
    assert UNVERIFIED_BANNER not in reply


async def test_verdict_gate_skipped_without_event_stream(tmp_path: Path) -> None:
    # No events wired → gate must short-circuit to the raw reply.
    pipe, _, _, _ = _make_pipeline(
        tmp_path,
        model=_FakeModel(canned_reply="I ran the tool."),
        events=None,
    )
    reply = await pipe.turn("chat1", "go")
    assert reply == "I ran the tool."


async def test_verdict_gate_passes_conversational_reply(tmp_path: Path) -> None:
    # Non-claim reply with no verified calls — not annotated.
    stream = EventStream(tmp_path / "sessions")
    pipe, _, _, _ = _make_pipeline(
        tmp_path,
        model=_FakeModel(canned_reply="Here's how I'd approach it."),
        events=stream,
    )
    reply = await pipe.turn("chat1", "thoughts?")
    assert reply == "Here's how I'd approach it."


async def test_verdict_gate_ignores_non_verified_tool_records(tmp_path: Path) -> None:
    # An exit_nonzero record does NOT count as verification.
    stream = EventStream(tmp_path / "sessions")
    record_tool_call(
        stream,
        imp_id="IMP-0000aaaa",
        skill="cleanup",
        tool="run",
        argv_hash="aaaa",
        verdict="exit_nonzero",
        outcome_bytes=0,
    )
    pipe, _, _, _ = _make_pipeline(
        tmp_path,
        model=_FakeModel(canned_reply="I ran the cleanup tool."),
        events=stream,
    )
    reply = await pipe.turn("chat1", "clean up")
    assert reply.startswith(UNVERIFIED_BANNER)


def test_model_name_required(tmp_path: Path) -> None:
    _write(tmp_path / "IDENTITY.md", "i")
    tier1 = Tier1Loader(tmp_path)
    tier3 = Tier3Store()
    builder = ContextBuilder(tier1, tier3)
    with pytest.raises(ValueError, match="model_name"):
        ChatPipeline(
            tier1=tier1,
            tier3=tier3,
            recall=_FakeRecall(),
            builder=builder,
            model=_FakeModel(),
            model_name="",
            provider="ollama",
        )


def test_provider_and_model_name_are_exposed(tmp_path: Path) -> None:
    pipe, _, _, _ = _make_pipeline(tmp_path)
    assert pipe.model_name == "fake-model"
    assert pipe.provider == "ollama"


def test_provider_required(tmp_path: Path) -> None:
    _write(tmp_path / "IDENTITY.md", "i")
    tier1 = Tier1Loader(tmp_path)
    tier3 = Tier3Store()
    builder = ContextBuilder(tier1, tier3)
    with pytest.raises(ValueError, match="provider"):
        ChatPipeline(
            tier1=tier1,
            tier3=tier3,
            recall=_FakeRecall(),
            builder=builder,
            model=_FakeModel(),
            model_name="fake-model",
            provider="",
        )
