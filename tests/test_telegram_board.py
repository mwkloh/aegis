"""BoardRunner — Telegram adapter for `/board`."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from runtime.board.config import BoardConfig, PanelistConfig
from runtime.board.engine import BoardResult, PanelistResponse
from runtime.board.writer import BoardWriter
from runtime.chat.telegram.board_handler import BoardRunner
from runtime.chat.telegram.dispatch import ParsedCommand
from runtime.chat.telegram.long_running import InFlightRegistry

pytestmark = pytest.mark.unit


@dataclass
class _FakeEditableMessage:
    initial_text: str
    edits: list[str] = field(default_factory=list)

    async def edit_text(self, text: str) -> None:
        self.edits.append(text)


@dataclass
class _FakeReplyable:
    replies: list[_FakeEditableMessage] = field(default_factory=list)

    async def reply_text(self, text: str) -> _FakeEditableMessage:
        m = _FakeEditableMessage(initial_text=text)
        self.replies.append(m)
        return m


class _StubEngine:
    def __init__(self, result: BoardResult | None = None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[str] = []
        self.panelist_count = 0 if result is None else len(result.panelist_responses)

    async def run(self, question: str) -> BoardResult:
        self.calls.append(question)
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result


def _result(*, synthesis: str | None = "synth text", error_names: tuple[str, ...] = ()) -> BoardResult:
    responses = tuple(
        PanelistResponse(
            name=name,
            model="m",
            provider="ollama",
            response=("" if name in error_names else "response " + name),
            latency_ms=1000,
            error=("timeout" if name in error_names else None),
        )
        for name in ("Analyst", "Strategist")
    )
    return BoardResult(
        board_id="BOARD-a3f2",
        question="Should we migrate?",
        created_at=datetime(2026, 4, 21, 12, tzinfo=UTC),
        panelist_responses=responses,
        synthesis=synthesis,
    )


async def test_empty_question_emits_usage_reply(tmp_path: Path) -> None:
    engine = _StubEngine()
    writer = BoardWriter(output_dir=tmp_path)
    runner = BoardRunner(engine=engine, writer=writer, registry=InFlightRegistry())
    msg = _FakeReplyable()
    await runner.run(chat_id=1, cmd=ParsedCommand(name="/board", args=()), message=msg)
    assert engine.calls == []
    assert "Usage" in msg.replies[0].initial_text


async def test_not_configured_when_engine_has_zero_panelists(tmp_path: Path) -> None:
    engine = _StubEngine(result=None)
    engine.panelist_count = 0
    writer = BoardWriter(output_dir=tmp_path)
    runner = BoardRunner(engine=engine, writer=writer, registry=InFlightRegistry())
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1, cmd=ParsedCommand(name="/board", args=("hello?",)), message=msg
    )
    assert engine.calls == []
    assert "not configured" in msg.replies[0].initial_text.lower()


async def test_in_flight_second_call_refused(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())
    writer = BoardWriter(output_dir=tmp_path)
    registry = InFlightRegistry()
    registry.try_acquire(99, "/board")
    runner = BoardRunner(engine=engine, writer=writer, registry=registry)
    msg = _FakeReplyable()
    await runner.run(
        chat_id=99, cmd=ParsedCommand(name="/board", args=("q",)), message=msg
    )
    assert engine.calls == []
    assert "Already running" in msg.replies[0].initial_text


async def test_successful_run_edits_message_with_summary_and_path(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result(synthesis="bottom line."))
    writer = BoardWriter(output_dir=tmp_path)
    runner = BoardRunner(engine=engine, writer=writer, registry=InFlightRegistry())
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1, cmd=ParsedCommand(name="/board", args=("Should", "we", "migrate?")), message=msg
    )
    assert engine.calls == ["Should we migrate?"]
    reply = msg.replies[0]
    final = reply.edits[-1]
    assert "BOARD-a3f2" in final
    assert "2 panelists" in final
    assert "Analyst" in final and "Strategist" in final
    assert "bottom line." in final
    assert "Full board →" in final
    assert str(tmp_path) in final


async def test_in_flight_released_on_engine_exception(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result(), raises=RuntimeError("boom"))
    engine.panelist_count = 2
    writer = BoardWriter(output_dir=tmp_path)
    registry = InFlightRegistry()
    runner = BoardRunner(engine=engine, writer=writer, registry=registry)
    msg = _FakeReplyable()
    await runner.run(
        chat_id=7, cmd=ParsedCommand(name="/board", args=("q",)), message=msg
    )
    assert registry.current(7) is None
    assert "internal error" in msg.replies[0].edits[-1].lower()


async def test_file_write_failure_sends_markdown_inline(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())

    class _FailingWriter:
        output_dir = tmp_path

        def write(self, result: BoardResult) -> Path:
            raise OSError("disk full")

    runner = BoardRunner(engine=engine, writer=_FailingWriter(), registry=InFlightRegistry())  # type: ignore[arg-type]
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1, cmd=ParsedCommand(name="/board", args=("q",)), message=msg
    )
    final = msg.replies[0].edits[-1]
    # File write failed → full markdown inline instead of a path pointer.
    assert "Full board →" not in final
    assert "# Board:" in final


async def test_excerpt_truncated_to_configured_chars(tmp_path: Path) -> None:
    long_response = "x" * 5000
    responses = (
        PanelistResponse(
            name="A",
            model="m",
            provider="ollama",
            response=long_response,
            latency_ms=1,
            error=None,
        ),
    )
    result = BoardResult(
        board_id="BOARD-aaaa",
        question="q",
        created_at=datetime(2026, 4, 21, tzinfo=UTC),
        panelist_responses=responses,
        synthesis=None,
    )
    engine = _StubEngine(result=result)
    engine.panelist_count = 1
    writer = BoardWriter(output_dir=tmp_path)
    runner = BoardRunner(
        engine=engine,
        writer=writer,
        registry=InFlightRegistry(),
        excerpt_chars=120,
    )
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1, cmd=ParsedCommand(name="/board", args=("q",)), message=msg
    )
    final = msg.replies[0].edits[-1]
    assert "x" * 120 in final
    assert "x" * 300 not in final
