"""`/logs [N]` — tail of today's structural event shard.

Pins:

* `tail_events` scans today's shard directory only; yesterday is
  ignored (the rollup belongs to `/status`).
* Results are chronological (oldest → newest).
* `N` is clamped to `[1, MAX_LOG_LINES]`.
* A missing shard directory returns `([], 0)` — no raise.
* Malformed JSONL lines are skipped silently.
* Multiple session files in today's shard are concatenated
  alphabetically (the sort is stable so tests can assert order).
* `render_logs` shows a "N of TOTAL" header and one line per event
  with clipped per-line payload summary.
* `logs_handler` accepts an integer argument, defaults to
  `DEFAULT_LOG_LINES`, and returns a usage reply on non-integer args.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.telegram import (
    DEFAULT_LOG_LINES,
    MAX_LOG_LINE_CHARS,
    MAX_LOG_LINES,
    IncomingMessage,
    ParsedCommand,
    build_read_only_handlers,
    logs_handler,
    render_logs,
    tail_events,
)

pytestmark = pytest.mark.unit


def _write_event(
    sessions_dir: Path,
    *,
    ts: datetime,
    session_id: str,
    event_type: str,
    payload: dict[str, object] | None = None,
) -> None:
    shard = sessions_dir / ts.strftime("%Y-%m-%d")
    shard.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": ts.isoformat(),
        "session_id": session_id,
        "type": event_type,
        "payload": payload or {},
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with (shard / f"{session_id}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _msg(text: str = "/logs") -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text=text)


def _cmd(*args: str) -> ParsedCommand:
    return ParsedCommand(name="/logs", args=tuple(args))


# --- tail_events --------------------------------------------------------


def test_tail_events_missing_sessions_dir(tmp_path: Path) -> None:
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    entries, total = tail_events(tmp_path / "nope", n=10, now=now)
    assert entries == []
    assert total == 0


def test_tail_events_empty_shard(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    (sessions / "2026-04-20").mkdir(parents=True)
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    entries, total = tail_events(sessions, n=10, now=now)
    assert entries == []
    assert total == 0


def test_tail_events_returns_all_when_n_exceeds_count(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    _write_event(sessions, ts=ts, session_id="a", event_type="x")
    _write_event(sessions, ts=ts, session_id="a", event_type="y")
    entries, total = tail_events(sessions, n=10, now=ts)
    assert total == 2
    assert [e["type"] for e in entries] == ["x", "y"]


def test_tail_events_returns_last_n_in_chronological_order(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    for i in range(5):
        _write_event(sessions, ts=ts, session_id="a", event_type=f"e{i}")
    entries, total = tail_events(sessions, n=3, now=ts)
    assert total == 5
    assert [e["type"] for e in entries] == ["e2", "e3", "e4"]


def test_tail_events_concatenates_multiple_files_alphabetically(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    _write_event(sessions, ts=ts, session_id="alpha", event_type="a1")
    _write_event(sessions, ts=ts, session_id="bravo", event_type="b1")
    entries, total = tail_events(sessions, n=10, now=ts)
    assert total == 2
    assert [e["type"] for e in entries] == ["a1", "b1"]


def test_tail_events_clamps_n_to_max(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    for i in range(MAX_LOG_LINES + 5):
        _write_event(sessions, ts=ts, session_id="a", event_type=f"e{i}")
    entries, total = tail_events(sessions, n=MAX_LOG_LINES + 50, now=ts)
    assert total == MAX_LOG_LINES + 5
    assert len(entries) == MAX_LOG_LINES


def test_tail_events_clamps_n_to_at_least_one(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    _write_event(sessions, ts=ts, session_id="a", event_type="x")
    entries, _ = tail_events(sessions, n=0, now=ts)
    assert len(entries) == 1


def test_tail_events_skips_malformed_lines(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    _write_event(sessions, ts=ts, session_id="a", event_type="ok")
    shard_path = sessions / "2026-04-20" / "a.jsonl"
    with shard_path.open("a", encoding="utf-8") as fh:
        fh.write("{not-json\n")
        fh.write("\n")
    entries, total = tail_events(sessions, n=10, now=ts)
    assert total == 1
    assert entries[0]["type"] == "ok"


def test_tail_events_ignores_yesterdays_shard(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    yesterday_ts = datetime(2026, 4, 19, 12, tzinfo=UTC)
    today_ts = datetime(2026, 4, 20, 12, tzinfo=UTC)
    _write_event(sessions, ts=yesterday_ts, session_id="a", event_type="old")
    _write_event(sessions, ts=today_ts, session_id="b", event_type="new")
    entries, total = tail_events(sessions, n=10, now=today_ts)
    assert total == 1
    assert entries[0]["type"] == "new"


# --- render_logs --------------------------------------------------------


def test_render_logs_empty_state() -> None:
    assert render_logs([], total=0) == "No events recorded yet today."


def test_render_logs_header_shows_count_and_total() -> None:
    entries = [
        {
            "ts": "2026-04-20T18:34:02.000000+00:00",
            "type": "chat.turn.reply",
            "payload": {"chat_id": "42", "reply_bytes": 93},
        }
    ]
    out = render_logs(entries, total=7)
    assert out.startswith("Last 1 of 7 event(s) today (UTC):")


def test_render_logs_includes_time_type_and_fields() -> None:
    entries = [
        {
            "ts": "2026-04-20T18:34:02.000000+00:00",
            "type": "chat.turn.reply",
            "payload": {
                "chat_id": "42",
                "reply_bytes": 93,
                "errored": False,
            },
        }
    ]
    out = render_logs(entries, total=1)
    body = out.splitlines()[1]
    assert body.startswith("18:34:02 chat.turn.reply")
    assert "chat_id=42" in body
    assert "reply_bytes=93" in body
    assert "errored=false" in body


def test_render_logs_skips_complex_payload_values() -> None:
    entries = [
        {
            "ts": "2026-04-20T18:34:02+00:00",
            "type": "pattern.observed",
            "payload": {
                "tags": ["x", "y"],
                "nested": {"inner": 1},
                "ok": True,
            },
        }
    ]
    out = render_logs(entries, total=1)
    body = out.splitlines()[1]
    assert "tags=" not in body
    assert "nested=" not in body
    assert "ok=true" in body


def test_render_logs_clips_long_lines() -> None:
    huge = "x" * 400
    entries = [
        {
            "ts": "2026-04-20T18:34:02+00:00",
            "type": "noisy",
            "payload": {"a": huge, "b": huge, "c": huge},
        }
    ]
    out = render_logs(entries, total=1)
    body = out.splitlines()[1]
    assert len(body) <= MAX_LOG_LINE_CHARS


def test_render_logs_handles_missing_fields_gracefully() -> None:
    entries = [{"payload": "not-a-dict"}]
    out = render_logs(entries, total=1)
    body = out.splitlines()[1]
    assert body.startswith("? ?")


# --- logs_handler -------------------------------------------------------


def test_logs_handler_default_n(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    for i in range(DEFAULT_LOG_LINES + 5):
        _write_event(sessions, ts=now, session_id="a", event_type=f"e{i}")
    handler = logs_handler(sessions, clock=lambda: now)
    reply = handler(_msg(), _cmd())
    assert f"Last {DEFAULT_LOG_LINES} of {DEFAULT_LOG_LINES + 5}" in reply


def test_logs_handler_respects_explicit_n(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    for i in range(10):
        _write_event(sessions, ts=now, session_id="a", event_type=f"e{i}")
    handler = logs_handler(sessions, clock=lambda: now)
    reply = handler(_msg(), _cmd("3"))
    assert "Last 3 of 10" in reply


def test_logs_handler_rejects_non_integer(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    handler = logs_handler(sessions, clock=lambda: now)
    reply = handler(_msg(), _cmd("abc"))
    assert "expects an integer line count" in reply
    assert "Usage: /logs" in reply


def test_logs_handler_empty_shard_reply(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    now = datetime(2026, 4, 20, 12, tzinfo=UTC)
    handler = logs_handler(sessions, clock=lambda: now)
    reply = handler(_msg(), _cmd())
    assert reply == "No events recorded yet today."


def test_logs_handler_in_default_registration(tmp_path: Path) -> None:
    handlers = build_read_only_handlers(tmp_path)
    assert "/logs" in handlers
