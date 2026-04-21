"""Reflection event reader — multi-day, malformed-tolerant, date-filtered."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from runtime.reflection.event_reader import Event, read_window

pytestmark = pytest.mark.unit


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(session_id: str, type_: str, **payload: object) -> dict[str, object]:
    return {
        "ts": datetime.now(tz=UTC).isoformat(),
        "session_id": session_id,
        "type": type_,
        "payload": payload,
    }


def test_returns_empty_when_sessions_dir_missing(tmp_path: Path) -> None:
    events, stats = read_window(tmp_path / "missing")
    assert events == []
    assert stats.sessions == 0
    assert stats.events == 0
    assert stats.skipped == 0


def test_reads_one_day_of_events(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    _write(
        tmp_path / today / "abc.jsonl",
        [_record("abc", "user.message", text="hi"),
         _record("abc", "assistant.reply", text="hello")],
    )
    events, stats = read_window(tmp_path)
    assert stats.events == 2
    assert stats.sessions == 1
    assert stats.skipped == 0
    assert all(isinstance(e, Event) for e in events)


def test_filters_out_days_before_since(tmp_path: Path) -> None:
    _write(tmp_path / "2025-01-01" / "old.jsonl", [_record("old", "user.message")])
    _write(tmp_path / "2099-12-31" / "new.jsonl", [_record("new", "user.message")])
    events, stats = read_window(tmp_path, since=date(2099, 1, 1))
    assert stats.events == 1
    assert events[0].session_id == "new"


def test_skips_malformed_lines_and_records_count(tmp_path: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    path = tmp_path / today / "x.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = json.dumps(_record("x", "user.message"), sort_keys=True)
    path.write_text(
        valid + "\n"
        + "{not json\n"          # decode error
        + "[1,2,3]\n"            # not a dict
        + json.dumps({"ts": "now", "bogus": True}) + "\n"  # schema violation
        + "\n"                                                # blank line ignored
        + valid + "\n",
        encoding="utf-8",
    )
    _events, stats = read_window(tmp_path)
    assert stats.events == 2
    assert stats.skipped == 3


def test_ignores_non_date_directories(tmp_path: Path) -> None:
    (tmp_path / "not-a-date").mkdir()
    (tmp_path / "not-a-date" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    events, stats = read_window(tmp_path)
    assert events == []
    assert stats.events == 0
    assert stats.skipped == 0
