"""Phase 7 §4.1 — `/status` 24h rollup contracts.

Pins:

* `compute_status` scans only the UTC-date shards overlapping the
  24h window and never the full tree.
* Distinct `session_id` values are counted (unique sessions, not
  events).
* `PATTERN_OBSERVED` events feed the `patterns` counter.
* `GOVERNANCE_DECISION` splits:
    - human verdicts (approve/reject/defer) → `decisions`
    - applier verdicts (applied_clean/applied_test_failed/
      apply_conflict/reverted) → `applies`
* Events outside [now - 24h, now] are ignored.
* Windows that cross UTC midnight read both shards.
* Malformed JSON lines are skipped — the scan never raises.
* A missing `sessions_dir` returns a zero-count snapshot.
* `status_handler` uses an injected clock so the 24h window is
  deterministic in tests; its rendered reply matches the live
  formatter output.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runtime.chat.telegram import (
    IncomingMessage,
    ParsedCommand,
    StatusSnapshot,
    compute_status,
    render_status,
    status_handler,
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


def _msg(text: str) -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text=text)


# --- compute_status -----------------------------------------------------


def test_compute_status_missing_sessions_dir(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    snap = compute_status(tmp_path / "does-not-exist", now=now)
    assert isinstance(snap, StatusSnapshot)
    assert snap.sessions == 0
    assert snap.patterns == 0
    assert snap.decisions == 0
    assert snap.applies == 0
    assert snap.window_end == now
    assert snap.window_start == now - timedelta(hours=24)


def test_compute_status_counts_distinct_sessions(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    # Same session_id across three events → 1 session
    _write_event(
        sdir,
        ts=now - timedelta(hours=1),
        session_id="s1",
        event_type="session.start",
    )
    _write_event(
        sdir,
        ts=now - timedelta(minutes=30),
        session_id="s1",
        event_type="user.message",
    )
    _write_event(
        sdir,
        ts=now - timedelta(minutes=15),
        session_id="s2",
        event_type="session.start",
    )
    snap = compute_status(sdir, now=now)
    assert snap.sessions == 2


def test_compute_status_pattern_counter(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    for i in range(3):
        _write_event(
            sdir,
            ts=now - timedelta(minutes=10 * (i + 1)),
            session_id="s1",
            event_type="pattern.observed",
            payload={"detector": f"d{i}"},
        )
    snap = compute_status(sdir, now=now)
    assert snap.patterns == 3


def test_compute_status_decision_split(tmp_path: Path) -> None:
    """Human verdicts feed `decisions`; applier verdicts feed `applies`."""
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    verdicts = [
        ("approve", "human"),
        ("reject", "human"),
        ("defer", "human"),
        ("applied_clean", "applier"),
        ("applied_test_failed", "applier"),
        ("apply_conflict", "applier"),
        ("reverted", "applier"),
    ]
    for i, (verdict, _bucket) in enumerate(verdicts):
        _write_event(
            sdir,
            ts=now - timedelta(minutes=i + 1),
            session_id="s1",
            event_type="governance.decision",
            payload={"verdict": verdict, "imp_id": f"IMP-{i:08x}"},
        )
    snap = compute_status(sdir, now=now)
    assert snap.decisions == 3  # approve/reject/defer
    assert snap.applies == 4


def test_compute_status_drops_events_outside_window(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    # Old event — 48h ago, outside window
    _write_event(
        sdir,
        ts=now - timedelta(hours=48),
        session_id="old",
        event_type="pattern.observed",
    )
    # Fresh event
    _write_event(
        sdir,
        ts=now - timedelta(hours=1),
        session_id="fresh",
        event_type="pattern.observed",
    )
    snap = compute_status(sdir, now=now)
    assert snap.patterns == 1
    assert snap.sessions == 1


def test_compute_status_reads_both_shards_across_midnight(tmp_path: Path) -> None:
    """24h window crossing UTC midnight reads yesterday + today shards."""
    now = datetime(2026, 4, 19, 3, tzinfo=UTC)  # 03:00 UTC
    sdir = tmp_path / "sessions"
    # Yesterday shard (within window: now - 24h = 2026-04-18 03:00)
    _write_event(
        sdir,
        ts=datetime(2026, 4, 18, 20, tzinfo=UTC),
        session_id="yday",
        event_type="pattern.observed",
    )
    # Today shard
    _write_event(
        sdir,
        ts=datetime(2026, 4, 19, 1, tzinfo=UTC),
        session_id="today",
        event_type="pattern.observed",
    )
    snap = compute_status(sdir, now=now)
    assert snap.patterns == 2
    assert snap.sessions == 2


def test_compute_status_skips_malformed_jsonl(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    _write_event(
        sdir,
        ts=now - timedelta(hours=1),
        session_id="s1",
        event_type="pattern.observed",
    )
    shard_file = sdir / now.strftime("%Y-%m-%d") / "s1.jsonl"
    # Append a torn line + a line with non-dict JSON + a blank line.
    with shard_file.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("[1, 2, 3]\n")
        fh.write("\n")
    snap = compute_status(sdir, now=now)
    # Only the single valid record counts; the rest are skipped silently.
    assert snap.patterns == 1


def test_compute_status_ignores_unknown_verdicts(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    _write_event(
        sdir,
        ts=now - timedelta(hours=1),
        session_id="s1",
        event_type="governance.decision",
        payload={"verdict": "something_else"},
    )
    snap = compute_status(sdir, now=now)
    assert snap.decisions == 0
    assert snap.applies == 0


def test_compute_status_frozen(tmp_path: Path) -> None:
    now = datetime(2026, 4, 19, 12, tzinfo=UTC)
    snap = compute_status(tmp_path, now=now)
    with pytest.raises(Exception, match="frozen"):
        snap.sessions = 99  # type: ignore[misc]


# --- status_handler -----------------------------------------------------


def test_status_handler_uses_injected_clock(tmp_path: Path) -> None:
    frozen = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    _write_event(
        sdir,
        ts=frozen - timedelta(hours=2),
        session_id="s1",
        event_type="pattern.observed",
    )
    handler = status_handler(sdir, clock=lambda: frozen)
    out = handler(_msg("/status"), ParsedCommand(name="/status", args=()))
    # Deterministic window labels
    assert "2026-04-18 12:00Z" in out  # window_start
    assert "2026-04-19 12:00Z" in out  # window_end
    assert "patterns:  1" in out
    assert "sessions:  1" in out


def test_status_handler_matches_formatter(tmp_path: Path) -> None:
    frozen = datetime(2026, 4, 19, 12, tzinfo=UTC)
    sdir = tmp_path / "sessions"
    handler = status_handler(sdir, clock=lambda: frozen)
    out = handler(_msg("/status"), ParsedCommand(name="/status", args=()))
    expected = render_status(compute_status(sdir, now=frozen))
    assert out == expected
