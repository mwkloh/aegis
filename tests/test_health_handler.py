"""Tests for the /health handler factory.

Covers:
* Happy path — heartbeat file exists, recent tick → "Health: OK"
* Stale path  — heartbeat file >120s old → "Health: STALE"
* No file     — scheduler hasn't ticked yet → "has not ticked yet"
* Job roster  — N jobs shown; never-run job shows "never"
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runtime.chat.telegram.handlers import health_handler
from runtime.chat.telegram.dispatch import IncomingMessage, ParsedCommand
from runtime.scheduler.store import ScheduledJobStore

pytestmark = pytest.mark.unit


# --- helpers ------------------------------------------------------------------


def _fake_msg() -> IncomingMessage:
    """Minimal IncomingMessage stub — handler never inspects it."""

    class _Msg:
        chat_id = 1
        user_id = 1

    return _Msg()  # type: ignore[return-value]


def _fake_cmd() -> ParsedCommand:
    return ParsedCommand(name="/health", args=())


def _call(handler, hb: Path, store: ScheduledJobStore, *, now: datetime) -> str:
    """Invoke the handler with a fixed clock."""
    h = health_handler(hb, store, clock=lambda: now)
    return h(_fake_msg(), _fake_cmd())


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "aegis-index.db")


@pytest.fixture
def hb(tmp_path: Path) -> Path:
    return tmp_path / "scheduler.heartbeat"


# --- tests --------------------------------------------------------------------


def test_health_ok_recent_tick(hb: Path, store: ScheduledJobStore) -> None:
    """Heartbeat file exists, ticked 10s ago → Health: OK."""
    hb.touch()
    # Backdate mtime by 10s so age_seconds ≈ 10
    mtime = time.time() - 10
    import os
    os.utime(hb, (mtime, mtime))
    now = datetime.fromtimestamp(mtime + 10, tz=UTC)
    reply = _call(health_handler, hb, store, now=now)
    assert "Health: OK" in reply
    assert "10" in reply or "s ago" in reply


def test_health_stale(hb: Path, store: ScheduledJobStore) -> None:
    """Heartbeat file >120s old → Health: STALE."""
    hb.touch()
    mtime = time.time() - 200
    import os
    os.utime(hb, (mtime, mtime))
    now = datetime.fromtimestamp(mtime + 200, tz=UTC)
    reply = _call(health_handler, hb, store, now=now)
    assert "STALE" in reply


def test_health_no_file(hb: Path, store: ScheduledJobStore) -> None:
    """No heartbeat file yet → informative reply, no crash."""
    now = datetime.now(tz=UTC)
    reply = _call(health_handler, hb, store, now=now)
    assert "not ticked yet" in reply.lower() or "no heartbeat" in reply.lower()


def test_health_job_roster(hb: Path, store: ScheduledJobStore) -> None:
    """Job roster shows all jobs; never-run job shows 'never'."""
    created = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    store.add("0 8 * * *", "another_skill", (), 1, now=created)

    hb.touch()
    mtime = time.time() - 5
    import os
    os.utime(hb, (mtime, mtime))
    now = datetime.fromtimestamp(mtime + 5, tz=UTC)
    reply = _call(health_handler, hb, store, now=now)

    assert "Jobs (2)" in reply
    assert "morning_brief" in reply
    assert "another_skill" in reply
    assert "never" in reply


def test_health_job_with_last_run(hb: Path, store: ScheduledJobStore) -> None:
    """A job that has run shows its last_run_at, not 'never'."""
    created = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    ran_at = datetime(2026, 4, 20, 7, 0, tzinfo=UTC)
    store.record_run(job.id, at=ran_at, status="ok")

    hb.touch()
    mtime = time.time() - 5
    import os
    os.utime(hb, (mtime, mtime))
    now = datetime.fromtimestamp(mtime + 5, tz=UTC)
    reply = _call(health_handler, hb, store, now=now)

    # "never" should NOT appear since the job has a last_run_at
    lines = [ln for ln in reply.splitlines() if "morning_brief" in ln]
    assert lines, "morning_brief should appear in roster"
    assert "never" not in lines[0]
