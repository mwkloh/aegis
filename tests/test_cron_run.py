"""Unit tests for `/cron run <job_id>` — P2-1 immediate-fire seam.

Covers:
* Handler-level: unknown job, known job queued, no fire_fn stub reply.
* Engine-level: queue_immediate fires on run_once(), missing job is silently skipped.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.telegram.cron_handler import cron_handler
from runtime.chat.telegram.dispatch import IncomingMessage, ParsedCommand
from runtime.events.stream import EventStream
from runtime.scheduler.engine import JobOutcome, SchedulerEngine
from runtime.scheduler.job import ScheduledJob
from runtime.scheduler.store import ScheduledJobStore

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "jobs.db")


@pytest.fixture
def events(tmp_path: Path) -> EventStream:
    return EventStream(tmp_path / "sessions", session_id="test")


def _fixed_clock() -> datetime:
    return datetime(2026, 4, 21, 8, 0, tzinfo=UTC)


def _make_msg() -> IncomingMessage:
    return IncomingMessage(chat_id=1, user_id=1, text="")


def _cmd(*args: str) -> ParsedCommand:
    return ParsedCommand(name="/cron", args=args)


async def _noop_sleep(_: float) -> None:
    return None


def _make_invoker(
    calls: list[ScheduledJob],
    *,
    result: JobOutcome = JobOutcome(status="ok"),
) -> Callable[[ScheduledJob], Awaitable[JobOutcome]]:
    async def invoker(job: ScheduledJob) -> JobOutcome:
        calls.append(job)
        return result

    return invoker


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


def test_run_unknown_job(store: ScheduledJobStore) -> None:
    """Returns 'Unknown job' and does NOT call fire_now_fn."""
    fired: list[str] = []
    handler = cron_handler(
        store=store,
        clock=_fixed_clock,
        fire_now_fn=fired.append,
    )
    reply = handler(_make_msg(), _cmd("run", "UNKNOWN-xyz"))
    assert reply == "Unknown job: UNKNOWN-xyz."
    assert fired == []


def test_run_known_job_queues(store: ScheduledJobStore) -> None:
    """fire_now_fn is called with the correct job_id; reply contains 'queued'."""
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=_fixed_clock())
    fired: list[str] = []
    handler = cron_handler(
        store=store,
        clock=_fixed_clock,
        fire_now_fn=fired.append,
    )
    reply = handler(_make_msg(), _cmd("run", job.id))
    assert "queued" in reply
    assert fired == [job.id]


def test_run_no_fire_fn_returns_stub(store: ScheduledJobStore) -> None:
    """When fire_now_fn=None, returns the 'not yet implemented' stub."""
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=_fixed_clock())
    handler = cron_handler(
        store=store,
        clock=_fixed_clock,
        fire_now_fn=None,
    )
    reply = handler(_make_msg(), _cmd("run", job.id))
    assert "not yet implemented" in reply


def test_run_paused_job_returns_rejection(tmp_path: Path) -> None:
    """Paused job: handler rejects with message, fire_now_fn not called."""
    store = ScheduledJobStore(tmp_path / "jobs.db")
    # Add a job then pause it
    job = store.add(
        "0 * * * *", "morning-brief", (), created_by=1, now=datetime(2024, 1, 1, tzinfo=UTC)
    )
    store.set_paused(job.id, paused=True)

    fired: list[str] = []
    handler = cron_handler(
        store=store, clock=lambda: datetime(2024, 1, 1, tzinfo=UTC), fire_now_fn=fired.append
    )

    msg = IncomingMessage(chat_id=1, user_id=1, text=f"/cron run {job.id}")
    cmd = ParsedCommand(name="/cron", args=("run", job.id))
    reply = handler(msg, cmd)

    assert "paused" in reply.lower()
    assert fired == []  # fire_now_fn must NOT have been called


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------


async def test_queue_immediate_fires_on_run_once(
    store: ScheduledJobStore, events: EventStream
) -> None:
    """queue_immediate() causes the job to fire on the next run_once() pass."""
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=_fixed_clock())
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=_fixed_clock,
        sleeper=_noop_sleep,
    )

    engine.queue_immediate(job.id)
    # Clock is set to 08:00 — job is not yet due at 07:00 today, so
    # the regular tick would skip it. Only the immediate queue fires it.
    await engine.run_once()

    assert [j.id for j in calls] == [job.id]


async def test_queue_immediate_skips_missing_job(
    store: ScheduledJobStore, events: EventStream
) -> None:
    """queue_immediate() with a non-existent job_id does not crash or invoke."""
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=_fixed_clock,
        sleeper=_noop_sleep,
    )

    engine.queue_immediate("JOB-does-not-exist")
    await engine.run_once()  # must not raise

    assert calls == []
