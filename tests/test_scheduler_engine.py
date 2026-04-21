"""Phase 10 Track A3 — `SchedulerEngine` tick loop.

Pins:

* Injectable `Clock` + `Sleeper` so tests never wait on wall-clock.
* `run_once()` is the unit-test surface — one pass over all jobs,
  then returns. `run()` wraps it in a forever loop that awaits the
  sleeper and exits on `asyncio.CancelledError`.
* Never raises. A job that blows up emits `scheduler.job_failed`
  and the loop carries on.
* Never replays stale runs. If the process was down, ticks whose
  due time is more than `stale_threshold_seconds` in the past get
  `scheduler.skipped_stale` and `record_run(status="skipped")` so
  the next computation advances past them.
* Busy-skip advances the job past this tick too (§5 policy A).
  Work-preserving queues are out of scope.
* Every pass emits exactly one `scheduler.tick` event, carrying
  only the count of jobs considered — no skill output, no args.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.events.stream import EventStream, EventType
from runtime.scheduler.engine import (
    JobOutcome,
    SchedulerEngine,
)
from runtime.scheduler.job import ScheduledJob
from runtime.scheduler.store import ScheduledJobStore

pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "aegis-index.db")


@pytest.fixture
def events(tmp_path: Path) -> EventStream:
    return EventStream(tmp_path / "sessions", session_id="test")


def _read_events(stream: EventStream) -> list[dict[str, object]]:
    if not stream.path.exists():
        return []
    return [json.loads(line) for line in stream.path.read_text().splitlines()]


def _types(stream: EventStream) -> list[str]:
    return [str(e["type"]) for e in _read_events(stream)]


# --- helpers --------------------------------------------------------------


class _FixedClock:
    """Mutable clock — lets tests advance time between run_once() calls."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


_OK = JobOutcome(status="ok")


def _make_invoker(
    calls: list[ScheduledJob],
    *,
    result: JobOutcome | Exception = _OK,
) -> Callable[[ScheduledJob], Awaitable[JobOutcome]]:
    async def invoker(job: ScheduledJob) -> JobOutcome:
        calls.append(job)
        if isinstance(result, Exception):
            raise result
        return result

    return invoker


async def _noop_sleep(_: float) -> None:
    return None


# --- run_once: firing semantics ------------------------------------------


async def test_run_once_fires_due_job(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    assert [j.id for j in calls] == [job.id]
    assert "scheduler.tick" in _types(events)
    assert "scheduler.job_started" in _types(events)
    assert "scheduler.job_succeeded" in _types(events)
    updated = store.get(job.id)
    assert updated is not None
    assert updated.last_status == "ok"
    assert updated.last_run_at == clock.now


async def test_run_once_skips_job_not_yet_due(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 0, tzinfo=UTC)
    store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 6, 30, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    assert calls == []
    types = _types(events)
    assert "scheduler.tick" in types
    assert "scheduler.job_started" not in types


async def test_run_once_skips_paused_jobs(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    store.set_paused(job.id, paused=True)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    assert calls == []


async def test_run_once_handles_multiple_jobs(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    j1 = store.add("0 7 * * *", "a", (), 1, now=created)
    j2 = store.add("0 7 * * *", "b", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    assert {j.id for j in calls} == {j1.id, j2.id}


# --- never raises ---------------------------------------------------------


async def test_job_failure_emits_failed_and_continues(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    j_bad = store.add("0 7 * * *", "bad", (), 1, now=created)
    j_good = store.add("0 7 * * *", "good", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))

    async def invoker(job: ScheduledJob) -> JobOutcome:
        if job.skill == "bad":
            raise RuntimeError("boom")
        return JobOutcome(status="ok")

    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=invoker,
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    types = _types(events)
    assert "scheduler.job_failed" in types
    assert "scheduler.job_succeeded" in types
    # bad job recorded as failed, good as ok — bad didn't poison the pass
    assert store.get(j_bad.id).last_status == "failed"  # type: ignore[union-attr]
    assert store.get(j_good.id).last_status == "ok"  # type: ignore[union-attr]
    # failure payload carries the error class but not the message
    failed = next(e for e in _read_events(events) if e["type"] == "scheduler.job_failed")
    payload = failed["payload"]
    assert isinstance(payload, dict)
    assert payload["error_class"] == "RuntimeError"
    assert "boom" not in json.dumps(payload)


# --- busy-skip ------------------------------------------------------------


async def test_busy_skip_does_not_invoke_and_advances(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
        is_busy=lambda j: j.skill == "morning_brief",
    )

    await engine.run_once()

    assert calls == []
    assert "scheduler.skipped_busy" in _types(events)
    updated = store.get(job.id)
    assert updated is not None
    assert updated.last_status == "skipped"
    assert updated.last_run_at == clock.now


# --- stale-skip -----------------------------------------------------------


async def test_stale_tick_is_skipped_not_replayed(
    store: ScheduledJobStore, events: EventStream
) -> None:
    # Job created at 06:59 yesterday, bot process was down, woke up
    # now at 07:05 today — the daily 07:00 tick from yesterday is
    # stale and must not retroactively fire.
    created = datetime(2026, 4, 19, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 5, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
        stale_threshold_seconds=90.0,
    )

    await engine.run_once()

    assert calls == []
    assert "scheduler.skipped_stale" in _types(events)
    updated = store.get(job.id)
    assert updated is not None
    assert updated.last_status == "skipped"


async def test_within_grace_still_fires(
    store: ScheduledJobStore, events: EventStream
) -> None:
    # Overdue by 30s — within the default stale threshold, so fires.
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, 30, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
        stale_threshold_seconds=90.0,
    )

    await engine.run_once()

    assert [j.id for j in calls] == [job.id]


# --- successive ticks advance correctly -----------------------------------


async def test_two_ticks_do_not_double_fire(
    store: ScheduledJobStore, events: EventStream
) -> None:
    # After a successful fire, the next run should be the following day
    # — a second tick 30s later must not re-invoke.
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()
    clock.now = datetime(2026, 4, 20, 7, 0, 30, tzinfo=UTC)
    await engine.run_once()

    assert len(calls) == 1


# --- tick event always emitted --------------------------------------------


async def test_tick_event_emitted_even_with_no_jobs(
    store: ScheduledJobStore, events: EventStream
) -> None:
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker([]),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    evs = _read_events(events)
    ticks = [e for e in evs if e["type"] == "scheduler.tick"]
    assert len(ticks) == 1
    payload = ticks[0]["payload"]
    assert isinstance(payload, dict)
    assert payload["jobs_considered"] == 0


# --- run() forever loop ---------------------------------------------------


async def test_run_loops_until_cancelled(
    store: ScheduledJobStore, events: EventStream
) -> None:
    created = datetime(2026, 4, 20, 6, 59, tzinfo=UTC)
    store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    sleep_calls: list[float] = []
    stop = asyncio.Event()

    async def sleeper(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 2:
            stop.set()
            raise asyncio.CancelledError()

    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=sleeper,
        tick_interval=60.0,
    )

    with pytest.raises(asyncio.CancelledError):
        await engine.run()

    assert stop.is_set()
    assert sleep_calls == [60.0, 60.0]


# --- stale detection uses last_run_at, not created_at ---------------------


async def test_stale_check_uses_last_run_when_present(
    store: ScheduledJobStore, events: EventStream
) -> None:
    # Job fired yesterday at 07:00, today is 07:00 — fires normally.
    # The stale-threshold window must be measured from today's 07:00
    # tick, not from creation.
    created = datetime(2026, 4, 18, 6, 59, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), 1, now=created)
    store.record_run(
        job.id,
        at=datetime(2026, 4, 19, 7, 0, tzinfo=UTC),
        status="ok",
    )
    clock = _FixedClock(datetime(2026, 4, 20, 7, 0, tzinfo=UTC))
    calls: list[ScheduledJob] = []
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=_make_invoker(calls),
        clock=clock,
        sleeper=_noop_sleep,
    )

    await engine.run_once()

    assert [j.id for j in calls] == [job.id]


# --- EventType members exist ---------------------------------------------


def test_new_event_types_are_registered() -> None:
    # Lock in the string values so `/logs` filters and downstream
    # consumers can rely on them.
    assert EventType.SCHEDULER_TICK.value == "scheduler.tick"
    assert EventType.SCHEDULER_JOB_STARTED.value == "scheduler.job_started"
    assert EventType.SCHEDULER_JOB_SUCCEEDED.value == "scheduler.job_succeeded"
    assert EventType.SCHEDULER_JOB_FAILED.value == "scheduler.job_failed"
    assert EventType.SCHEDULER_SKIPPED_BUSY.value == "scheduler.skipped_busy"
    assert EventType.SCHEDULER_SKIPPED_STALE.value == "scheduler.skipped_stale"
