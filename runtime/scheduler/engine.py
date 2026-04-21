"""Async tick loop for the scheduler.

This module owns the time-driven firing semantics — no skill lookup,
no Telegram, no subprocess handling. Callers inject a `JobInvoker`
(how to run a job) and optionally a `BusyCheck` (is the skill
already in flight); the engine sequences them against a `Clock`.

Design pins (from `docs/PLAN_PHASE_10_SCHEDULER.md`):

* **Never raises.** Every job invocation is wrapped in `try/except
  BaseException`. A bad skill must not take down the bot.
* **Never replays stale runs.** If `now - due > stale_threshold`,
  the tick is recorded as `"skipped"` and `scheduler.skipped_stale`
  is emitted. This prevents a post-restart push storm when the
  process was down through a scheduled window.
* **Busy-skip advances the job.** Policy A from §5: if the skill
  is already in flight, we emit `scheduler.skipped_busy` and stamp
  `last_run_at = now` so the loop picks up at the *next* cron tick
  rather than immediately retrying.
* **Deterministic tests.** The `Clock` + `Sleeper` injections let
  unit tests step time without real sleeps. `run_once()` is the
  single-pass surface; `run()` is the forever loop.
* **Structural events only.** Payloads carry job_id, skill,
  latency_ms, error_class — never args, never output.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from runtime.events.stream import EventStream, EventType
from runtime.scheduler.cron import next_run
from runtime.scheduler.job import JobStatus, ScheduledJob
from runtime.scheduler.store import ScheduledJobStore

Clock = Callable[[], datetime]
Sleeper = Callable[[float], Awaitable[None]]
BusyCheck = Callable[[ScheduledJob], bool]


@dataclass(frozen=True)
class JobOutcome:
    """Result of one job invocation. Only `status` is load-bearing."""

    status: JobStatus
    detail: str | None = None


JobInvoker = Callable[[ScheduledJob], Awaitable[JobOutcome]]


_DEFAULT_TICK_SECONDS = 60.0
_DEFAULT_STALE_SECONDS = 90.0  # 1.5x tick — anything more was a downtime gap


class SchedulerEngine:
    """Single-operator scheduler. One instance per bot process."""

    def __init__(
        self,
        *,
        store: ScheduledJobStore,
        events: EventStream,
        invoker: JobInvoker,
        clock: Clock,
        sleeper: Sleeper,
        is_busy: BusyCheck | None = None,
        tick_interval: float = _DEFAULT_TICK_SECONDS,
        stale_threshold_seconds: float = _DEFAULT_STALE_SECONDS,
        heartbeat_path: Path | None = None,
    ) -> None:
        self._store = store
        self._events = events
        self._invoker = invoker
        self._clock = clock
        self._sleeper = sleeper
        self._is_busy = is_busy if is_busy is not None else (lambda _job: False)
        self._tick_interval = tick_interval
        self._stale_threshold = stale_threshold_seconds
        self._heartbeat_path = heartbeat_path

    # --- public API ------------------------------------------------------

    async def run(self) -> None:
        """Forever loop. Returns on `CancelledError`."""
        while True:
            await self.run_once()
            await self._sleeper(self._tick_interval)

    async def run_once(self) -> None:
        """One pass over all jobs. Never raises."""
        now = self._clock()
        jobs = self._store.list_all()
        self._events.append(
            EventType.SCHEDULER_TICK,
            {"jobs_considered": len(jobs)},
        )
        if self._heartbeat_path is not None:
            self._heartbeat_path.touch()
        for job in jobs:
            if job.paused:
                continue
            await self._maybe_fire(job, now=now)

    # --- internals -------------------------------------------------------

    async def _maybe_fire(self, job: ScheduledJob, *, now: datetime) -> None:
        anchor = job.last_run_at or job.created_at
        try:
            due = next_run(job.cron_expr, after=anchor)
        except ValueError:
            # Shouldn't happen — the store validates at insert time —
            # but if the expression somehow got corrupted we log it
            # as a failure rather than crash the loop.
            self._events.append(
                EventType.SCHEDULER_JOB_FAILED,
                {
                    "job_id": job.id,
                    "skill": job.skill,
                    "latency_ms": 0,
                    "error_class": "ValueError",
                },
            )
            self._store.record_run(job.id, at=now, status="failed")
            return

        if due > now:
            return

        overdue_seconds = (now - due).total_seconds()
        if overdue_seconds > self._stale_threshold:
            self._events.append(
                EventType.SCHEDULER_SKIPPED_STALE,
                {
                    "job_id": job.id,
                    "skill": job.skill,
                    "overdue_seconds": round(overdue_seconds, 3),
                },
            )
            self._store.record_run(job.id, at=now, status="skipped")
            return

        if self._is_busy(job):
            self._events.append(
                EventType.SCHEDULER_SKIPPED_BUSY,
                {"job_id": job.id, "skill": job.skill},
            )
            self._store.record_run(job.id, at=now, status="skipped")
            return

        await self._invoke(job, now=now)

    async def _invoke(self, job: ScheduledJob, *, now: datetime) -> None:
        self._events.append(
            EventType.SCHEDULER_JOB_STARTED,
            {"job_id": job.id, "skill": job.skill},
        )
        started = time.perf_counter()
        try:
            outcome = await self._invoker(job)
        except asyncio.CancelledError:
            # Shutdown path — propagate so `run()` unwinds cleanly.
            raise
        except BaseException as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            self._events.append(
                EventType.SCHEDULER_JOB_FAILED,
                {
                    "job_id": job.id,
                    "skill": job.skill,
                    "latency_ms": latency_ms,
                    "error_class": type(exc).__name__,
                },
            )
            self._store.record_run(job.id, at=now, status="failed")
            return

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        if outcome.status == "ok":
            self._events.append(
                EventType.SCHEDULER_JOB_SUCCEEDED,
                {
                    "job_id": job.id,
                    "skill": job.skill,
                    "latency_ms": latency_ms,
                },
            )
        else:
            self._events.append(
                EventType.SCHEDULER_JOB_FAILED,
                {
                    "job_id": job.id,
                    "skill": job.skill,
                    "latency_ms": latency_ms,
                    "error_class": outcome.detail or outcome.status,
                },
            )
        self._store.record_run(job.id, at=now, status=outcome.status)


def system_clock() -> datetime:
    """Default `Clock` — wall-clock UTC. Not used in tests."""
    return datetime.now(tz=UTC)


async def system_sleeper(seconds: float) -> None:
    """Default `Sleeper` — `asyncio.sleep`. Not used in tests."""
    await asyncio.sleep(seconds)


__all__ = [
    "BusyCheck",
    "Clock",
    "JobInvoker",
    "JobOutcome",
    "SchedulerEngine",
    "Sleeper",
    "system_clock",
    "system_sleeper",
]
