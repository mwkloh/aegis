"""Phase 10 — Autonomous scheduler.

Runs as an asyncio task inside the Telegram bot process; no
external cron, no separate daemon. See
`docs/PLAN_PHASE_10_SCHEDULER.md` for the load-bearing design pins.

Track A1 ships the pure pieces (no IO, no asyncio):

* `cron.py` — cron-expression validation + next-run math.
* `job.py` — `ScheduledJob` record.

Later tracks layer on sqlite persistence (A2), the tick loop (A3),
skill invocation + delivery (B), and `/cron` slashes (C).
"""
from __future__ import annotations

from .cron import describe, next_run, validate
from .engine import (
    BusyCheck,
    Clock,
    JobInvoker,
    JobOutcome,
    SchedulerEngine,
    Sleeper,
    system_clock,
    system_sleeper,
)
from .job import ScheduledJob, new_job_id
from .runner import ArgvResolver, Deliverer, JobRunner
from .seed import SYSTEM_JOBS, SystemJobSpec, seed_system_jobs
from .store import ScheduledJobStore

__all__ = [
    "SYSTEM_JOBS",
    "ArgvResolver",
    "BusyCheck",
    "Clock",
    "Deliverer",
    "JobInvoker",
    "JobOutcome",
    "JobRunner",
    "ScheduledJob",
    "ScheduledJobStore",
    "SchedulerEngine",
    "Sleeper",
    "SystemJobSpec",
    "describe",
    "new_job_id",
    "next_run",
    "seed_system_jobs",
    "system_clock",
    "system_sleeper",
    "validate",
]
