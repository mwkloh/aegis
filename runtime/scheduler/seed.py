"""Phase 10 Track D — seed built-in recurring jobs at boot.

Four system jobs are seeded every time `build_scheduler` runs. They
use stable `SYS-*` ids so `ScheduledJobStore.upsert_system_job` can
be idempotent — an already-existing row keeps its operator-set
`paused` flag and `last_run_at` bookkeeping across restarts.

Design pins:

* **Idempotent.** Re-seeding is a no-op if the row already exists.
  Operators can `/cron pause SYS-morning-brief` once; the pause
  survives every subsequent boot.
* **Never raises.** A seeding failure (DB locked, disk full) is
  logged and skipped — the bot still comes up, the scheduler still
  ticks, and the next boot retries. One bad system job must not
  block the other three.
* **Agnostic about catalog state.** If a skill isn't in the
  catalog yet, the job still gets seeded; `JobRunner` will classify
  the fire as `unknown_skill` (silent-to-operator) until the catalog
  entry lands. This lets D1/D2 ship before D3 without rollbacks.
* **UTC cron expressions.** Single-operator deployment today picks
  the UTC offset it wants explicitly; no per-job `timezone` column.
  Defaults below target NZ (UTC+12/+13): morning_brief at 20:00 UTC
  (08:00 NZST / 09:00 NZDT); maintenance triad at 15:00/15:15/15:30
  UTC (03:00–03:30 NZST). The 15-minute spread keeps busy-skip
  interaction deterministic — if one overruns, the next skips and
  retries tomorrow. Adjust via ``/cron`` after first boot if needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from runtime.scheduler.store import ScheduledJobStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SystemJobSpec:
    """Declarative definition of a built-in recurring job."""

    id: str
    cron_expr: str
    skill: str
    args: tuple[str, ...] = ()


SYSTEM_JOBS: tuple[SystemJobSpec, ...] = (
    SystemJobSpec(
        id="SYS-morning-brief",
        cron_expr="0 20 * * *",  # 08:00 NZST / 09:00 NZDT
        skill="morning_brief",
    ),
    SystemJobSpec(
        id="SYS-tier2-compress",
        cron_expr="0 15 * * *",  # 03:00 NZST
        skill="tier2_compress",
    ),
    SystemJobSpec(
        id="SYS-reflection-sweep",
        cron_expr="15 15 * * *",  # 03:15 NZST
        skill="reflection_sweep",
    ),
    SystemJobSpec(
        id="SYS-vault-reindex",
        cron_expr="30 15 * * *",  # 03:30 NZST
        skill="vault_reindex",
    ),
)


def seed_system_jobs(
    store: ScheduledJobStore,
    *,
    now: datetime,
    specs: tuple[SystemJobSpec, ...] = SYSTEM_JOBS,
) -> int:
    """Idempotently insert system jobs. Returns count actually inserted.

    Each spec is seeded independently; a failure on one is logged
    and does not block the others. The caller is expected to invoke
    this before `SchedulerEngine.run()` — a partially-seeded store
    is still a valid store.
    """
    inserted = 0
    for spec in specs:
        try:
            existed_before = store.get(spec.id) is not None
            store.upsert_system_job(
                job_id=spec.id,
                cron_expr=spec.cron_expr,
                skill=spec.skill,
                args=spec.args,
                now=now,
            )
            if not existed_before:
                inserted += 1
        except Exception:
            # Don't let one bad seed block the rest. Boot must not hang.
            logger.exception(
                "scheduler.seed_failed",
                extra={"job_id": spec.id, "skill": spec.skill},
            )
    return inserted


__all__ = [
    "SYSTEM_JOBS",
    "SystemJobSpec",
    "seed_system_jobs",
]
