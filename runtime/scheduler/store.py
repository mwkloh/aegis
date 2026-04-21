"""sqlite-backed `ScheduledJobStore`.

The schema lives in `memory/store_sqlite.py` v3. This module wires
CRUD on top so the rest of the scheduler can stay pure.

Design pins:

* **Short-lived connections.** Every call opens, commits, closes.
  At single-operator scale (tens of jobs, ticks every minute at
  most) the overhead is invisible and we dodge the "stale connection
  in long-running process" class of bugs.
* **ID minting with retry.** `new_job_id()` is random hex; the
  space (65k) is vastly bigger than the expected job count, but we
  still check for collisions and retry a bounded number of times,
  then raise. Silent re-mint would mask a broken RNG.
* **Timestamps as ISO-8601 tz-aware.** Round-tripped via
  `datetime.isoformat()` / `datetime.fromisoformat()`. Naive input
  is rejected at the boundary, matching the rest of AEGIS.
* **Args as JSON array.** The `ScheduledJob` model already enforces
  tuple-of-strings; storing as JSON preserves whitespace and unicode
  that a shell-style split would mangle.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import cast, get_args

from memory.store_sqlite import ensure as ensure_db
from runtime.scheduler.cron import validate as validate_cron
from runtime.scheduler.job import JobStatus, ScheduledJob, new_job_id

_ID_COLLISION_RETRIES = 8
_VALID_STATUSES: frozenset[str] = frozenset(get_args(JobStatus))


class ScheduledJobStore:
    """CRUD over `scheduled_jobs`. One instance per DB path."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = ensure_db(db_path)

    # --- mutators --------------------------------------------------------

    def add(
        self,
        cron_expr: str,
        skill: str,
        args: tuple[str, ...],
        created_by: int,
        *,
        now: datetime,
    ) -> ScheduledJob:
        """Insert a new job. Raises on invalid cron / naive timestamp."""
        if now.tzinfo is None:
            raise ValueError("`now` must be tz-aware")
        validate_cron(cron_expr)
        args_json = json.dumps(list(args), ensure_ascii=False)
        with sqlite3.connect(self._db_path) as conn:
            for _ in range(_ID_COLLISION_RETRIES):
                job_id = new_job_id()
                try:
                    conn.execute(
                        "INSERT INTO scheduled_jobs "
                        "(id, cron_expr, skill, args_json, created_at, "
                        "created_by, paused) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (
                            job_id,
                            cron_expr,
                            skill,
                            args_json,
                            now.isoformat(),
                            created_by,
                        ),
                    )
                    conn.commit()
                    break
                except sqlite3.IntegrityError:
                    continue
            else:
                raise RuntimeError(
                    f"could not mint unique job id after "
                    f"{_ID_COLLISION_RETRIES} attempts — collision rate "
                    f"suggests a broken RNG"
                )
        job = ScheduledJob(
            id=job_id,
            cron_expr=cron_expr,
            skill=skill,
            args=args,
            created_at=now,
            created_by=created_by,
        )
        return job

    def upsert_system_job(
        self,
        *,
        job_id: str,
        cron_expr: str,
        skill: str,
        args: tuple[str, ...],
        now: datetime,
    ) -> ScheduledJob:
        """Insert a system-owned job iff absent; return the live record.

        System jobs use caller-chosen stable ids (e.g. ``SYS-morning-brief``)
        so boot-time seeding is idempotent. If a row with this id already
        exists it's left untouched — operator-set ``paused`` flags and
        ``last_run_at`` bookkeeping survive restarts. The ``created_by``
        column is stamped 0, which operator-minted rows (positive Telegram
        user ids) never collide with.
        """
        if now.tzinfo is None:
            raise ValueError("`now` must be tz-aware")
        validate_cron(cron_expr)
        args_json = json.dumps(list(args), ensure_ascii=False)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO scheduled_jobs "
                "(id, cron_expr, skill, args_json, created_at, "
                "created_by, paused) "
                "VALUES (?, ?, ?, ?, ?, 0, 0)",
                (job_id, cron_expr, skill, args_json, now.isoformat()),
            )
            conn.commit()
        existing = self.get(job_id)
        if existing is None:
            raise RuntimeError(
                f"upsert failed — no row after INSERT OR IGNORE for {job_id!r}"
            )
        return existing

    def remove(self, job_id: str) -> bool:
        """Delete a job. Returns True if a row was removed."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "DELETE FROM scheduled_jobs WHERE id = ?", (job_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def set_paused(self, job_id: str, *, paused: bool) -> ScheduledJob | None:
        """Flip the pause flag. Returns the updated record, or None if missing."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE scheduled_jobs SET paused = ? WHERE id = ?",
                (1 if paused else 0, job_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(job_id)

    def record_run(
        self, job_id: str, *, at: datetime, status: JobStatus
    ) -> ScheduledJob | None:
        """Stamp last run. Called by the engine after a tick completes."""
        if at.tzinfo is None:
            raise ValueError("`at` must be tz-aware")
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)}, got {status!r}"
            )
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE scheduled_jobs "
                "SET last_run_at = ?, last_status = ? WHERE id = ?",
                (at.isoformat(), status, job_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(job_id)

    # --- readers ---------------------------------------------------------

    def get(self, job_id: str) -> ScheduledJob | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT id, cron_expr, skill, args_json, created_at, "
                "created_by, last_run_at, last_status, paused "
                "FROM scheduled_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_job(row)

    def list_all(self) -> list[ScheduledJob]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, cron_expr, skill, args_json, created_at, "
                "created_by, last_run_at, last_status, paused "
                "FROM scheduled_jobs "
                "ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [_row_to_job(r) for r in rows]


def _row_to_job(row: tuple[object, ...]) -> ScheduledJob:
    (
        job_id,
        cron_expr,
        skill,
        args_json,
        created_at,
        created_by,
        last_run_at,
        last_status,
        paused,
    ) = row
    args = tuple(json.loads(cast(str, args_json)))
    return ScheduledJob(
        id=cast(str, job_id),
        cron_expr=cast(str, cron_expr),
        skill=cast(str, skill),
        args=args,
        created_at=datetime.fromisoformat(cast(str, created_at)),
        created_by=cast(int, created_by),
        last_run_at=(
            datetime.fromisoformat(cast(str, last_run_at))
            if last_run_at is not None
            else None
        ),
        last_status=cast("JobStatus | None", last_status),
        paused=bool(paused),
    )


__all__ = ["ScheduledJobStore"]
