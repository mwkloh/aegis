"""`ScheduledJob` record.

Immutable + schema-validated at construction. Mutation (pause,
last-run update) happens through the store layer, which builds a
new record via `model_copy(update=...)`.

Job IDs are short (`JOB-` + 4 hex chars) because they show up in
`/cron list` and the operator needs to type them into `/cron rm`
from a phone. Collision handling is the store's job — at single-
operator scale (tens of jobs) collisions are a theoretical concern
only.
"""
from __future__ import annotations

import secrets
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime.scheduler.cron import validate as validate_cron

JobStatus = Literal["ok", "failed", "skipped"]


class ScheduledJob(BaseModel):
    """One row in the `scheduled_jobs` table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    cron_expr: str = Field(min_length=1)
    skill: str = Field(min_length=1)
    args: tuple[str, ...] = ()
    created_at: datetime
    created_by: int
    last_run_at: datetime | None = None
    last_status: JobStatus | None = None
    paused: bool = False

    @field_validator("cron_expr")
    @classmethod
    def _validate_cron_expr(cls, v: str) -> str:
        validate_cron(v)
        return v

    @field_validator("created_at", "last_run_at")
    @classmethod
    def _require_tz(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("timestamp must be tz-aware")
        return v


def new_job_id() -> str:
    """`"JOB-" + 4 lowercase hex chars`. Short for phone-typing."""
    return f"JOB-{secrets.token_hex(2)}"


__all__ = [
    "JobStatus",
    "ScheduledJob",
    "new_job_id",
]
