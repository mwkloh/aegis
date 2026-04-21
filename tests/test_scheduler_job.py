"""Phase 10 Track A1 — `ScheduledJob` record.

Pins:

* Immutable Pydantic model (`extra="forbid", frozen=True`) — once
  a job is registered, mutations go through store operations that
  return a new record.
* `new_job_id()` → `"JOB-" + 4 lowercase hex chars`. Short enough
  for the operator to type in `/cron rm`; collision handling
  belongs in the store layer, not here.
* `cron_expr` validation happens at construction time — a bad
  expression raises `ValueError` immediately, not at first tick.
* Timestamps are tz-aware UTC at every boundary (same rule as the
  cron module).
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from runtime.scheduler.job import ScheduledJob, new_job_id

pytestmark = pytest.mark.unit


def _job(**overrides: object) -> ScheduledJob:
    base: dict[str, object] = {
        "id": "JOB-abcd",
        "cron_expr": "0 7 * * *",
        "skill": "morning_brief",
        "args": (),
        "created_at": datetime(2026, 4, 20, 10, 0, tzinfo=UTC),
        "created_by": 7846523803,
    }
    base.update(overrides)
    return ScheduledJob(**base)  # type: ignore[arg-type]


def test_job_happy_path() -> None:
    job = _job()
    assert job.id == "JOB-abcd"
    assert job.cron_expr == "0 7 * * *"
    assert job.skill == "morning_brief"
    assert job.args == ()
    assert job.paused is False
    assert job.last_run_at is None
    assert job.last_status is None


def test_job_is_frozen() -> None:
    job = _job()
    with pytest.raises(ValidationError):
        job.paused = True


def test_job_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ScheduledJob(  # type: ignore[call-arg]
            id="JOB-abcd",
            cron_expr="0 7 * * *",
            skill="x",
            args=(),
            created_at=datetime(2026, 4, 20, tzinfo=UTC),
            created_by=1,
            mystery_field=42,
        )


def test_job_validates_cron_expr_at_construction() -> None:
    with pytest.raises(ValidationError, match="cron"):
        _job(cron_expr="not a cron")


def test_job_rejects_empty_skill() -> None:
    with pytest.raises(ValidationError):
        _job(skill="")


def test_job_rejects_naive_created_at() -> None:
    with pytest.raises(ValidationError):
        _job(created_at=datetime(2026, 4, 20, 10, 0))  # no tz


def test_job_args_is_tuple_of_strings() -> None:
    job = _job(args=("--dry-run", "7"))
    assert job.args == ("--dry-run", "7")
    with pytest.raises(ValidationError):
        _job(args=("ok", 42))  # non-string


def test_job_last_status_is_enum_like() -> None:
    for status in ("ok", "failed", "skipped"):
        j = _job(last_status=status)
        assert j.last_status == status
    with pytest.raises(ValidationError):
        _job(last_status="bogus")


# --- new_job_id ----------------------------------------------------------


def test_new_job_id_format() -> None:
    jid = new_job_id()
    assert re.fullmatch(r"JOB-[0-9a-f]{4}", jid)


def test_new_job_id_is_random() -> None:
    # Not deterministic, but 100 draws should yield >50 uniques
    # unless the RNG is catastrophically broken.
    ids = {new_job_id() for _ in range(100)}
    assert len(ids) > 50
