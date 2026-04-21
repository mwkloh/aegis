"""Phase 10 Track D — `seed_system_jobs` idempotency + resilience."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runtime.scheduler.seed import (
    SYSTEM_JOBS,
    SystemJobSpec,
    seed_system_jobs,
)
from runtime.scheduler.store import ScheduledJobStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "aegis-index.db")


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 21, 10, 0, tzinfo=UTC)


def test_default_specs_cover_four_recurring_jobs() -> None:
    ids = {spec.id for spec in SYSTEM_JOBS}
    assert ids == {
        "SYS-morning-brief",
        "SYS-tier2-compress",
        "SYS-reflection-sweep",
        "SYS-vault-reindex",
    }


def test_all_default_specs_use_sys_prefix() -> None:
    # The `/cron rm` guard keys on this prefix — it's load-bearing.
    for spec in SYSTEM_JOBS:
        assert spec.id.startswith("SYS-")


def test_seed_inserts_all_on_first_run(
    store: ScheduledJobStore, now: datetime
) -> None:
    count = seed_system_jobs(store, now=now)
    assert count == len(SYSTEM_JOBS)
    jobs = store.list_all()
    assert len(jobs) == len(SYSTEM_JOBS)
    assert all(j.created_by == 0 for j in jobs)


def test_seed_is_idempotent(store: ScheduledJobStore, now: datetime) -> None:
    first = seed_system_jobs(store, now=now)
    second = seed_system_jobs(store, now=now + timedelta(hours=1))
    assert first > 0
    assert second == 0  # nothing new inserted on the re-run
    assert len(store.list_all()) == len(SYSTEM_JOBS)


def test_seed_preserves_operator_pause(
    store: ScheduledJobStore, now: datetime
) -> None:
    seed_system_jobs(store, now=now)
    store.set_paused("SYS-morning-brief", paused=True)
    seed_system_jobs(store, now=now + timedelta(days=1))
    paused_after_reboot = store.get("SYS-morning-brief")
    assert paused_after_reboot is not None
    assert paused_after_reboot.paused is True


def test_seed_continues_after_one_spec_fails(
    store: ScheduledJobStore, now: datetime
) -> None:
    specs = (
        SystemJobSpec(id="SYS-a", cron_expr="0 7 * * *", skill="a"),
        SystemJobSpec(id="SYS-b", cron_expr="garbage", skill="b"),   # invalid cron
        SystemJobSpec(id="SYS-c", cron_expr="0 9 * * *", skill="c"),
    )
    count = seed_system_jobs(store, now=now, specs=specs)
    # Two succeed; the garbage-cron row is skipped with a log, not raised.
    assert count == 2
    ids = {j.id for j in store.list_all()}
    assert ids == {"SYS-a", "SYS-c"}


def test_seed_accepts_custom_specs(
    store: ScheduledJobStore, now: datetime
) -> None:
    custom = (SystemJobSpec(id="SYS-custom", cron_expr="*/5 * * * *", skill="x"),)
    seed_system_jobs(store, now=now, specs=custom)
    got = store.get("SYS-custom")
    assert got is not None
    assert got.skill == "x"
    assert got.cron_expr == "*/5 * * * *"
