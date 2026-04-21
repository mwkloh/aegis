"""Phase 10 Track A2 — `ScheduledJobStore` (sqlite persistence).

Pins:

* Schema lives in `memory/store_sqlite.py` (v3 migration, additive
  — no touch to Phase 7 tables). Store just wires CRUD on top.
* `add()` mints a `JOB-xxxx` id, retrying on the (theoretical)
  collision case. After N retries it raises `RuntimeError` rather
  than silently looping.
* Timestamps round-trip as tz-aware UTC. Naive datetimes never
  enter the DB — same posture as the cron module.
* `record_run()` is the only mutation that touches `last_run_at` /
  `last_status`. The engine will call it; `/cron` handlers won't.
* `list_all()` is deterministic (sorted by `created_at`, then id)
  so `/cron list` output stays stable across calls.
* `remove()` returns a bool — True if deleted, False if not found
  — so the slash handler can tell the operator "no such job".
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from runtime.scheduler.store import ScheduledJobStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "aegis-index.db")


# --- add / get -----------------------------------------------------------


def test_add_returns_complete_record(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add(
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        created_by=7846523803,
        now=now,
    )
    assert job.cron_expr == "0 7 * * *"
    assert job.skill == "morning_brief"
    assert job.args == ()
    assert job.created_at == now
    assert job.created_by == 7846523803
    assert job.last_run_at is None
    assert job.last_status is None
    assert job.paused is False
    assert job.id.startswith("JOB-")


def test_add_then_get_roundtrip(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    added = store.add(
        cron_expr="*/5 * * * *",
        skill="reindex_vault",
        args=("--fast",),
        created_by=1,
        now=now,
    )
    fetched = store.get(added.id)
    assert fetched == added


def test_get_missing_returns_none(store: ScheduledJobStore) -> None:
    assert store.get("JOB-dead") is None


def test_add_rejects_invalid_cron(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="cron"):
        store.add(
            cron_expr="not a cron",
            skill="x",
            args=(),
            created_by=1,
            now=now,
        )


def test_add_rejects_naive_now(store: ScheduledJobStore) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        store.add(
            cron_expr="0 7 * * *",
            skill="x",
            args=(),
            created_by=1,
            now=datetime(2026, 4, 20, 10, 0),
        )


# --- list_all ------------------------------------------------------------


def test_list_all_empty(store: ScheduledJobStore) -> None:
    assert store.list_all() == []


def test_list_all_sorted_by_created_then_id(store: ScheduledJobStore) -> None:
    base = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    j1 = store.add("0 7 * * *", "a", (), 1, now=base)
    j2 = store.add("0 8 * * *", "b", (), 1, now=base + timedelta(seconds=1))
    j3 = store.add("0 9 * * *", "c", (), 1, now=base + timedelta(seconds=2))
    jobs = store.list_all()
    assert [j.id for j in jobs] == [j1.id, j2.id, j3.id]


# --- remove --------------------------------------------------------------


def test_remove_existing_returns_true(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "x", (), 1, now=now)
    assert store.remove(job.id) is True
    assert store.get(job.id) is None


def test_remove_missing_returns_false(store: ScheduledJobStore) -> None:
    assert store.remove("JOB-dead") is False


# --- set_paused ----------------------------------------------------------


def test_set_paused_toggles_flag(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "x", (), 1, now=now)
    paused = store.set_paused(job.id, paused=True)
    assert paused is not None
    assert paused.paused is True
    resumed = store.set_paused(job.id, paused=False)
    assert resumed is not None
    assert resumed.paused is False


def test_set_paused_missing_returns_none(store: ScheduledJobStore) -> None:
    assert store.set_paused("JOB-dead", paused=True) is None


# --- record_run ----------------------------------------------------------


def test_record_run_updates_last_run_and_status(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "x", (), 1, now=now)
    later = datetime(2026, 4, 20, 7, 0, tzinfo=UTC)
    updated = store.record_run(job.id, at=later, status="ok")
    assert updated is not None
    assert updated.last_run_at == later
    assert updated.last_status == "ok"


def test_record_run_missing_returns_none(store: ScheduledJobStore) -> None:
    at = datetime(2026, 4, 20, 7, 0, tzinfo=UTC)
    assert store.record_run("JOB-dead", at=at, status="ok") is None


def test_record_run_rejects_naive_at(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "x", (), 1, now=now)
    with pytest.raises(ValueError, match="tz-aware"):
        store.record_run(job.id, at=datetime(2026, 4, 20, 7, 0), status="ok")


def test_record_run_rejects_bad_status(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "x", (), 1, now=now)
    at = datetime(2026, 4, 20, 7, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="status"):
        store.record_run(job.id, at=at, status="bogus")  # type: ignore[arg-type]


# --- args roundtrip ------------------------------------------------------


def test_args_tuple_with_whitespace_and_unicode(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    args = ("--flag", "value with spaces", "日本語", "")
    job = store.add("0 7 * * *", "x", args, 1, now=now)
    fetched = store.get(job.id)
    assert fetched is not None
    assert fetched.args == args


# --- persistence across instances ----------------------------------------


def test_jobs_persist_across_store_instances(tmp_path: Path) -> None:
    db = tmp_path / "aegis-index.db"
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    s1 = ScheduledJobStore(db)
    job = s1.add("0 7 * * *", "x", ("a",), 1, now=now)
    s2 = ScheduledJobStore(db)
    assert s2.get(job.id) == job


# --- collision handling --------------------------------------------------


def test_add_retries_on_id_collision(
    store: ScheduledJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    first = store.add("0 7 * * *", "x", (), 1, now=now)
    # Force the next id mint to collide once, then return a fresh id.
    taken = first.id
    fresh = "JOB-beef"
    calls = iter([taken, fresh])
    monkeypatch.setattr(
        "runtime.scheduler.store.new_job_id", lambda: next(calls)
    )
    second = store.add("0 8 * * *", "y", (), 1, now=now)
    assert second.id == fresh


def test_add_raises_after_exhausting_retries(
    store: ScheduledJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    first = store.add("0 7 * * *", "x", (), 1, now=now)
    monkeypatch.setattr(
        "runtime.scheduler.store.new_job_id", lambda: first.id
    )
    with pytest.raises(RuntimeError, match="collision"):
        store.add("0 8 * * *", "y", (), 1, now=now)


# --- upsert_system_job ---------------------------------------------------


def test_upsert_system_job_inserts_when_absent(store: ScheduledJobStore) -> None:
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    job = store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now,
    )
    assert job.id == "SYS-morning-brief"
    assert job.skill == "morning_brief"
    assert job.created_by == 0
    assert job.paused is False
    assert job.last_run_at is None


def test_upsert_system_job_is_idempotent(store: ScheduledJobStore) -> None:
    now1 = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    store.upsert_system_job(
        job_id="SYS-vault-reindex",
        cron_expr="30 3 * * *",
        skill="vault_reindex",
        args=(),
        now=now1,
    )
    now2 = now1 + timedelta(days=1)
    again = store.upsert_system_job(
        job_id="SYS-vault-reindex",
        cron_expr="0 0 * * *",        # different cron — should be ignored
        skill="vault_reindex_new",     # different skill — should be ignored
        args=("would-overwrite",),
        now=now2,
    )
    # Original row wins — second call is a no-op.
    assert again.cron_expr == "30 3 * * *"
    assert again.skill == "vault_reindex"
    assert again.args == ()
    assert again.created_at == now1
    assert len(store.list_all()) == 1


def test_upsert_system_job_preserves_paused_state(
    store: ScheduledJobStore,
) -> None:
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now,
    )
    # Operator pauses the system job.
    store.set_paused("SYS-morning-brief", paused=True)
    # Reboot: seed runs again.
    reseeded = store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now + timedelta(hours=1),
    )
    assert reseeded.paused is True


def test_upsert_system_job_preserves_last_run(
    store: ScheduledJobStore,
) -> None:
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    store.upsert_system_job(
        job_id="SYS-reflection-sweep",
        cron_expr="15 3 * * *",
        skill="reflection_sweep",
        args=(),
        now=now,
    )
    ran_at = now + timedelta(hours=2)
    store.record_run("SYS-reflection-sweep", at=ran_at, status="ok")
    reseeded = store.upsert_system_job(
        job_id="SYS-reflection-sweep",
        cron_expr="15 3 * * *",
        skill="reflection_sweep",
        args=(),
        now=now + timedelta(days=1),
    )
    assert reseeded.last_run_at == ran_at
    assert reseeded.last_status == "ok"


def test_upsert_system_job_rejects_naive_now(
    store: ScheduledJobStore,
) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        store.upsert_system_job(
            job_id="SYS-x",
            cron_expr="0 7 * * *",
            skill="x",
            args=(),
            now=datetime(2026, 4, 21, 10, 0),  # naive
        )


def test_upsert_system_job_rejects_invalid_cron(
    store: ScheduledJobStore,
) -> None:
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="5 whitespace-separated fields"):
        store.upsert_system_job(
            job_id="SYS-x",
            cron_expr="not a cron",
            skill="x",
            args=(),
            now=now,
        )


def test_upsert_system_job_and_operator_add_coexist(
    store: ScheduledJobStore,
) -> None:
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    sys_job = store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now,
    )
    op_job = store.add(
        cron_expr="*/5 * * * *",
        skill="echo",
        args=("hi",),
        created_by=7846523803,
        now=now,
    )
    all_jobs = store.list_all()
    assert {j.id for j in all_jobs} == {sys_job.id, op_job.id}
    # Operator's row has a positive created_by; system's is zero —
    # handlers can distinguish without parsing the id prefix.
    op_row = next(j for j in all_jobs if j.id == op_job.id)
    sys_row = next(j for j in all_jobs if j.id == sys_job.id)
    assert op_row.created_by > 0
    assert sys_row.created_by == 0
