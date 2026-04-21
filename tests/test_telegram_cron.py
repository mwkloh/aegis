"""Phase 10 Track C — `/cron` slash handler.

Pins:

* Sync `Handler` — returns the reply string. No subprocess, no
  wall-clock sleep. Async/long-running semantics live in the
  scheduler engine; `/cron` just mutates the store and renders.
* Never raises. Bad input (malformed cron, missing args, unknown
  job id) produces a human-readable reply, never a handler_error.
* The handler is produced by a factory that closes over a
  `ScheduledJobStore` + a `Clock`. Tests inject both.
* Sub-verbs: ``add``, ``list``, ``rm``, ``pause``, ``resume``.
  `/cron run` is surfaced in the usage string but replies
  "not yet implemented" — run-now needs an engine-level seam
  we're deferring until Track C' (follow-up).
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.telegram.cron_handler import cron_handler
from runtime.chat.telegram.dispatch import IncomingMessage, ParsedCommand
from runtime.scheduler.store import ScheduledJobStore
from runtime.skills.registry import SkillDescriptor, SkillRegistry, ToolSpec

pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


def _msg(
    *,
    chat_id: int = 1,
    user_id: int = 100,
    text: str = "/cron",
) -> IncomingMessage:
    return IncomingMessage(chat_id=chat_id, user_id=user_id, text=text)


def _cmd(*args: str) -> ParsedCommand:
    return ParsedCommand(name="/cron", args=tuple(args))


def _clock(ts: datetime | None = None) -> Callable[[], datetime]:
    fixed = ts or datetime(2026, 4, 20, 10, 0, tzinfo=UTC)

    def fn() -> datetime:
        return fixed

    return fn


def _store(tmp_path: Path) -> ScheduledJobStore:
    return ScheduledJobStore(tmp_path / "aegis.db")


# --- usage / unknown ------------------------------------------------------


def test_no_subverb_returns_usage(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd())
    assert "add" in reply
    assert "list" in reply
    assert "rm" in reply
    assert "pause" in reply


def test_unknown_subverb_returns_usage(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("frobnicate"))
    assert "add" in reply  # usage hint surfaced


# --- add ------------------------------------------------------------------


def test_add_with_valid_cron_and_skill_succeeds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())
    reply = handle(_msg(user_id=42), _cmd("add", "0 7 * * *", "morning_brief"))

    assert "JOB-" in reply
    assert "morning_brief" in reply
    jobs = store.list_all()
    assert len(jobs) == 1
    assert jobs[0].skill == "morning_brief"
    assert jobs[0].cron_expr == "0 7 * * *"
    assert jobs[0].created_by == 42
    assert jobs[0].args == ()


def test_add_captures_trailing_args(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())
    handle(_msg(), _cmd("add", "*/5 * * * *", "echo", "hello", "world"))

    jobs = store.list_all()
    assert len(jobs) == 1
    assert jobs[0].args == ("hello", "world")


def test_add_with_missing_args_returns_usage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())
    reply = handle(_msg(), _cmd("add"))
    assert "Usage" in reply or "usage" in reply
    assert store.list_all() == []


def test_add_with_only_cron_returns_usage(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())
    reply = handle(_msg(), _cmd("add", "0 7 * * *"))
    assert "Usage" in reply or "usage" in reply
    assert store.list_all() == []


def test_add_with_bad_cron_expression_returns_error(tmp_path: Path) -> None:
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())
    reply = handle(_msg(), _cmd("add", "not-a-cron", "morning_brief"))
    assert "cron" in reply.lower()
    assert store.list_all() == []


# --- list -----------------------------------------------------------------


def test_list_with_no_jobs(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("list"))
    assert "no" in reply.lower() or "0" in reply


def test_list_renders_jobs_with_id_skill_and_cron(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job_a = store.add("0 7 * * *", "morning_brief", (), created_by=1, now=now)
    job_b = store.add("*/6 * * * *", "reindex_vault", (), created_by=1, now=now)
    handle = cron_handler(store=store, clock=_clock(now))

    reply = handle(_msg(), _cmd("list"))

    assert job_a.id in reply
    assert job_b.id in reply
    assert "morning_brief" in reply
    assert "reindex_vault" in reply


def test_list_marks_paused_jobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), created_by=1, now=now)
    store.set_paused(job.id, paused=True)
    handle = cron_handler(store=store, clock=_clock(now))

    reply = handle(_msg(), _cmd("list"))
    # Whatever marker is used, the word "paused" should appear somewhere.
    assert "paused" in reply.lower()


# --- rm -------------------------------------------------------------------


def test_rm_existing_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), created_by=1, now=now)
    handle = cron_handler(store=store, clock=_clock(now))

    reply = handle(_msg(), _cmd("rm", job.id))

    assert job.id in reply
    assert "removed" in reply.lower() or "deleted" in reply.lower()
    assert store.list_all() == []


def test_rm_is_case_insensitive_for_job_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), created_by=1, now=now)
    handle = cron_handler(store=store, clock=_clock(now))

    # job.id is like "JOB-ab12"; operator might type "job-ab12" from phone.
    reply = handle(_msg(), _cmd("rm", job.id.lower()))
    assert "removed" in reply.lower() or "deleted" in reply.lower()
    assert store.list_all() == []


def test_rm_unknown_job(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("rm", "JOB-xxxx"))
    assert "unknown" in reply.lower() or "no such" in reply.lower()


def test_rm_with_no_arg(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("rm"))
    assert "Usage" in reply or "usage" in reply


def test_rm_rejects_system_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now,
    )
    handle = cron_handler(store=store, clock=_clock(now))

    reply = handle(_msg(), _cmd("rm", "SYS-morning-brief"))

    assert "system job" in reply.lower()
    assert "pause" in reply.lower()
    # Row must still be present.
    assert store.get("SYS-morning-brief") is not None


def test_pause_still_works_on_system_job(tmp_path: Path) -> None:
    # The guard is `rm`-only; pause/resume are how operators quiet
    # a system job without removing it.
    store = _store(tmp_path)
    now = datetime(2026, 4, 21, 10, 0, tzinfo=UTC)
    store.upsert_system_job(
        job_id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
        now=now,
    )
    handle = cron_handler(store=store, clock=_clock(now))

    reply = handle(_msg(), _cmd("pause", "SYS-morning-brief"))

    assert "paused" in reply.lower()
    job = store.get("SYS-morning-brief")
    assert job is not None
    assert job.paused is True


# --- pause / resume -------------------------------------------------------


def test_pause_and_resume(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime(2026, 4, 20, 10, 0, tzinfo=UTC)
    job = store.add("0 7 * * *", "morning_brief", (), created_by=1, now=now)
    handle = cron_handler(store=store, clock=_clock(now))

    reply_pause = handle(_msg(), _cmd("pause", job.id))
    assert "paused" in reply_pause.lower()
    assert store.get(job.id) is not None
    assert store.get(job.id).paused is True  # type: ignore[union-attr]

    reply_resume = handle(_msg(), _cmd("resume", job.id))
    assert "resumed" in reply_resume.lower() or "active" in reply_resume.lower()
    assert store.get(job.id).paused is False  # type: ignore[union-attr]


def test_pause_unknown_job(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("pause", "JOB-xxxx"))
    assert "unknown" in reply.lower() or "no such" in reply.lower()


def test_resume_unknown_job(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("resume", "JOB-xxxx"))
    assert "unknown" in reply.lower() or "no such" in reply.lower()


def test_pause_with_no_arg(tmp_path: Path) -> None:
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("pause"))
    assert "Usage" in reply or "usage" in reply


# --- run ------------------------------------------------------------------


def test_run_returns_not_yet_implemented(tmp_path: Path) -> None:
    # Kept in the surface for now; a future track wires this through the
    # engine. Until then the reply must be informative rather than a
    # handler_error exception.
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("run", "JOB-xxxx"))
    assert "not yet" in reply.lower() or "not implemented" in reply.lower()


# --- never raises ---------------------------------------------------------


def test_bad_cron_with_many_fields_handled(tmp_path: Path) -> None:
    # Caller could pass a plausible-looking but invalid 5-field expression.
    handle = cron_handler(store=_store(tmp_path), clock=_clock())
    reply = handle(_msg(), _cmd("add", "60 25 32 13 8", "morning_brief"))
    # Must surface as a user-facing error, not raise.
    assert "cron" in reply.lower()


# --- registry-aware add validation ----------------------------------------


def _registry(*descriptors: SkillDescriptor) -> SkillRegistry:
    return SkillRegistry(list(descriptors))


def _schedulable(skill_id: str) -> SkillDescriptor:
    return SkillDescriptor(
        id=skill_id,
        description=f"schedulable test skill {skill_id}",
        tool=skill_id,
        tools=[ToolSpec(name=skill_id, argv_template=["python", "-m", "x"])],
    )


def _intent_only(skill_id: str) -> SkillDescriptor:
    # Mirrors `echo.yaml` — declares intents but no `tools[]`, so the
    # scheduler can't subprocess it even though the registry finds it.
    return SkillDescriptor(
        id=skill_id,
        description=f"intent-router-only test skill {skill_id}",
        tool=skill_id,
        intents=[skill_id],
    )


def test_add_rejects_unknown_skill_when_registry_supplied(tmp_path: Path) -> None:
    store = _store(tmp_path)
    registry = _registry(_schedulable("morning_brief"))
    handle = cron_handler(store=store, clock=_clock(), registry=registry)

    reply = handle(_msg(), _cmd("add", "0 7 * * *", "frobnicate"))

    assert "unknown" in reply.lower()
    assert "frobnicate" in reply
    # Must list what IS schedulable so operators can self-correct.
    assert "morning_brief" in reply
    # Row must not have been inserted.
    assert store.list_all() == []


def test_add_rejects_intent_router_only_skill(tmp_path: Path) -> None:
    # The scenario that bit us during the 2026-04-21 smoke: `echo` is a
    # registered intent but has no argv_template, so scheduling it
    # produces `skill_misconfigured` every tick forever.
    store = _store(tmp_path)
    registry = _registry(
        _schedulable("morning_brief"),
        _intent_only("echo"),
    )
    handle = cron_handler(store=store, clock=_clock(), registry=registry)

    reply = handle(_msg(), _cmd("add", "* * * * *", "echo", "hi"))

    assert "not schedulable" in reply.lower() or "argv_template" in reply.lower()
    assert "echo" in reply
    assert store.list_all() == []


def test_add_accepts_schedulable_skill_with_registry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    registry = _registry(_schedulable("morning_brief"))
    handle = cron_handler(store=store, clock=_clock(), registry=registry)

    reply = handle(_msg(), _cmd("add", "0 7 * * *", "morning_brief"))

    assert "JOB-" in reply
    assert len(store.list_all()) == 1


def test_add_without_registry_skips_skill_validation(tmp_path: Path) -> None:
    # Backwards-compat: the existing dispatch wiring and many unit tests
    # construct cron_handler without a registry. In that mode the add
    # path has to keep accepting arbitrary skill strings — the scheduler
    # will surface `skill_misconfigured` at tick time instead.
    store = _store(tmp_path)
    handle = cron_handler(store=store, clock=_clock())

    reply = handle(_msg(), _cmd("add", "0 7 * * *", "anything_goes"))

    assert "JOB-" in reply
    assert len(store.list_all()) == 1


def test_add_error_lists_nothing_when_registry_has_no_schedulable_skills(
    tmp_path: Path,
) -> None:
    # Degenerate case: registry is loaded but contains only intent-only
    # descriptors. We shouldn't emit "Schedulable skills: ." with an
    # empty list — the reply should just say the requested skill is
    # unknown.
    store = _store(tmp_path)
    registry = _registry(_intent_only("echo"))
    handle = cron_handler(store=store, clock=_clock(), registry=registry)

    reply = handle(_msg(), _cmd("add", "0 7 * * *", "frobnicate"))

    assert "unknown" in reply.lower()
    assert "frobnicate" in reply
    assert "schedulable skills:" not in reply.lower()
