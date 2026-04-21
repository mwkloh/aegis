"""`/cron` slash — add, list, rm, pause, resume scheduled jobs.

Phase 10 Track C. One handler slot on the dispatcher. Sub-verbs keep
the surface small and operator-legible:

* ``/cron add "<cron>" <skill> [args...]`` — register a new job.
* ``/cron list`` — show all jobs with their id, skill, cron
  expression, next-run ETA, and paused marker.
* ``/cron rm <job_id>`` — delete a job.
* ``/cron pause <job_id>`` — temporarily stop a job from firing
  without losing its definition.
* ``/cron resume <job_id>`` — reactivate a paused job.
* ``/cron run <job_id>`` — surfaced in the usage string but returns
  a "not yet implemented" reply. Run-now needs an engine-level
  seam; shipping it is a follow-up track.

The handler is **sync** (returns ``str``) — scheduling state lives
in sqlite and mutations are fast. The tick loop + subprocess work
happens elsewhere (engine + runner), so this file stays IO-light
and testable without asyncio.
"""
from __future__ import annotations

from datetime import datetime

from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.scheduler.cron import describe, next_run
from runtime.scheduler.engine import Clock
from runtime.scheduler.job import ScheduledJob
from runtime.scheduler.store import ScheduledJobStore
from runtime.skills import SkillRegistry

_USAGE = (
    'Usage: /cron add "<cron>" <skill> [args...] '
    "| list | rm <id> | pause <id> | resume <id> | run <id>\n"
    "Cron expressions are interpreted as UTC."
)

_MIN_ADD_ARGS = 2  # cron expression + skill id


def cron_handler(
    *,
    store: ScheduledJobStore,
    clock: Clock,
    registry: SkillRegistry | None = None,
) -> Handler:
    """Factory for the dispatcher. Closes over store + clock deps.

    When ``registry`` is supplied, ``/cron add`` validates that the
    requested skill is both registered and *schedulable* (i.e. its
    descriptor carries at least one ``tools[]`` entry with an
    ``argv_template``). Without the registry the add-path still
    works for unit tests that don't care about skill wiring — the
    scheduler will surface ``skill_misconfigured`` at tick time
    instead of rejecting at add time.
    """

    def _handle(msg: IncomingMessage, cmd: ParsedCommand) -> str:  # noqa: PLR0911 - one return per sub-verb
        if not cmd.args:
            return _USAGE
        sub = cmd.args[0].strip().lower()
        tail = cmd.args[1:]

        if sub == "add":
            return _add(store, clock, tail, created_by=msg.user_id, registry=registry)
        if sub == "list":
            return _list(store, clock)
        if sub == "rm":
            return _rm(store, tail)
        if sub == "pause":
            return _set_paused(store, tail, paused=True)
        if sub == "resume":
            return _set_paused(store, tail, paused=False)
        if sub == "run":
            return (
                "/cron run is not yet implemented — deferred until the "
                "engine exposes an immediate-fire seam."
            )
        return _USAGE

    return _handle


# --- sub-verb implementations ----------------------------------------------


def _add(
    store: ScheduledJobStore,
    clock: Clock,
    tail: tuple[str, ...],
    *,
    created_by: int,
    registry: SkillRegistry | None,
) -> str:
    if len(tail) < _MIN_ADD_ARGS:
        return _USAGE
    cron_expr, skill, *extra = tail
    args = tuple(extra)
    if registry is not None:
        reject = _validate_skill(registry, skill)
        if reject is not None:
            return reject
    try:
        job = store.add(cron_expr, skill, args, created_by=created_by, now=clock())
    except ValueError as exc:
        # Covers naive timestamp + invalid cron (validate_cron raises ValueError).
        return f"Cannot schedule: {exc}"
    try:
        eta = next_run(job.cron_expr, after=clock())
        eta_label = eta.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        # `store.add` already ran `validate`, so this branch is unreachable
        # in practice; keeping it defensive so `/cron add` never raises.
        eta_label = "unknown"
    return (
        f"{job.id} scheduled · {job.skill} · {describe(job.cron_expr)}\n"
        f"next run: {eta_label}"
    )


def _list(store: ScheduledJobStore, clock: Clock) -> str:
    jobs = store.list_all()
    if not jobs:
        return "No scheduled jobs."
    now = clock()
    lines = [f"{len(jobs)} scheduled job(s):"]
    for job in jobs:
        lines.append(_render_list_row(job, now=now))
    return "\n".join(lines)


def _render_list_row(job: ScheduledJob, *, now: datetime) -> str:
    try:
        eta = next_run(job.cron_expr, after=job.last_run_at or now)
        eta_label = eta.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        eta_label = "unknown"
    marker = " [paused]" if job.paused else ""
    return (
        f"  • {job.id}  {job.skill}  {describe(job.cron_expr)}  "
        f"→ {eta_label}{marker}"
    )


def _validate_skill(registry: SkillRegistry, skill: str) -> str | None:
    """Return a user-facing rejection message, or None if OK to schedule.

    Two failure modes deserve different wording so the operator knows
    whether to fix a typo or pick a different skill:

    * Unknown skill id → list the available schedulable ones.
    * Known skill but intent-router-only (no ``tools[]``) → tell them
      the skill exists but can't be scheduled as a subprocess.
    """
    desc = registry.get(skill)
    if desc is None:
        schedulable = sorted(d.id for d in registry.all() if d.tools)
        if schedulable:
            options = ", ".join(schedulable)
            return f"Unknown skill {skill!r}. Schedulable skills: {options}."
        return f"Unknown skill {skill!r}."
    if not desc.tools:
        return (
            f"Skill {skill!r} exists but is not schedulable — it has no "
            f"argv_template. Use it via chat/intent routing instead."
        )
    return None


def _rm(store: ScheduledJobStore, tail: tuple[str, ...]) -> str:
    if not tail:
        return _USAGE
    job_id = _normalize_job_id(tail[0])
    if job_id.startswith("SYS-"):
        # System jobs are seeded at boot and owned by the deployment,
        # not the operator. `/cron pause` still works if they want to
        # temporarily stop one; removing would just get re-seeded.
        return (
            f"{job_id} is a system job and can't be removed. "
            f"Use /cron pause {job_id} instead."
        )
    if store.remove(job_id):
        return f"{job_id} removed."
    return f"Unknown job: {job_id}."


def _set_paused(
    store: ScheduledJobStore,
    tail: tuple[str, ...],
    *,
    paused: bool,
) -> str:
    if not tail:
        return _USAGE
    job_id = _normalize_job_id(tail[0])
    updated = store.set_paused(job_id, paused=paused)
    if updated is None:
        return f"Unknown job: {job_id}."
    if paused:
        return f"{job_id} paused."
    return f"{job_id} resumed."


def _normalize_job_id(raw: str) -> str:
    """Canonicalise `job-abcd` → `JOB-abcd` for phone-typed ids."""
    trimmed = raw.strip()
    if trimmed.lower().startswith("job-"):
        return "JOB-" + trimmed.split("-", 1)[1].lower()
    return trimmed


__all__ = ["cron_handler"]
