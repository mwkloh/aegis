"""Phase 10 Track B — `JobRunner` skill-invocation adapter.

Plugs into `SchedulerEngine` as a `JobInvoker`. For each fired job it:

1. Looks the skill up in `SkillRegistry`.
2. Resolves argv via an injected resolver (so this runner stays
   argv-only — no shell, no substitution).
3. Runs the subprocess via the same `SubprocessRunner` protocol the
   Telegram `/apply` / `/brief` path already uses.
4. Pushes user-facing output back to the operator — but only if the
   skill produced non-empty output. Silent skills (`reindex_vault`)
   don't spam chat at 03:00.

Failure modes all surface as `JobOutcome(status="failed",
detail=<short_classifier>)`:

* ``unknown_skill``         — registry lookup returned None
* ``skill_misconfigured``   — resolver could not build argv
* ``exit_<N>``              — subprocess exited non-zero
* ``<ExceptionClassName>``  — subprocess runner raised
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.scheduler.engine import JobOutcome
from runtime.scheduler.job import ScheduledJob
from runtime.skills.registry import SkillDescriptor, SkillRegistry

if TYPE_CHECKING:
    # Type-only — importing at runtime loads `runtime.chat.telegram.__init__`
    # which imports bot.py which imports `runtime.scheduler` → circular.
    # `SubprocessRunner` is a Protocol, so duck typing works at runtime.
    from runtime.chat.telegram.long_running import SubprocessRunner

ArgvResolver = Callable[[SkillDescriptor, tuple[str, ...]], list[str] | None]
Deliverer = Callable[[str], Awaitable[None]]

# Aligned with `MAX_LONG_RUNNING_CHARS` — headroom below Telegram's 4096 cap.
_DEFAULT_MAX_BODY_CHARS = 3500


class JobRunner:
    """Skill-invocation adapter. Matches `JobInvoker` signature. Never raises."""

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        subprocess_runner: SubprocessRunner,
        workspace: Path,
        argv_resolver: ArgvResolver,
        deliver: Deliverer | None = None,
        max_body_chars: int = _DEFAULT_MAX_BODY_CHARS,
    ) -> None:
        self._registry = registry
        self._runner = subprocess_runner
        self._workspace = workspace
        self._resolver = argv_resolver
        self._deliver = deliver
        self._max_body_chars = max_body_chars

    async def __call__(self, job: ScheduledJob) -> JobOutcome:
        descriptor = self._registry.get(job.skill)
        if descriptor is None:
            return JobOutcome(status="failed", detail="unknown_skill")

        argv = self._resolver(descriptor, job.args)
        if argv is None:
            return JobOutcome(status="failed", detail="skill_misconfigured")

        try:
            exit_code, output = await self._runner.run(argv, cwd=self._workspace)
        except Exception as exc:
            # Only the class name — skill stderr / exc args never leak
            # into `detail`, which the engine surfaces as `error_class`.
            return JobOutcome(status="failed", detail=type(exc).__name__)

        if exit_code != 0:
            banner = f"{job.skill} failed (exit={exit_code})"
            if output:
                banner = f"{banner}\n\n{output}"
            await self._push(banner)
            return JobOutcome(status="failed", detail=f"exit_{exit_code}")

        if output.strip():
            await self._push(output)
        return JobOutcome(status="ok")

    async def _push(self, body: str) -> None:
        if self._deliver is None:
            return
        if len(body) > self._max_body_chars:
            body = body[: self._max_body_chars]
        await self._deliver(body)


__all__ = ["ArgvResolver", "Deliverer", "JobRunner"]
