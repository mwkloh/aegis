"""Phase 10 Track B — `JobRunner` skill-invocation adapter.

Pins:

* Plugs into `SchedulerEngine` as a `JobInvoker` (`async (job) -> JobOutcome`).
* Skill invocation goes through the same `SubprocessRunner` protocol
  the Telegram `LongRunningRunner` already uses — no private path.
* Delivery is optional. A silent skill (e.g. `reindex_vault`) may
  return empty stdout; a noisy skill (`morning_brief`) returns
  markdown the operator wants pushed. The runner pushes iff the
  skill produced non-empty user-facing output.
* Failures never raise. A missing skill, a bad resolver, or a
  crashing subprocess produces `JobOutcome(status="failed", detail=...)`.
* `detail` on failure carries a short classifier string (not a
  stack trace / not skill output) so the engine can emit it as
  `error_class` in the event payload.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.scheduler.job import ScheduledJob, new_job_id
from runtime.scheduler.runner import JobRunner
from runtime.skills.registry import SkillDescriptor, SkillRegistry, ToolSpec

pytestmark = pytest.mark.unit


# --- fixtures -------------------------------------------------------------


def _descriptor(
    skill_id: str = "morning_brief",
    *,
    with_tool: bool = True,
) -> SkillDescriptor:
    tools: list[ToolSpec] = []
    if with_tool:
        tools.append(
            ToolSpec(
                name="run",
                argv_template=["python", "-m", f"runtime.skills.scripts.{skill_id}"],
            )
        )
    return SkillDescriptor(
        id=skill_id,
        description="test skill",
        intents=[skill_id.replace("_", " ")],
        tool="subprocess",
        tools=tools,
    )


def _registry(*descriptors: SkillDescriptor) -> SkillRegistry:
    return SkillRegistry(list(descriptors))


def _job(
    skill: str = "morning_brief",
    args: tuple[str, ...] = (),
) -> ScheduledJob:
    return ScheduledJob(
        id=new_job_id(),
        cron_expr="0 7 * * *",
        skill=skill,
        args=args,
        created_at=datetime(2026, 4, 20, 10, 0, tzinfo=UTC),
        created_by=1,
    )


class _FakeSubprocess:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        output: str = "",
        exc: Exception | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.output = output
        self.exc = exc
        self.calls: list[tuple[list[str], Path]] = []

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]:
        self.calls.append((list(argv), cwd))
        if self.exc is not None:
            raise self.exc
        return self.exit_code, self.output


def _identity_resolver(
    scheduled_args: tuple[str, ...] | None = None,
) -> Callable[[SkillDescriptor, tuple[str, ...]], list[str] | None]:
    """Resolver that just returns the descriptor's argv_template, plus args."""

    def resolve(
        descriptor: SkillDescriptor, args: tuple[str, ...]
    ) -> list[str] | None:
        if not descriptor.tools:
            return None
        base = list(descriptor.tools[0].argv_template)
        return base + list(args)

    return resolve


def _null_resolver(
    _: SkillDescriptor, __: tuple[str, ...]
) -> list[str] | None:
    return None


def _collect_deliveries() -> tuple[
    list[str], Callable[[str], Awaitable[None]]
]:
    sink: list[str] = []

    async def deliver(text: str) -> None:
        sink.append(text)

    return sink, deliver


# --- happy path -----------------------------------------------------------


async def test_fires_skill_and_delivers_output(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exit_code=0, output="# Morning brief\n\nall quiet.")
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job())

    assert outcome.status == "ok"
    assert len(subp.calls) == 1
    argv, cwd = subp.calls[0]
    assert argv == ["python", "-m", "runtime.skills.scripts.morning_brief"]
    assert cwd == tmp_path
    assert len(sink) == 1
    # Delivery contains the skill output; formatting is allowed to
    # add a header but must preserve the body.
    assert "all quiet." in sink[0]


async def test_scheduled_args_appended_to_argv(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exit_code=0, output="done")
    _sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    await runner(_job(args=("--dry-run", "--verbose")))

    argv, _ = subp.calls[0]
    assert argv[-2:] == ["--dry-run", "--verbose"]


# --- silent success -------------------------------------------------------


async def test_empty_output_skips_delivery(tmp_path: Path) -> None:
    reg = _registry(_descriptor("reindex_vault"))
    subp = _FakeSubprocess(exit_code=0, output="")
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job(skill="reindex_vault"))

    assert outcome.status == "ok"
    assert sink == []  # silent skill — no push


async def test_whitespace_only_output_skips_delivery(tmp_path: Path) -> None:
    reg = _registry(_descriptor("reindex_vault"))
    subp = _FakeSubprocess(exit_code=0, output="   \n\t\n")
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job(skill="reindex_vault"))

    assert outcome.status == "ok"
    assert sink == []


# --- failure modes --------------------------------------------------------


async def test_unknown_skill_returns_failed(tmp_path: Path) -> None:
    reg = _registry(_descriptor())  # only morning_brief
    subp = _FakeSubprocess()
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job(skill="nonexistent"))

    assert outcome.status == "failed"
    assert outcome.detail == "unknown_skill"
    assert subp.calls == []
    assert sink == []  # no delivery on "unknown skill"


async def test_resolver_returns_none_is_misconfigured(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess()
    _sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_null_resolver,
        deliver=deliver,
    )

    outcome = await runner(_job())

    assert outcome.status == "failed"
    assert outcome.detail == "skill_misconfigured"
    assert subp.calls == []


async def test_nonzero_exit_delivers_error_and_fails(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exit_code=2, output="traceback: KaBoom")
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job())

    assert outcome.status == "failed"
    assert outcome.detail == "exit_2"
    assert len(sink) == 1
    assert "failed" in sink[0].lower()
    assert "morning_brief" in sink[0]


async def test_subprocess_exception_fails_without_raising(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exc=RuntimeError("exec failed"))
    _sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
    )

    outcome = await runner(_job())

    assert outcome.status == "failed"
    assert outcome.detail == "RuntimeError"
    # Error message body never leaks skill stderr into detail
    assert "exec failed" not in (outcome.detail or "")


# --- delivery callback is optional ---------------------------------------


async def test_runs_without_delivery_callback(tmp_path: Path) -> None:
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exit_code=0, output="# Brief")
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=None,
    )

    outcome = await runner(_job())

    assert outcome.status == "ok"
    assert len(subp.calls) == 1


# --- output clipping ------------------------------------------------------


async def test_large_output_is_clipped_before_delivery(tmp_path: Path) -> None:
    huge = "x" * 10_000
    reg = _registry(_descriptor())
    subp = _FakeSubprocess(exit_code=0, output=huge)
    sink, deliver = _collect_deliveries()
    runner = JobRunner(
        registry=reg,
        subprocess_runner=subp,
        workspace=tmp_path,
        argv_resolver=_identity_resolver(),
        deliver=deliver,
        max_body_chars=512,
    )

    outcome = await runner(_job())

    assert outcome.status == "ok"
    assert len(sink) == 1
    assert len(sink[0]) <= 512
