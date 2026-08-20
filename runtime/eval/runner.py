"""Builds a real HarnessDispatcher per run, dispatches each task variant, grades it.

Reuses `build_harness_dispatcher` directly -- the real classifier, real
Tier1Reasoner, real synthesizer, whichever `smart_provider`/model the
operator's live config pins. This module requires a reachable model to do
anything meaningful and is not exercised by `pytest -m unit` (see
docs/superpowers/specs/2026-08-20-eval-harness-design.md, Testing section).
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.telegram.bot import build_harness_dispatcher
from runtime.config import AegisConfig
from runtime.eval.grading import CallRecord, grade_calls
from runtime.eval.report import TaskResult, VariantResult
from runtime.eval.tasks import EvalTask, ExpectedCall, substitute_sandbox
from runtime.events import EventStream
from runtime.files.client import FilesClient
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.skills.registry import SkillRegistry


class _ObservingHarness:
    """Wraps a real HarnessAdapter; records (tool, args, status) before delegating.

    Eval-only test-seam code. Production code and the real evidence ledger
    are untouched -- see the spec's Grading section for why this exists
    instead of reading tool-call arguments back from the ledger.
    """

    def __init__(self, inner: HarnessAdapter) -> None:
        self._inner = inner
        self.calls: list[CallRecord] = []

    def has_tool(self, name: str) -> bool:
        return self._inner.has_tool(name)

    def execute(self, intent: ToolIntent) -> ToolResult:
        result = self._inner.execute(intent)
        self.calls.append((intent.tool, dict(intent.args), result.status))
        return result


def _seed_fixture(task: EvalTask, sandbox: Path) -> None:
    for f in task.fixture.files:
        target = sandbox / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")


async def run_variant(
    cfg: AegisConfig,
    registry: SkillRegistry,
    tier1_loader: Tier1Loader,
    task: EvalTask,
    variant_text: str,
) -> VariantResult:
    """Run one (task, variant) pair against a fresh sandbox and fresh dispatcher."""
    t0 = time.monotonic()
    sandbox = Path(tempfile.mkdtemp(prefix="aegis-eval-"))
    # Events live in a sibling temp dir, never nested under `sandbox` -- the
    # sandbox is also what FilesClient(allowed_roots=[sandbox]) exposes to
    # the model via files_list/files_search, and an undeclared `sessions/`
    # directory growing inside it would pollute the model's view of the
    # fixture the task author actually declared.
    events_dir = Path(tempfile.mkdtemp(prefix="aegis-eval-events-"))
    try:
        try:
            _seed_fixture(task, sandbox)
            resolved_text = substitute_sandbox(variant_text, sandbox)

            tier3 = Tier3Store()
            events = EventStream(events_dir)
            files_client = FilesClient(allowed_roots=[sandbox])

            dispatcher = build_harness_dispatcher(
                cfg,
                skill_registry=registry,
                tier3=tier3,
                tier1_loader=tier1_loader,
                files_client=files_client,
                events=events,
            )
            if dispatcher is None:
                return VariantResult(
                    task_id=task.id,
                    variant_text=variant_text,
                    passed=False,
                    reason=(
                        "build_harness_dispatcher returned None "
                        "(a hard dependency is unavailable)"
                    ),
                    duration_s=time.monotonic() - t0,
                )

            observing = _ObservingHarness(dispatcher._harness)
            dispatcher._harness = observing

            collected: list[str] = []

            async def _capture(text: str) -> None:
                collected.append(text)

            await dispatcher.dispatch(
                chat_id=1, user_text=resolved_text, message=None, reply=_capture
            )

            resolved_expected = tuple(
                ExpectedCall(
                    tool=ec.tool,
                    args_match=substitute_sandbox(ec.args_match, sandbox),
                )
                for ec in task.expected_calls
            )
            grade = grade_calls(resolved_expected, observing.calls)
            return VariantResult(
                task_id=task.id,
                variant_text=variant_text,
                passed=grade.passed,
                reason=grade.reason,
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:  # eval harness must never crash the batch
            return VariantResult(
                task_id=task.id,
                variant_text=variant_text,
                passed=False,
                reason=f"run_variant raised: {exc!r}",
                duration_s=time.monotonic() - t0,
            )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        shutil.rmtree(events_dir, ignore_errors=True)


async def run_task(
    cfg: AegisConfig,
    registry: SkillRegistry,
    tier1_loader: Tier1Loader,
    task: EvalTask,
) -> TaskResult:
    """Run every variant of one task, sequentially (no concurrency -- keeps
    live-model call ordering predictable and easy to read in the console
    report as it streams)."""
    results: list[VariantResult] = []
    for variant_text in task.variants:
        results.append(await run_variant(cfg, registry, tier1_loader, task, variant_text))
    return TaskResult(task_id=task.id, description=task.description, variants=tuple(results))


__all__ = ["run_task", "run_variant"]
