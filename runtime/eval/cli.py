"""Live-model benchmark entrypoint. Slow, costs real tokens against a live model.

Usage:
    python -m runtime.eval.cli [--tasks-dir DIR] [--out-dir DIR] [--yes]

Never runs in CI, never part of `pytest -m unit`. See
docs/superpowers/specs/2026-08-20-eval-harness-design.md.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.config import get_config
from runtime.eval.report import EvalReport, TaskResult, render_console, write_json
from runtime.eval.runner import run_task
from runtime.eval.tasks import load_tasks
from runtime.llm.router import ModelRouter, ModelTier
from runtime.skills.registry import SkillRegistry

_DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "tasks"
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aegis eval harness (live models).")
    parser.add_argument("--tasks-dir", type=Path, default=_DEFAULT_TASKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument(
        "--yes", action="store_true", help="Skip the cost-confirmation prompt."
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    tasks = load_tasks(args.tasks_dir)
    if not tasks:
        print(f"No tasks found under {args.tasks_dir}", file=sys.stderr)
        return 1

    target = ModelRouter(cfg).route(ModelTier.SMART)
    total_runs = sum(len(t.variants) for t in tasks)
    print(
        f"Pinned: {target.provider} / {target.model}  |  "
        f"{len(tasks)} tasks, {total_runs} live model calls (at least)"
    )
    if not args.yes:
        confirm = (await asyncio.to_thread(input, "Proceed? [y/N] ")).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0

    registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
    tier1_loader = Tier1Loader(cfg.storage.workspace)

    started_at = datetime.now(UTC)
    task_results: list[TaskResult] = []
    for task in tasks:
        print(f"Running {task.id}...")
        result = await run_task(cfg, registry, tier1_loader, task)
        task_results.append(result)

    report = EvalReport(
        provider=target.provider,
        model=target.model,
        started_at=started_at,
        tasks=tuple(task_results),
    )
    print()
    print(render_console(report))
    path = write_json(report, args.out_dir)
    print(f"\nWritten: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
