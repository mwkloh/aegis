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
from runtime.eval.prewarm import prewarm
from runtime.eval.report import EvalReport, TaskResult, render_console, write_json
from runtime.eval.runner import run_task
from runtime.eval.tasks import load_tasks
from runtime.llm.clients import OllamaClient, OpenRouterClient
from runtime.llm.router import ModelRouter, ModelTier
from runtime.llm.telemetry import PRODUCTION_READ_TIMEOUT_S
from runtime.llm.timeouts import read_timeout_override
from runtime.skills.registry import SkillRegistry

_DEFAULT_EVAL_READ_TIMEOUT_S = 300.0

_DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "tasks"
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aegis eval harness (live models).")
    parser.add_argument("--tasks-dir", type=Path, default=_DEFAULT_TASKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument(
        "--yes", action="store_true", help="Skip the cost-confirmation prompt."
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=_DEFAULT_EVAL_READ_TIMEOUT_S,
        help=(
            "Read timeout for eval model calls, in seconds. Deliberately far "
            "above the shipped "
            f"{PRODUCTION_READ_TIMEOUT_S:.0f}s so a slow model is measured on "
            "capability rather than cut off; the report still says which runs "
            "would have breached the production budget. Does not affect the "
            "runtime."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the whole variant set N times per task and report a pass "
            "rate. Live calls are non-deterministic; at n=1 a flaky task is "
            "indistinguishable from a regressed one. Default 1 -- the suite "
            "is slow and costs real tokens."
        ),
    )
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help=(
            "Skip the warm-up call. A cold load was measured at 23.5s, which "
            "would otherwise land inside whichever variant runs first."
        ),
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
        f"{len(tasks)} tasks x{args.repeat}, "
        f"{total_runs * args.repeat} live model calls (at least)"
    )
    if not args.yes:
        confirm = (await asyncio.to_thread(input, "Proceed? [y/N] ")).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0

    registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
    tier1_loader = Tier1Loader(cfg.storage.workspace)

    if not args.no_prewarm:
        client = (
            OllamaClient(cfg) if target.provider == "ollama" else OpenRouterClient(cfg)
        )
        print(f"Pre-warming {target.model}...")
        warmed = await prewarm(client, target.model)
        if warmed is None:
            print("  warm-up failed; cold-load cost will land inside the first task")
        else:
            print(
                f"  resident after {warmed.wall_ms / 1000:.1f}s "
                f"(load {warmed.load_ms / 1000:.1f}s) -- excluded from results"
            )

    started_at = datetime.now(UTC)
    task_results: list[TaskResult] = []
    # Capability is measured under a generous budget; `tgc_within_budget` in the
    # report is what answers the production-timeout question. See
    # docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md.
    with read_timeout_override(args.read_timeout):
        for task in tasks:
            print(f"Running {task.id}...")
            result = await run_task(cfg, registry, tier1_loader, task, repeat=args.repeat)
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
