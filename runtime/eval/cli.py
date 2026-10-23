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
from runtime.config import AegisConfig, get_config
from runtime.eval.bare import run_variant_bare
from runtime.eval.locality import classify_locality
from runtime.eval.prewarm import prewarm
from runtime.eval.report import EvalReport, TaskResult, render_console, write_json
from runtime.eval.runner import VariantRunner, run_task, run_variant
from runtime.eval.tasks import load_tasks
from runtime.llm.clients import OllamaClient, OpenRouterClient
from runtime.llm.router import ModelRouter, ModelTier
from runtime.llm.telemetry import PRODUCTION_READ_TIMEOUT_S
from runtime.llm.timeouts import read_timeout_override
from runtime.skills.registry import SkillRegistry

_DEFAULT_EVAL_READ_TIMEOUT_S = 300.0

# Every arm but `bare` is a preset of the production config, so whatever wins can ship
# as a ModelProfile (spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md).
_GUARDED = frozenset({"autofill", "claim_gate", "evidence"})
# (harness mode, guardrails, native prompt). The arms measured on nights 1-2 keep the
# bare prompt so reruns stay comparable; native-guard-id is native-guard with the
# production identity prompt. The prompt only matters in native mode.
_ARM_PRESETS: dict[str, tuple[str, frozenset[str], str]] = {
    "full": ("planner", _GUARDED | {"plan_lock"}, "bare"),
    "full-nolock": ("planner", _GUARDED, "bare"),
    "native": ("native", frozenset(), "bare"),
    "native-guard": ("native", _GUARDED, "bare"),
    "native-guard-synth": ("native", _GUARDED | {"synthesis"}, "bare"),
    "native-guard-id": ("native", _GUARDED, "identity"),
}
ARMS: tuple[str, ...] = (*_ARM_PRESETS, "bare")
_NATIVE_ARMS = frozenset(
    {"native", "native-guard", "native-guard-synth", "native-guard-id", "bare"}
)


def _apply_arm(cfg: AegisConfig, arm: str) -> AegisConfig:
    """Pin the harness mode, guardrails and native prompt for one arm, overriding any
    profile."""
    preset = _ARM_PRESETS.get(arm)
    if preset is None:  # `bare` runs its own eval-only loop
        return cfg
    mode, guardrails, native_prompt = preset
    return cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={
                    "harness_mode": mode,
                    "guardrails": guardrails,
                    "native_prompt": native_prompt,
                }
            )
        }
    )

_DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "tasks"
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "results"
# The workspace a fresh install is seeded from (bootstrap copies it to ~/.aegis/workspace).
_TEMPLATE_WORKSPACE = Path(__file__).resolve().parent.parent.parent / "templates" / "workspace"


def _tier1_root() -> Path:
    """Where every arm reads IDENTITY.md (and USER.md) from: the repo's Aegis template.

    The eval measures what a fresh install sends: pinned by the commit, the same on every
    machine, and never the operator's own persona (`cfg.storage.workspace`), which may be
    private. This reaches the native identity prompt and reply synthesis alike. Arms
    with synthesis therefore differ from nights 1-2, which read the operator workspace,
    in that one prompt line (decided 2026-10-23).
    """
    return _TEMPLATE_WORKSPACE


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
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default="full",
        help=(
            "Which harness drives the model; same models, tools, tasks and "
            "grader, to measure what each guardrail adds. "
            "full (default) = AEGIS planner; "
            "full-nolock = planner without plan-lock; "
            "native = native tool calls, no guardrails; "
            "native-guard = plus autofill, claim gate and evidence; "
            "native-guard-synth = plus a synthesis call; "
            "native-guard-id = native-guard with the production identity prompt "
            "(the other native arms keep the bare prompt they were measured with); "
            "bare = the eval-only control loop. "
            "Native arms and bare: Ollama-served models only."
        ),
    )
    parser.add_argument(
        "--require-local",
        action="store_true",
        help=(
            "Refuse to run unless the model is pulled onto this machine's "
            "Ollama. The benchmark asks how well small models run on-device, so "
            "an SLM config must not quietly resolve to a cloud or hosted model."
        ),
    )
    return parser.parse_args(argv)


def _locality_refusal(locality: str, *, require_local: bool) -> str | None:
    """Why a run must not proceed, or None. Only `--require-local` ever refuses."""
    if require_local and locality != "local":
        return f"--require-local: this model is {locality}, not on this device"
    return None


def _variant_runner(arm: str) -> VariantRunner:
    return run_variant_bare if arm == "bare" else run_variant


def _arm_unsupported_reason(arm: str, provider: str) -> str | None:
    """Why this arm cannot run against this provider, or None if it can."""
    if arm in _NATIVE_ARMS and provider != "ollama":
        return (
            f"the {arm} arm drives Ollama's native tools API; {provider!r} is not "
            "supported yet"
        )
    return None


async def _run(args: argparse.Namespace) -> int:
    cfg = _apply_arm(get_config(), args.arm)
    tasks = load_tasks(args.tasks_dir)
    if not tasks:
        print(f"No tasks found under {args.tasks_dir}", file=sys.stderr)
        return 1

    target = ModelRouter(cfg).route(ModelTier.SMART)
    unsupported = _arm_unsupported_reason(args.arm, target.provider)
    if unsupported is not None:
        print(f"Cannot run: {unsupported}.", file=sys.stderr)
        return 2
    local_models: set[str] = set()
    if target.provider == "ollama":
        try:
            local_models = set(await OllamaClient(cfg).list_models())
        except Exception:  # daemon down: nothing is provably local, so refuse under --require-local
            local_models = set()
    locality = classify_locality(target.provider, target.model, local_models)
    refusal = _locality_refusal(locality, require_local=args.require_local)
    if refusal is not None:
        print(f"Cannot run: {refusal}.", file=sys.stderr)
        return 2
    total_runs = sum(len(t.variants) for t in tasks)
    print(
        f"Pinned: {target.provider} / {target.model} ({args.arm} arm, {locality})  |  "
        f"{len(tasks)} tasks x{args.repeat}, "
        f"{total_runs * args.repeat} live model calls (at least)"
    )
    if not args.yes:
        confirm = (await asyncio.to_thread(input, "Proceed? [y/N] ")).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0

    registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
    tier1_loader = Tier1Loader(_tier1_root())

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
            result = await run_task(
                cfg,
                registry,
                tier1_loader,
                task,
                repeat=args.repeat,
                run_fn=_variant_runner(args.arm),
            )
            task_results.append(result)

    report = EvalReport(
        provider=target.provider,
        model=target.model,
        started_at=started_at,
        tasks=tuple(task_results),
        arm=args.arm,
        locality=locality,
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
