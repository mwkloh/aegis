"""Coding harness driver — drafts only, never applies.

```text
python -m runtime.coding_harness.cli              # draft for every eligible CT
python -m runtime.coding_harness.cli --list       # show eligible CTs (no LLM calls)
python -m runtime.coding_harness.cli --task CT-001
python -m runtime.coding_harness.cli --force      # redraft even if a patch exists
python -m runtime.coding_harness.cli --with-context  # gather + critique-then-revise
```

Eligibility = `CodingTask` whose latest `Decision.verdict == "approve"`.
For each eligible CT we ask the coding model for one `Draft` and write a
timestamped `.patch.md` to `<workspace>/coding_harness/diffs/`. Existing
patches are never overwritten — `--force` writes a new file alongside.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from runtime.config import AegisConfig, get_config
from runtime.events import EventStream, EventType
from runtime.improvement.coding_tasks import CodingTask, load_tasks
from runtime.improvement.decisions import Decision, latest_by_imp, load_decisions
from runtime.llm.clients import (
    InstrumentedModelClient,
    ModelClient,
    OllamaClient,
    OpenRouterClient,
)
from runtime.llm.clients.openrouter_client import OpenRouterConfigError
from runtime.skills import SkillRegistry

from .coder import draft_for
from .context import ContextBundle, gather_context
from .critic import critique_then_revise
from .patch_writer import existing_drafts_for, write_patch

CATALOG_DIR = Path(__file__).parent.parent / "skills" / "catalog"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = get_config()
    return asyncio.run(_run(cfg, args))


async def _run(cfg: AegisConfig, args: argparse.Namespace) -> int:
    tasks = load_tasks(cfg.aegis_home)
    decisions = load_decisions(cfg.aegis_home)
    eligible = _eligible(tasks, decisions)

    if args.list:
        return _do_list(eligible, len(tasks))

    if args.task:
        target = next((t for t in eligible if t.ct_id == args.task), None)
        if target is None:
            print(
                f"[harness] error: {args.task!r} not eligible (unknown or not approved)",
                file=sys.stderr,
            )
            return 1
        eligible = [target]

    events = EventStream(cfg.storage.sessions_dir)
    client, label = _build_coding_client(cfg, events)
    skills = sorted(s.id for s in SkillRegistry.from_directory(CATALOG_DIR).all())
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    mode = "context-mode ON" if args.with_context else "context-mode OFF"
    print(f"[harness] {len(eligible)} eligible CT(s)  ({label})  [{mode}]")

    written = 0
    skipped = 0
    for task in eligible:
        prior = existing_drafts_for(cfg.aegis_home, task.ct_id)
        if prior and not args.force:
            print(f"[harness] {task.ct_id} already drafted ({prior[-1].name}) — skip")
            skipped += 1
            continue
        bundle = _maybe_gather(task, repo_root, args, events)
        draft = await draft_for(
            task,
            client=client,
            model=cfg.models.coding,
            available_skills=skills,
            events=events,
            context=bundle,
        )
        if args.with_context and client is not None and draft.status == "ok":
            revised = await critique_then_revise(
                draft, task, bundle,
                client=client, model=cfg.models.coding, events=events,
            )
            arrow = "draft → critique → revise" if revised is not draft else "draft → critique"
            print(f"[harness] {task.ct_id} {arrow}")
            draft = revised
        path = write_patch(cfg.aegis_home, draft)
        print(f"[harness] {task.ct_id} → {draft.status:<8} {path.name}")
        written += 1

    print(f"[harness] done  written={written}  skipped={skipped}")
    return 0


def _maybe_gather(
    task: CodingTask,
    repo_root: Path,
    args: argparse.Namespace,
    events: EventStream,
) -> ContextBundle | None:
    if not args.with_context:
        return None
    bundle = gather_context(repo_root, list(task.scope))
    kb_files = sum(len(f.content.encode("utf-8")) for f in bundle.files) / 1024
    kb_skills = sum(len(s.content.encode("utf-8")) for s in bundle.skills) / 1024
    print(
        f"[harness] {task.ct_id} {task.imp_id} — context-mode ON"
    )
    print(
        f"[harness] gathered {len(bundle.files)} in-scope file(s) "
        f"({kb_files:.1f} KB) + {len(bundle.skills)} skill YAML(s) "
        f"({kb_skills:.1f} KB)"
    )
    events.append(
        EventType.PATTERN_OBSERVED,
        {
            "pattern": "harness_with_context",
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "files": len(bundle.files),
            "skills": len(bundle.skills),
            "total_bytes": bundle.total_bytes,
            "truncated": bundle.truncated,
        },
    )
    return bundle


def _do_list(eligible: list[CodingTask], total: int) -> int:
    print(f"[harness] {len(eligible)} eligible of {total} queued")
    for task in eligible:
        scope = ", ".join(task.scope) or "—"
        print(f"  {task.ct_id} → {task.imp_id}  scope: {scope}")
    return 0


def _eligible(
    tasks: list[CodingTask], decisions: list[Decision]
) -> list[CodingTask]:
    """Return tasks whose latest decision is `approve` (latest-wins)."""
    latest = latest_by_imp(decisions)
    return [t for t in tasks if (d := latest.get(t.imp_id)) and d.verdict == "approve"]


def _build_coding_client(
    cfg: AegisConfig, events: EventStream
) -> tuple[ModelClient | None, str]:
    """Prefer OpenRouter (matches `cfg.models.coding`); fall back to None → stubs."""
    try:
        raw = OpenRouterClient(cfg)
    except OpenRouterConfigError:
        events.append(
            EventType.PATTERN_OBSERVED,
            {"pattern": "coding_client_unavailable", "reason": "openrouter_key_missing"},
        )
        return _try_ollama(cfg, events)
    wrapped = InstrumentedModelClient(
        inner=raw, events=events, tier="coding", provider="openrouter"
    )
    return wrapped, f"coding={cfg.models.coding} via openrouter"


def _try_ollama(
    cfg: AegisConfig, events: EventStream
) -> tuple[ModelClient | None, str]:
    """Fallback: use Ollama if it's reachable and the model is local."""
    try:
        raw = OllamaClient(cfg)
    except (ValueError, RuntimeError) as exc:
        events.append(
            EventType.PATTERN_OBSERVED,
            {"pattern": "coding_client_unavailable", "reason": str(exc)},
        )
        return None, "no client (will stub)"
    wrapped = InstrumentedModelClient(
        inner=raw, events=events, tier="coding", provider="ollama"
    )
    return wrapped, f"coding={cfg.models.coding} via ollama"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="coding-harness")
    p.add_argument(
        "--list",
        action="store_true",
        help="Print eligible CTs and exit (no LLM calls, no files written).",
    )
    p.add_argument(
        "--task",
        metavar="CT-NNN",
        default=None,
        help="Draft for a single CT id. Exits 1 if unknown or not approved.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redraft even if a prior patch exists (writes a new timestamped file).",
    )
    p.add_argument(
        "--with-context",
        action="store_true",
        help=(
            "Gather skill-aware in-scope context for each draft and run a"
            " bounded critique-then-revise pass. Default OFF."
        ),
    )
    p.add_argument(
        "--repo-root",
        metavar="PATH",
        default=None,
        help="Repo root used to resolve scope paths (defaults to cwd).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
