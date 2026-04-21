"""One-shot reflection driver.

`python -m runtime.reflection.cli [--since YYYY-MM-DD] [--dry-run] [--quiet]`

Reads events from `<workspace>/sessions/`, runs deterministic detectors,
optionally drafts proposals via the Reflection LLM, and appends both to
`<workspace>/reflection/{PATTERNS,PROPOSALS}.md`. Exit 0 on success
(including "no patterns found"), 1 only on hard config / I/O failure.

``--quiet`` routes progress lines to stderr instead of stdout — used by
the scheduled ``SYS-reflection-sweep`` job so an empty stdout satisfies
the silent-success contract (see ``docs/PLAN_PHASE_10_TRACK_D_RECURRING_JOBS.md``).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

from runtime.config import AegisConfig, get_config
from runtime.events import EventStream, EventType
from runtime.model_router.clients import (
    InstrumentedModelClient,
    ModelClient,
    OllamaClient,
)
from runtime.skills import SkillRegistry

from .event_reader import read_window
from .patterns import detect_all
from .proposals import draft
from .writer import write_patterns, write_proposals

CATALOG_DIR = Path(__file__).parent.parent / "skills" / "catalog"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = get_config()
    return asyncio.run(
        _run(cfg, since=args.since, dry_run=args.dry_run, quiet=args.quiet)
    )


async def _run(
    cfg: AegisConfig, *, since: date | None, dry_run: bool, quiet: bool = False
) -> int:
    out = sys.stderr if quiet else sys.stdout
    workspace = cfg.aegis_home
    sessions_dir = cfg.storage.sessions_dir
    events_log = EventStream(sessions_dir)

    events, stats = read_window(sessions_dir, since=since)
    print(
        f"[reflection] window={'today' if since is None else since.isoformat()}  "
        f"sessions={stats.sessions}  events={stats.events}  skipped={stats.skipped}",
        file=out,
    )

    patterns = detect_all(events)
    print(f"[reflection] patterns: {len(patterns)} detected", file=out)

    client, label = _build_reflection_client(cfg, events_log)
    skills = sorted(s.id for s in SkillRegistry.from_directory(CATALOG_DIR).all())
    proposals = await draft(
        patterns,
        client=client,
        model=cfg.models.reflection,
        available_skills=skills,
        events=events_log,
    )
    print(f"[reflection] proposals: {len(proposals)} drafted  ({label})", file=out)

    if dry_run:
        print("[reflection] dry-run: no files written", file=out)
        return 0

    p_path = write_patterns(workspace, patterns, stats)
    pr_path = write_proposals(workspace, proposals, stats)
    print(f"[reflection] wrote {p_path}", file=out)
    print(f"[reflection] wrote {pr_path}", file=out)
    return 0


def _build_reflection_client(
    cfg: AegisConfig, events: EventStream
) -> tuple[ModelClient | None, str]:
    """Return a wired Ollama client for the reflection model — or None."""
    try:
        raw = OllamaClient(cfg)
    except (ValueError, RuntimeError) as exc:
        events.append(
            EventType.PATTERN_OBSERVED,
            {"pattern": "reflection_client_unavailable", "reason": str(exc)},
        )
        return None, "no client"
    wrapped = InstrumentedModelClient(
        inner=raw, events=events, tier="reflection", provider="ollama"
    )
    return wrapped, f"reflection={cfg.models.reflection}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reflection")
    p.add_argument(
        "--since",
        type=_iso_date,
        default=None,
        help="Earliest day to include (YYYY-MM-DD). Default: today.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print, but do not write PATTERNS/PROPOSALS files.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Route progress lines to stderr instead of stdout (silent-success for scheduler).",
    )
    return p.parse_args(argv)


def _iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
