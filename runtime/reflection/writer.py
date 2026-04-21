"""Append-only Markdown writer for `PATTERNS.md` and `PROPOSALS.md`.

Both files live under `<workspace>/reflection/`, **outside canon**.
Every reflection run appends a dated section; we never rewrite history.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .event_reader import ReadStats
from .patterns import PatternRecord
from .proposals import Proposal

_PATTERNS_FILENAME = "PATTERNS.md"
_PROPOSALS_FILENAME = "PROPOSALS.md"


def reflection_dir(workspace: Path) -> Path:
    out = Path(workspace) / "reflection"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_patterns(
    workspace: Path,
    patterns: Iterable[PatternRecord],
    stats: ReadStats,
    *,
    when: datetime | None = None,
) -> Path:
    path = reflection_dir(workspace) / _PATTERNS_FILENAME
    header = _header(when, stats)
    body = "\n".join(_render_pattern(p) for p in patterns) or "_No patterns detected._"
    _append(path, f"{header}\n\n{body}\n")
    return path


def write_proposals(
    workspace: Path,
    proposals: Iterable[Proposal],
    stats: ReadStats,
    *,
    when: datetime | None = None,
) -> Path:
    path = reflection_dir(workspace) / _PROPOSALS_FILENAME
    header = _header(when, stats)
    body = "\n".join(_render_proposal(p) for p in proposals) or "_No proposals drafted._"
    _append(path, f"{header}\n\n{body}\n")
    return path


def _header(when: datetime | None, stats: ReadStats) -> str:
    ts = (when or datetime.now(tz=UTC)).strftime("%Y-%m-%dT%H:%MZ")
    return (
        f"## {ts} — sessions={stats.sessions}, events={stats.events}, "
        f"skipped={stats.skipped}"
    )


def _render_pattern(p: PatternRecord) -> str:
    samples = ", ".join(p.sample_session_ids) or "—"
    return (
        f"- **{p.detector}** ({p.severity}, count={p.count}) — {p.summary}\n"
        f"  - sessions: {samples}"
    )


def _render_proposal(p: Proposal) -> str:
    affected = ", ".join(p.affected) or "—"
    rationale = p.rationale.strip() or "—"
    return (
        f"### {p.id} — {p.pattern_detector} (risk: {p.risk})\n"
        f"- **Affected:** {affected}\n"
        f"- **Change:** {p.change}\n"
        f"- **Rationale:** {rationale}"
    )


def _append(path: Path, text: str) -> None:
    sep = os.linesep
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text + sep)
