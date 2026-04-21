"""Append-only `CODING_TASKS.md` queue for the Improvement plane.

Approval of a proposal materializes one `CT-NNN` row, monotonically
numbered. Re-approving the same `IMP-id` is a no-op (returns `None`).
This file is consumed by Phase 4 (the coding harness) — we never
generate diffs here.
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .decisions import improvement_dir
from .proposal_loader import LoadedProposal

_TASKS_FILENAME = "CODING_TASKS.md"
_HEADER = "# AEGIS — Improvement Coding Tasks (Plane 3 queue)\n\nAppend-only.\n"
_HEADER_RE = re.compile(
    r"^## (?P<ct>CT-\d+) — (?P<imp>IMP-[0-9a-f]{8})\s*$"
)
_FIELD_RE = re.compile(
    r"^- \*\*(?P<key>Queued|Scope|Constraints|Expected output):\*\* (?P<value>.*)$"
)

_DEFAULT_CONSTRAINTS = "do not modify canon; produce diffs only"
_MAX_SCOPE = 8
_MAX_FIELD = 1024


class CodingTask(BaseModel):
    """One queued task linked back to its source proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ct_id: str = Field(min_length=4, max_length=16)
    imp_id: str = Field(min_length=12, max_length=12)
    scope: list[str] = Field(default_factory=list, max_length=_MAX_SCOPE)
    constraints: str = Field(min_length=1, max_length=_MAX_FIELD)
    expected_output: str = Field(min_length=1, max_length=_MAX_FIELD)
    queued_at: datetime


def tasks_path(workspace: Path) -> Path:
    return improvement_dir(workspace) / _TASKS_FILENAME


def load_tasks(workspace: Path) -> list[CodingTask]:
    """Return every queued CT in file order."""
    path = tasks_path(workspace)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")

    out: list[CodingTask] = []
    pending_header: re.Match[str] | None = None
    fields: dict[str, str] = {}

    def flush() -> None:
        nonlocal pending_header, fields
        if pending_header is None:
            return
        queued_raw = fields.get("Queued", "").strip()
        try:
            queued_at = datetime.strptime(queued_raw, "%Y-%m-%dT%H:%MZ").replace(
                tzinfo=UTC
            )
        except ValueError:
            pending_header = None
            fields = {}
            return
        scope_raw = fields.get("Scope", "").strip()
        scope = (
            []
            if not scope_raw or scope_raw == "—"
            else [s.strip() for s in scope_raw.split(",") if s.strip()]
        )
        out.append(
            CodingTask(
                ct_id=pending_header.group("ct"),
                imp_id=pending_header.group("imp"),
                scope=scope,
                constraints=fields.get("Constraints", _DEFAULT_CONSTRAINTS).strip()
                or _DEFAULT_CONSTRAINTS,
                expected_output=fields.get("Expected output", "").strip() or "—",
                queued_at=queued_at,
            )
        )
        pending_header = None
        fields = {}

    for raw in text.splitlines():
        line = raw.rstrip()
        header = _HEADER_RE.match(line)
        if header:
            flush()
            pending_header = header
            fields = {}
            continue
        if pending_header is None:
            continue
        field = _FIELD_RE.match(line)
        if field:
            fields[field.group("key")] = field.group("value")

    flush()
    return out


def queue_task(
    workspace: Path,
    proposal: LoadedProposal,
    *,
    when: datetime | None = None,
) -> CodingTask | None:
    """Append a new `CT-NNN` for `proposal` if none exists. Otherwise return None."""
    existing = load_tasks(workspace)
    if any(task.imp_id == proposal.imp_id for task in existing):
        return None
    next_n = _next_ct_number(existing)
    task = CodingTask(
        ct_id=f"CT-{next_n:03d}",
        imp_id=proposal.imp_id,
        scope=list(proposal.affected),
        constraints=_DEFAULT_CONSTRAINTS,
        expected_output=proposal.change,
        queued_at=when or datetime.now(tz=UTC),
    )
    _append(tasks_path(workspace), _render(task))
    return task


def _next_ct_number(existing: list[CodingTask]) -> int:
    highest = 0
    for t in existing:
        try:
            n = int(t.ct_id.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        highest = max(highest, n)
    return highest + 1


def _render(t: CodingTask) -> str:
    scope = ", ".join(t.scope) or "—"
    return (
        f"## {t.ct_id} — {t.imp_id}\n"
        f"- **Queued:** {t.queued_at.astimezone(UTC).strftime('%Y-%m-%dT%H:%MZ')}\n"
        f"- **Scope:** {scope}\n"
        f"- **Constraints:** {t.constraints}\n"
        f"- **Expected output:** {t.expected_output}\n"
    )


def _append(path: Path, body: str) -> None:
    sep = os.linesep
    needs_header = not path.is_file()
    with path.open("a", encoding="utf-8") as fh:
        if needs_header:
            fh.write(_HEADER + sep)
        fh.write(body + sep)
