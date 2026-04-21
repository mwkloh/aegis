"""Append-only `DECISIONS.md` log for the Improvement plane.

Every human verdict (approve / reject / defer) gets one dated section.
Re-deciding the same `IMP-id` writes a new section whose `supersedes`
points at the prior `decided_at`. Earlier sections are *kept* as
history — never overwritten.

Each recorded decision also emits a `governance.decision` event via
`EventStream` so Plane 1's audit trail mirrors Plane 3's log file.
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from runtime.events import EventStream, EventType

HumanVerdict = Literal["approve", "reject", "defer"]
ApplierVerdict = Literal[
    "applied_clean", "applied_test_failed", "apply_conflict", "reverted",
]
Verdict = Literal[
    # Phase 3 — human review verdicts
    "approve",
    "reject",
    "defer",
    # Phase 5 — applier verdicts (recorded after `make apply`)
    "applied_clean",
    "applied_test_failed",
    "apply_conflict",
    "reverted",
]

_VERDICT_PATTERN = (
    "approve|reject|defer|applied_clean|applied_test_failed|apply_conflict|reverted"
)

_DECISIONS_FILENAME = "DECISIONS.md"
_HEADER = "# AEGIS — Improvement Decisions (Plane 3 log)\n\nAppend-only.\n"

_SECTION_RE = re.compile(
    rf"^## (?P<ts>\S+Z) — (?P<imp>IMP-[0-9a-f]{{8}}) — (?P<verdict>{_VERDICT_PATTERN})\s*$"
)
_RATIONALE_RE = re.compile(r"^- \*\*Rationale:\*\* (?P<value>.*)$")
_SUPERSEDES_RE = re.compile(r"^- \*\*Supersedes:\*\* (?P<value>.*)$")
_MAX_RATIONALE = 1024


class Decision(BaseModel):
    """One verdict against an `IMP-id`. Frozen, append-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    imp_id: str = Field(min_length=12, max_length=12)
    verdict: Verdict
    rationale: str = Field(min_length=0, max_length=_MAX_RATIONALE)
    decided_at: datetime
    supersedes: str | None = Field(default=None, max_length=32)


def improvement_dir(workspace: Path) -> Path:
    out = Path(workspace) / "improvement"
    out.mkdir(parents=True, exist_ok=True)
    return out


def decisions_path(workspace: Path) -> Path:
    return improvement_dir(workspace) / _DECISIONS_FILENAME


def load_decisions(workspace: Path) -> list[Decision]:
    """Return every decision row in file order (oldest first)."""
    path = decisions_path(workspace)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")

    out: list[Decision] = []
    pending_header: re.Match[str] | None = None
    rationale = ""
    supersedes: str | None = None

    def flush() -> None:
        nonlocal pending_header, rationale, supersedes
        if pending_header is None:
            return
        ts = pending_header.group("ts")
        try:
            decided_at = datetime.strptime(ts, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        except ValueError:
            pending_header = None
            rationale = ""
            supersedes = None
            return
        verdict_raw = pending_header.group("verdict")
        verdict: Verdict = verdict_raw  # type: ignore[assignment]
        out.append(
            Decision(
                imp_id=pending_header.group("imp"),
                verdict=verdict,
                rationale=rationale.strip(),
                decided_at=decided_at,
                supersedes=supersedes,
            )
        )
        pending_header = None
        rationale = ""
        supersedes = None

    for raw in text.splitlines():
        line = raw.rstrip()
        section = _SECTION_RE.match(line)
        if section:
            flush()
            pending_header = section
            rationale = ""
            supersedes = None
            continue
        if pending_header is None:
            continue
        rat = _RATIONALE_RE.match(line)
        if rat:
            rationale = rat.group("value")
            continue
        sup = _SUPERSEDES_RE.match(line)
        if sup:
            value = sup.group("value").strip()
            supersedes = None if value in ("", "—") else value

    flush()
    return out


def latest_by_imp(decisions: list[Decision]) -> dict[str, Decision]:
    """Map `imp_id` → most recent decision (last write wins)."""
    out: dict[str, Decision] = {}
    for d in decisions:
        out[d.imp_id] = d
    return out


def record_decision(
    workspace: Path,
    *,
    imp_id: str,
    verdict: Verdict,
    rationale: str,
    events: EventStream | None = None,
    when: datetime | None = None,
) -> Decision:
    """Append one decision section. Returns the validated `Decision`."""
    decided_at = when or datetime.now(tz=UTC)
    prior = latest_by_imp(load_decisions(workspace))
    supersedes = (
        _format_ts(prior[imp_id].decided_at) if imp_id in prior else None
    )
    decision = Decision(
        imp_id=imp_id,
        verdict=verdict,
        rationale=(rationale or "").strip()[:_MAX_RATIONALE],
        decided_at=decided_at,
        supersedes=supersedes,
    )
    _append(decisions_path(workspace), _render(decision))
    if events is not None:
        events.append(
            EventType.GOVERNANCE_DECISION,
            {
                "imp_id": decision.imp_id,
                "verdict": decision.verdict,
                "supersedes": decision.supersedes,
            },
        )
    return decision


def _format_ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _render(d: Decision) -> str:
    sup = d.supersedes or "—"
    rationale = d.rationale or "—"
    return (
        f"## {_format_ts(d.decided_at)} — {d.imp_id} — {d.verdict}\n"
        f"- **Rationale:** {rationale}\n"
        f"- **Supersedes:** {sup}\n"
    )


def _append(path: Path, body: str) -> None:
    sep = os.linesep
    needs_header = not path.is_file()
    with path.open("a", encoding="utf-8") as fh:
        if needs_header:
            fh.write(_HEADER + sep)
        fh.write(body + sep)
