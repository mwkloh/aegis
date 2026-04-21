"""Parse `<workspace>/reflection/PROPOSALS.md` into typed records.

The Reflection writer (`runtime/reflection/writer.py`) emits one
`### P-NNN — <detector> (risk: ...)` block per proposal under a
`## YYYY-MM-DDTHH:MMZ — sessions=...` run header. This loader walks
those blocks, validates each into a `LoadedProposal`, and assigns a
content-hash `IMP-<8 hex>` so the same logical proposal across runs
maps to the same review item.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Risk = Literal["low", "medium", "high"]

_HEADER_RE = re.compile(r"^## (?P<run>\S+Z) —")
_PROPOSAL_RE = re.compile(
    r"^### (?P<pid>P-\d+) — (?P<detector>\S+) \(risk: (?P<risk>low|medium|high)\)\s*$"
)
_FIELD_RE = re.compile(r"^- \*\*(?P<key>Affected|Change|Rationale):\*\* (?P<value>.*)$")

_MAX_AFFECTED = 8
_MAX_CHANGE = 512
_MAX_RATIONALE = 2048


class LoadedProposal(BaseModel):
    """One proposal loaded from `PROPOSALS.md`. `imp_id` is content-stable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    imp_id: str = Field(min_length=12, max_length=12)
    pattern_detector: str = Field(min_length=1, max_length=64)
    affected: list[str] = Field(default_factory=list, max_length=_MAX_AFFECTED)
    change: str = Field(min_length=1, max_length=_MAX_CHANGE)
    risk: Risk
    rationale: str = Field(min_length=0, max_length=_MAX_RATIONALE)
    source_run: str = Field(min_length=1, max_length=32)


def proposals_path(workspace: Path) -> Path:
    """Canonical path to the file produced by the Reflection writer."""
    return Path(workspace) / "reflection" / "PROPOSALS.md"


class _ParseState:
    """Mutable per-block state for the line-oriented PROPOSALS.md parser."""

    def __init__(self) -> None:
        self.current_run: str | None = None
        self.pending: dict[str, str] | None = None
        self.detector: str | None = None
        self.risk: Risk | None = None

    def reset_block(self) -> None:
        self.pending = None
        self.detector = None
        self.risk = None

    def open_block(self, detector: str, risk: Risk | None) -> None:
        self.detector = detector
        self.risk = risk
        self.pending = {}

    def flush(self, by_id: dict[str, LoadedProposal]) -> None:
        if (
            self.pending is None
            or self.detector is None
            or self.risk is None
            or self.current_run is None
        ):
            self.reset_block()
            return
        change = self.pending.get("Change", "").strip()
        if not change:
            self.reset_block()
            return
        affected = _split_affected(self.pending.get("Affected", ""))
        rationale = self.pending.get("Rationale", "").strip()
        proposal = _build(
            self.detector, affected, change, self.risk, rationale, self.current_run
        )
        by_id[proposal.imp_id] = proposal
        self.reset_block()


def load_proposals(workspace: Path) -> list[LoadedProposal]:
    """Return the latest occurrence of each unique `imp_id` (content-hashed)."""
    path = proposals_path(workspace)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")

    by_id: dict[str, LoadedProposal] = {}
    state = _ParseState()
    for raw in text.splitlines():
        line = raw.rstrip()
        header = _HEADER_RE.match(line)
        if header:
            state.flush(by_id)
            state.current_run = header.group("run")
            continue
        prop = _PROPOSAL_RE.match(line)
        if prop:
            state.flush(by_id)
            risk_raw = prop.group("risk")
            risk: Risk | None = (
                risk_raw if risk_raw in ("low", "medium", "high") else None  # type: ignore[assignment]
            )
            state.open_block(prop.group("detector"), risk)
            continue
        if state.pending is None:
            continue
        field = _FIELD_RE.match(line)
        if field:
            state.pending[field.group("key")] = field.group("value")

    state.flush(by_id)
    return list(by_id.values())


def derive_imp_id(detector: str, affected: list[str], change: str) -> str:
    """`IMP-<8 hex>` derived from `(detector, sorted(affected), change)`."""
    payload = f"{detector}|{','.join(sorted(affected))}|{change}".encode()
    return f"IMP-{hashlib.sha256(payload).hexdigest()[:8]}"


def _split_affected(value: str) -> list[str]:
    text = value.strip()
    if not text or text == "—":
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _build(
    detector: str,
    affected: list[str],
    change: str,
    risk: Risk,
    rationale: str,
    source_run: str,
) -> LoadedProposal:
    imp_id = derive_imp_id(detector, affected, change)
    return LoadedProposal(
        imp_id=imp_id,
        pattern_detector=detector,
        affected=affected,
        change=change,
        risk=risk,
        rationale=rationale,
        source_run=source_run,
    )
