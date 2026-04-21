"""Tool-invocation record — audit trail for skill-scoped tool calls.

Phase 8 §D1. Every time the tool harness returns a verdict (§C3), the
caller records it here. The record is idempotent on
``(session_id, imp_id, skill, tool, argv_hash)`` so a retried harness
call or a chat-pipeline replay cannot double-log.

Records are written as ``EventType.TOOL_INVOKED`` events on the
Plane-1 session shard. The shard itself is the source of truth — we
don't maintain a parallel file — so operators and reviewers can see
the audit trail alongside decisions, harness lifecycle, and chat
telemetry.

Design:

* **No stdout bodies.** Payload carries ``outcome_bytes`` and
  ``verdict`` only. The harness already clips stdout to 32 KB in
  :mod:`runtime.tools.harness`; the record plane stays structural to
  match §3.3/§3.6 in the Telegram plan.
* **Idempotency via scan.** ``record_tool_call`` reads the session
  shard once per call (tiny, per-day file) to check for an existing
  composite-key match. Mirrors ``record_decision``'s "read then
  append" discipline rather than maintaining a parallel in-memory set
  that could drift after a restart.
* **Stable argv hashing.** :func:`compute_argv_hash` joins argv with
  a NUL delimiter (forbidden inside argv tokens per §C3) and returns
  the first 16 hex chars of SHA-256. Short enough for logs, wide
  enough to avoid realistic collisions.
* **Never raises.** A malformed shard line is skipped; a missing
  shard just yields an empty history. Emission failures propagate
  only if ``EventStream.append`` itself raises (disk full, perms) —
  the caller decides whether that blocks the turn.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from runtime.events import EventStream, EventType
from runtime.tools.harness import ToolVerdict

ToolOutcome = Literal["ok", "skipped_idempotent"]


@dataclass(frozen=True)
class ToolCallRecord:
    """One row from the session's tool-invocation audit trail."""

    session_id: str
    imp_id: str
    skill: str
    tool: str
    argv_hash: str
    verdict: ToolVerdict
    outcome_bytes: int
    recorded_at: datetime

    def composite_key(self) -> tuple[str, str, str, str, str]:
        return (self.session_id, self.imp_id, self.skill, self.tool, self.argv_hash)


def compute_argv_hash(argv: Iterable[str]) -> str:
    """SHA-256(NUL-joined argv), first 16 hex chars.

    Stable across runs: identical argv tokens always yield the same
    hash regardless of list identity, ordering outside the sequence,
    or process boundaries. NUL is safe as a delimiter because the
    harness rejects any argv token containing NUL.
    """
    joined = "\x00".join(argv)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def load_tool_calls(events: EventStream) -> list[ToolCallRecord]:
    """Return every ``tool.invoked`` record on the session's shard.

    Shard is read once per call. Lines that don't parse as JSON, don't
    match the expected payload shape, or have an unknown verdict are
    silently skipped — an operator hand-editing the shard shouldn't
    crash the audit reader.
    """
    path = events.path
    if not path.is_file():
        return []
    records: list[ToolCallRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != EventType.TOOL_INVOKED.value:
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        record = _payload_to_record(
            session_id=str(entry.get("session_id", "")),
            recorded_at_raw=entry.get("ts"),
            payload=payload,
        )
        if record is not None:
            records.append(record)
    return records


def record_tool_call(
    events: EventStream,
    *,
    imp_id: str,
    skill: str,
    tool: str,
    argv_hash: str,
    verdict: ToolVerdict,
    outcome_bytes: int,
    when: datetime | None = None,
) -> ToolOutcome:
    """Emit ``tool.invoked`` iff no prior event matches the composite key.

    Returns ``"ok"`` if a new event was written, ``"skipped_idempotent"``
    if a record for this ``(session, imp_id, skill, tool, argv_hash)``
    already exists on the shard. Never raises except on a genuine
    disk / perms failure inside ``EventStream.append`` — we do not
    mask those.
    """
    if outcome_bytes < 0:
        raise ValueError("outcome_bytes must be >= 0")
    if not imp_id or not skill or not tool or not argv_hash:
        raise ValueError("imp_id, skill, tool, argv_hash must all be non-empty")

    session_id = events.session_id
    existing_keys = {r.composite_key() for r in load_tool_calls(events)}
    composite = (session_id, imp_id, skill, tool, argv_hash)
    if composite in existing_keys:
        return "skipped_idempotent"

    payload = {
        "imp_id": imp_id,
        "skill": skill,
        "tool": tool,
        "argv_hash": argv_hash,
        "verdict": verdict,
        "outcome_bytes": int(outcome_bytes),
    }
    # ``when`` is accepted for test determinism but we intentionally do
    # NOT thread it into EventStream.append — the stream is the clock.
    # Exposing `when` just keeps the signature symmetric with
    # ``record_decision`` for future operators.
    _ = when
    events.append(EventType.TOOL_INVOKED, payload)
    return "ok"


_VALID_VERDICTS: frozenset[str] = frozenset(
    {
        "verified",
        "argv_rejected",
        "exit_nonzero",
        "timeout",
        "schema_violation",
        "host_denied",
    }
)


def _payload_to_record(
    *,
    session_id: str,
    recorded_at_raw: object,
    payload: dict[str, object],
) -> ToolCallRecord | None:
    """Coerce one shard entry into a ``ToolCallRecord`` — or None if malformed.

    We accept both the rich payload written by :func:`record_tool_call`
    and the thin legacy payload (``{"tool": ...}``) written by the
    Phase-1 CLI. Thin payloads are skipped because they carry no
    composite key — they never contributed to the audit trail and
    they cannot match an idempotency check.
    """
    required = ("imp_id", "skill", "tool", "argv_hash", "verdict", "outcome_bytes")
    if not all(k in payload for k in required):
        return None
    verdict_raw = str(payload["verdict"])
    if verdict_raw not in _VALID_VERDICTS:
        return None
    recorded_at = _parse_ts(recorded_at_raw)
    raw_bytes = payload["outcome_bytes"]
    if not isinstance(raw_bytes, (int, str, float)):
        return None
    try:
        outcome_bytes = int(raw_bytes)
    except (TypeError, ValueError):
        return None
    return ToolCallRecord(
        session_id=session_id,
        imp_id=str(payload["imp_id"]),
        skill=str(payload["skill"]),
        tool=str(payload["tool"]),
        argv_hash=str(payload["argv_hash"]),
        verdict=verdict_raw,  # type: ignore[arg-type]
        outcome_bytes=outcome_bytes,
        recorded_at=recorded_at,
    )


def _parse_ts(raw: object) -> datetime:
    """Parse ``ts`` from the shard. Falls back to epoch on failure."""
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=UTC)


__all__ = [
    "ToolCallRecord",
    "ToolOutcome",
    "compute_argv_hash",
    "load_tool_calls",
    "record_tool_call",
]
