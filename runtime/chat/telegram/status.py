"""Phase 7 §4.1 — `/status` activity scan.

Reads the last 24 hours of append-only session JSONL files written
by `runtime/events/EventStream` and rolls them up into a small
frozen record the formatter can print.

Design:

* **Scan two date shards.** Because event logs shard by UTC date
  (`<sessions_dir>/YYYY-MM-DD/<session_id>.jsonl`), a 24h window
  starting `now - 24h` can span two shards. We read today's and
  yesterday's directories only — never the full tree.
* **Never raises.** A malformed JSON line is skipped (the file is
  append-only and can be torn if the process dies mid-write). A
  missing shard directory is treated as empty. Any unexpected
  exception ends the scan and returns whatever was counted so far
  — consistent with Phase 7's stub-on-failure posture (§2.8).
* **GOVERNANCE_DECISION splits.** The Improvement plane emits one
  event per verdict — human (`approve`/`reject`/`defer`) and
  applier (`applied_clean`/`applied_test_failed`/...). `/status`
  reports "decisions" (human) and "applies" (applier) separately
  so the operator can see at a glance which half of the loop is
  active.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from runtime.config import AegisConfig
from runtime.events import EventType
from runtime.model_router.router import ModelRouter, ModelTier

_HUMAN_VERDICTS = frozenset({"approve", "reject", "defer"})
_APPLIER_VERDICTS = frozenset(
    {"applied_clean", "applied_test_failed", "apply_conflict", "reverted"}
)


class StatusSnapshot(BaseModel):
    """Activity summary for the last 24h window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_start: datetime
    window_end: datetime
    sessions: int = Field(ge=0)
    patterns: int = Field(ge=0)
    decisions: int = Field(ge=0)
    applies: int = Field(ge=0)


class SystemInfo(BaseModel):
    """Snapshot of the currently-routed model stack + runtime posture.

    Pure data — `render_system_info` formats it and the handler prepends
    the block to the 24h activity rollup. Computed on demand, never
    cached: `prefer_local` + Ollama liveness can flip between `/status`
    calls and operators expect the answer to reflect reality *now*.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    smart_model: str
    smart_provider: str
    smart_degraded: bool
    smart_local_model: str
    fast_model: str
    reflection_model: str
    coding_model: str
    prefer_local: bool
    local_ready: bool
    ollama_base_url: str
    openrouter_configured: bool
    vault_enabled: bool
    vault_root: str | None
    vault_sources: int
    allowlist_size: int


def collect_system_info(cfg: AegisConfig, *, router: ModelRouter) -> SystemInfo:
    """Resolve the current routing decisions + runtime config.

    Uses the shared 30s `local_ready` probe cache on the router so we
    don't hammer the loopback port when `/status` is fired repeatedly.
    """
    smart_target = router.route(ModelTier.SMART)
    vault_root = (
        str(cfg.vault_indexing.vault_root)
        if cfg.vault_indexing.vault_root is not None
        else None
    )
    return SystemInfo(
        smart_model=smart_target.model,
        smart_provider=smart_target.provider,
        smart_degraded=smart_target.degraded,
        smart_local_model=cfg.models.smart_local,
        fast_model=cfg.models.fast,
        reflection_model=cfg.models.reflection,
        coding_model=cfg.models.coding,
        prefer_local=cfg.models.prefer_local,
        local_ready=router.is_local_ready(),
        ollama_base_url=cfg.providers.ollama_base_url,
        openrouter_configured=bool(cfg.providers.openrouter_api_key),
        vault_enabled=cfg.vault_indexing.is_enabled(),
        vault_root=vault_root,
        vault_sources=len(cfg.vault_indexing.sources),
        allowlist_size=len(cfg.telegram.user_allowlist),
    )


def compute_status(sessions_dir: Path, *, now: datetime) -> StatusSnapshot:
    """Return a 24h rollup ending at `now` (UTC)."""
    window_end = now.astimezone(UTC)
    window_start = window_end - timedelta(hours=24)

    session_ids: set[str] = set()
    patterns = 0
    human_decisions = 0
    applier_decisions = 0

    for record in _iter_records(sessions_dir, window_start, window_end):
        ts = record.get("ts")
        if not isinstance(ts, str):
            continue
        try:
            event_ts = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if event_ts.tzinfo is None:
            event_ts = event_ts.replace(tzinfo=UTC)
        if event_ts < window_start or event_ts > window_end:
            continue
        event_type = record.get("type")
        payload = record.get("payload") or {}
        sid = record.get("session_id")
        if isinstance(sid, str) and sid:
            session_ids.add(sid)
        if event_type == EventType.PATTERN_OBSERVED.value:
            patterns += 1
        elif event_type == EventType.GOVERNANCE_DECISION.value:
            verdict = payload.get("verdict") if isinstance(payload, dict) else None
            if verdict in _HUMAN_VERDICTS:
                human_decisions += 1
            elif verdict in _APPLIER_VERDICTS:
                applier_decisions += 1

    return StatusSnapshot(
        window_start=window_start,
        window_end=window_end,
        sessions=len(session_ids),
        patterns=patterns,
        decisions=human_decisions,
        applies=applier_decisions,
    )


def _iter_records(
    sessions_dir: Path, window_start: datetime, window_end: datetime
) -> Iterable[dict[str, object]]:
    if not sessions_dir.is_dir():
        return
    # Scan the UTC-date shards that can contain events in the window:
    # one for today, one for yesterday (24h window never spans more).
    for shard_date in _shards_for(window_start, window_end):
        shard = sessions_dir / shard_date
        if not shard.is_dir():
            continue
        try:
            jsonl_files = sorted(shard.glob("*.jsonl"))
        except OSError:
            continue
        for path in jsonl_files:
            yield from _read_jsonl(path)


def _shards_for(window_start: datetime, window_end: datetime) -> list[str]:
    out: list[str] = []
    cursor = window_start
    while cursor.date() <= window_end.date():
        shard = cursor.strftime("%Y-%m-%d")
        if shard not in out:
            out.append(shard)
        cursor += timedelta(days=1)
    end_shard = window_end.strftime("%Y-%m-%d")
    if end_shard not in out:
        out.append(end_shard)
    return out


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


__all__ = [
    "StatusSnapshot",
    "SystemInfo",
    "collect_system_info",
    "compute_status",
]
