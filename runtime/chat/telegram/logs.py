"""Operator-facing `/logs [N]` — tail of today's structural event shard.

Reads the append-only JSONL files written by `runtime.events.EventStream`
at `<sessions_dir>/YYYY-MM-DD/<session_id>.jsonl`. The pattern mirrors
`runtime.chat.telegram.status` (same shard layout, same never-raise
posture) but scopes to a single UTC date — the operator asked for a
tail, not a rollup.

Never raises. Malformed JSONL lines, missing shards, and unreadable
files are all skipped silently — an audit tool that crashes on a torn
append is worse than one that returns a partial view.

Structural-only by design: payloads stored on the shard already omit
message bodies (Phase 7 §3.3). `/logs` just surfaces what's already
safe to expose.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LOG_LINES = 20
MAX_LOG_LINES = 100


def tail_events(
    sessions_dir: Path, *, n: int, now: datetime
) -> tuple[list[dict[str, object]], int]:
    """Return today's last `n` events + the scanned-today total.

    The returned list is chronological (oldest→newest) within the tail
    — matches operator mental model of `tail -f`. Total count is the
    full event count for today (useful in the header so the operator
    knows how much was truncated).

    `n` is clamped to `[1, MAX_LOG_LINES]`.
    """
    n = max(1, min(n, MAX_LOG_LINES))
    now_utc = now.astimezone(UTC)
    today = now_utc.strftime("%Y-%m-%d")
    shard_dir = sessions_dir / today
    if not shard_dir.is_dir():
        return [], 0
    try:
        files = sorted(shard_dir.glob("*.jsonl"))
    except OSError:
        return [], 0
    events: list[dict[str, object]] = []
    for path in files:
        events.extend(_read_jsonl(path))
    total = len(events)
    return events[-n:], total


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
    "DEFAULT_LOG_LINES",
    "MAX_LOG_LINES",
    "tail_events",
]
