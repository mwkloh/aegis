"""Read JSONL events out of `<sessions_dir>/YYYY-MM-DD/*.jsonl`.

Plane-2 read-only path. Validates each line against `Event` (Pydantic,
extra=forbid). Malformed or out-of-schema lines are skipped — the count
is surfaced in the returned `ReadStats` so the CLI can warn the user
instead of failing the run.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Event(BaseModel):
    """Schema mirror of `EventStream.append()` records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: datetime
    session_id: str = Field(min_length=1, max_length=128)
    type: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any]


class ReadStats(BaseModel):
    """Summary of one window read — used by the CLI banner and tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: int
    events: int
    skipped: int


def read_window(
    sessions_dir: Path, since: date | None = None
) -> tuple[list[Event], ReadStats]:
    """Return all events on/after `since` (default: today UTC)."""
    sessions_dir = Path(sessions_dir)
    cutoff = since or datetime.now(tz=UTC).date()
    events: list[Event] = []
    sessions_seen: set[str] = set()
    skipped = 0

    if not sessions_dir.is_dir():
        return events, ReadStats(sessions=0, events=0, skipped=0)

    for day_dir in sorted(sessions_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        day = _parse_day(day_dir.name)
        if day is None or day < cutoff:
            continue
        for path in sorted(day_dir.glob("*.jsonl")):
            for evt in _iter_events(path):
                if evt is None:
                    skipped += 1
                    continue
                events.append(evt)
                sessions_seen.add(evt.session_id)

    return events, ReadStats(
        sessions=len(sessions_seen), events=len(events), skipped=skipped
    )


def _parse_day(name: str) -> date | None:
    try:
        return datetime.strptime(name, "%Y-%m-%d").replace(tzinfo=UTC).date()
    except ValueError:
        return None


def _iter_events(path: Path) -> Iterator[Event | None]:
    """Yield validated `Event` per line, or `None` for malformed lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            yield None
            continue
        if not isinstance(data, dict):
            yield None
            continue
        try:
            yield Event.model_validate(data)
        except ValidationError:
            yield None
