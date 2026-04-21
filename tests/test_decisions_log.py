"""Append-only DECISIONS.md governance log."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.events import EventStream
from runtime.improvement.decisions import (
    decisions_path,
    latest_by_imp,
    load_decisions,
    record_decision,
)

pytestmark = pytest.mark.unit

_IMP = "IMP-7f3a1c2b"


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_decisions(tmp_path) == []


def test_record_creates_file_with_header(tmp_path: Path) -> None:
    when = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="approve",
        rationale="safe; matches demo",
        when=when,
    )
    text = decisions_path(tmp_path).read_text(encoding="utf-8")
    assert "AEGIS — Improvement Decisions" in text
    assert "## 2026-04-18T12:00Z — IMP-7f3a1c2b — approve" in text
    assert "**Rationale:** safe; matches demo" in text
    assert "**Supersedes:** —" in text


def test_redeciding_supersedes_prior(tmp_path: Path) -> None:
    first = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    second = datetime(2026, 4, 18, 13, 30, tzinfo=UTC)
    record_decision(tmp_path, imp_id=_IMP, verdict="defer", rationale="not yet", when=first)
    record_decision(
        tmp_path, imp_id=_IMP, verdict="approve", rationale="ok now", when=second
    )
    loaded = load_decisions(tmp_path)
    assert len(loaded) == 2
    assert loaded[0].verdict == "defer"
    assert loaded[0].supersedes is None
    assert loaded[1].verdict == "approve"
    assert loaded[1].supersedes == "2026-04-18T12:00Z"
    # Latest-by-imp returns the most recent verdict.
    latest = latest_by_imp(loaded)
    assert latest[_IMP].verdict == "approve"


def test_record_emits_governance_event(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    events = EventStream(sessions_dir)
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="reject",
        rationale="too risky",
        events=events,
    )
    log_text = events.path.read_text(encoding="utf-8")
    assert '"type": "governance.decision"' in log_text
    assert '"verdict": "reject"' in log_text
    assert '"imp_id": "IMP-7f3a1c2b"' in log_text


def test_does_not_touch_canon_files(tmp_path: Path) -> None:
    canon = tmp_path / "AGENTS.md"
    canon.write_text("# canonical\n", encoding="utf-8")
    record_decision(
        tmp_path, imp_id=_IMP, verdict="approve", rationale="x",
        when=datetime(2026, 4, 18, 0, 0, tzinfo=UTC),
    )
    assert canon.read_text(encoding="utf-8") == "# canonical\n"
    assert (tmp_path / "improvement").is_dir()
