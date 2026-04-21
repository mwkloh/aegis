"""Append-only writer for PATTERNS.md / PROPOSALS.md."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.reflection.event_reader import ReadStats
from runtime.reflection.patterns import PatternRecord
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_patterns, write_proposals

pytestmark = pytest.mark.unit

_FIXED = datetime(2026, 4, 18, 18, 42, tzinfo=UTC)
_STATS = ReadStats(sessions=3, events=147, skipped=0)


def _patterns() -> list[PatternRecord]:
    return [
        PatternRecord(
            detector="unknown_intent",
            severity="medium",
            count=4,
            sample_session_ids=["s1", "s2"],
            summary="Classifier returned 'unknown' 4 times.",
        ),
    ]


def _proposals() -> list[Proposal]:
    return [
        Proposal(
            id="P-001",
            pattern_detector="unknown_intent",
            affected=["intent_classifier"],
            change="Lower the threshold for new keyword aliases.",
            risk="low",
            rationale="Repeated misses suggest missing aliases.",
        ),
    ]


def test_writes_patterns_under_workspace_reflection(tmp_path: Path) -> None:
    out = write_patterns(tmp_path, _patterns(), _STATS, when=_FIXED)
    assert out == tmp_path / "reflection" / "PATTERNS.md"
    text = out.read_text(encoding="utf-8")
    assert "## 2026-04-18T18:42Z — sessions=3, events=147, skipped=0" in text
    assert "**unknown_intent** (medium, count=4)" in text
    assert "Classifier returned 'unknown' 4 times." in text


def test_writes_proposals_with_id_and_risk(tmp_path: Path) -> None:
    out = write_proposals(tmp_path, _proposals(), _STATS, when=_FIXED)
    text = out.read_text(encoding="utf-8")
    assert "### P-001 — unknown_intent (risk: low)" in text
    assert "**Affected:** intent_classifier" in text
    assert "**Change:** Lower the threshold" in text


def test_append_only_preserves_history(tmp_path: Path) -> None:
    write_patterns(tmp_path, _patterns(), _STATS, when=_FIXED)
    later = datetime(2026, 4, 19, 9, 0, tzinfo=UTC)
    out = write_patterns(tmp_path, [], _STATS, when=later)
    text = out.read_text(encoding="utf-8")
    assert text.count("## 2026-04-18T18:42Z") == 1
    assert text.count("## 2026-04-19T09:00Z") == 1
    # Earlier section MUST appear before later section.
    assert text.index("2026-04-18") < text.index("2026-04-19")
    # Empty pattern run still emits its header + the empty marker.
    assert "_No patterns detected._" in text


def test_does_not_touch_canon_files(tmp_path: Path) -> None:
    canon = tmp_path / "AGENTS.md"
    canon.write_text("# canonical\n", encoding="utf-8")
    write_patterns(tmp_path, _patterns(), _STATS, when=_FIXED)
    write_proposals(tmp_path, _proposals(), _STATS, when=_FIXED)
    assert canon.read_text(encoding="utf-8") == "# canonical\n"
    assert (tmp_path / "reflection").is_dir()
