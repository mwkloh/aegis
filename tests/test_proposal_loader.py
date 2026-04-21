"""Parse `<workspace>/reflection/PROPOSALS.md` into typed records."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.improvement.proposal_loader import (
    derive_imp_id,
    load_proposals,
    proposals_path,
)
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.unit

_STATS = ReadStats(sessions=2, events=10, skipped=0)


def _seed(workspace: Path, proposals: list[Proposal], when: datetime) -> Path:
    return write_proposals(workspace, proposals, _STATS, when=when)


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_proposals(tmp_path) == []
    assert proposals_path(tmp_path) == tmp_path / "reflection" / "PROPOSALS.md"


def test_load_parses_two_proposals_from_one_run(tmp_path: Path) -> None:
    when = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    _seed(
        tmp_path,
        [
            Proposal(
                id="P-001",
                pattern_detector="unknown_intent",
                affected=["intent_classifier"],
                change="Add more keyword aliases for short phrases.",
                risk="low",
                rationale="Most unknowns are short greetings.",
            ),
            Proposal(
                id="P-002",
                pattern_detector="low_confidence",
                affected=["intent_classifier"],
                change="Lower the confidence floor to 0.4.",
                risk="medium",
                rationale="Borderline classifications miss real signal.",
            ),
        ],
        when,
    )
    loaded = load_proposals(tmp_path)
    assert len(loaded) == 2
    by_detector = {p.pattern_detector: p for p in loaded}
    assert "unknown_intent" in by_detector
    assert "low_confidence" in by_detector
    unknown = by_detector["unknown_intent"]
    assert unknown.affected == ["intent_classifier"]
    assert unknown.change.startswith("Add more keyword aliases")
    assert unknown.risk == "low"
    assert unknown.rationale.startswith("Most unknowns")
    assert unknown.source_run == "2026-04-18T12:00Z"
    assert unknown.imp_id.startswith("IMP-")
    assert len(unknown.imp_id) == 12


def test_imp_id_stable_across_two_runs(tmp_path: Path) -> None:
    proposal = Proposal(
        id="P-001",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Add more keyword aliases for short phrases.",
        risk="low",
        rationale="Demo.",
    )
    _seed(tmp_path, [proposal], datetime(2026, 4, 18, 9, 0, tzinfo=UTC))
    _seed(tmp_path, [proposal], datetime(2026, 4, 19, 9, 0, tzinfo=UTC))
    loaded = load_proposals(tmp_path)
    # Latest source_run wins on the same content hash.
    assert len(loaded) == 1
    assert loaded[0].source_run == "2026-04-19T09:00Z"


def test_imp_id_changes_when_change_text_differs(tmp_path: Path) -> None:
    a = Proposal(
        id="P-001",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Lower the confidence floor.",
        risk="low",
        rationale="r",
    )
    b = Proposal(
        id="P-002",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Raise the confidence floor.",
        risk="low",
        rationale="r",
    )
    _seed(tmp_path, [a, b], datetime(2026, 4, 18, 0, 0, tzinfo=UTC))
    loaded = load_proposals(tmp_path)
    assert len({p.imp_id for p in loaded}) == 2


def test_derive_imp_id_is_deterministic() -> None:
    one = derive_imp_id("unknown_intent", ["a", "b"], "do x")
    two = derive_imp_id("unknown_intent", ["b", "a"], "do x")
    assert one == two  # affected order does not matter
    assert one.startswith("IMP-")
    assert len(one) == 12


def test_load_skips_malformed_section(tmp_path: Path) -> None:
    # Hand-craft a file with one valid and one truncated entry.
    path = proposals_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "## 2026-04-18T12:00Z — sessions=1, events=1, skipped=0\n"
        "\n"
        "### P-001 — unknown_intent (risk: low)\n"
        "- **Affected:** intent_classifier\n"
        "- **Change:** Add aliases.\n"
        "- **Rationale:** ok\n"
        "### P-002 — broken_one (risk: low)\n"
        # missing Change/Affected/Rationale → skipped
        "\n",
        encoding="utf-8",
    )
    loaded = load_proposals(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].pattern_detector == "unknown_intent"
