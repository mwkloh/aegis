"""End-to-end Phase 3 — fixture PROPOSALS.md → DECISIONS.md + CODING_TASKS.md."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.config import get_config
from runtime.improvement.cli import main as improvement_main
from runtime.improvement.proposal_loader import derive_imp_id, load_proposals
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.e2e

_STATS = ReadStats(sessions=2, events=10, skipped=0)


def _seed(workspace: Path) -> tuple[str, str]:
    proposals = [
        Proposal(
            id="P-001",
            pattern_detector="unknown_intent",
            affected=["intent_classifier"],
            change="Add aliases for 'hello'.",
            risk="low",
            rationale="r",
        ),
        Proposal(
            id="P-002",
            pattern_detector="low_confidence",
            affected=["intent_classifier"],
            change="Lower threshold to 0.4.",
            risk="medium",
            rationale="r",
        ),
    ]
    write_proposals(
        workspace, proposals, _STATS, when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    )
    return (
        derive_imp_id(
            proposals[0].pattern_detector,
            list(proposals[0].affected),
            proposals[0].change,
        ),
        derive_imp_id(
            proposals[1].pattern_detector,
            list(proposals[1].affected),
            proposals[1].change,
        ),
    )


def test_e2e_approve_then_reject_writes_both_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    imp_a, imp_b = _seed(cfg.aegis_home)
    assert len(load_proposals(cfg.aegis_home)) == 2

    rc1 = improvement_main([
        "--decide", imp_a, "--verdict", "approve", "--rationale", "ship it",
    ])
    rc2 = improvement_main([
        "--decide", imp_b, "--verdict", "reject", "--rationale", "drift risk",
    ])
    assert rc1 == 0
    assert rc2 == 0

    decisions_path = cfg.aegis_home / "improvement" / "DECISIONS.md"
    tasks_path = cfg.aegis_home / "improvement" / "CODING_TASKS.md"
    assert decisions_path.is_file()
    assert tasks_path.is_file()
    decisions_text = decisions_path.read_text(encoding="utf-8")
    tasks_text = tasks_path.read_text(encoding="utf-8")
    assert imp_a in decisions_text
    assert imp_b in decisions_text
    assert "approve" in decisions_text
    assert "reject" in decisions_text
    assert imp_a in tasks_text  # only the approval queues a CT
    assert imp_b not in tasks_text


def test_e2e_list_after_approve_drops_pending(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    imp_a, _ = _seed(cfg.aegis_home)
    improvement_main([
        "--decide", imp_a, "--verdict", "approve", "--rationale", "ok",
    ])
    capsys.readouterr()
    rc = improvement_main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 pending of 2 loaded" in out
