"""Scriptable single-decision mode of the improvement CLI."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.config import get_config
from runtime.improvement.cli import main as improvement_main
from runtime.improvement.coding_tasks import load_tasks
from runtime.improvement.decisions import load_decisions
from runtime.improvement.proposal_loader import derive_imp_id
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.unit

_STATS = ReadStats(sessions=1, events=1, skipped=0)


def _seed(workspace: Path) -> str:
    proposal = Proposal(
        id="P-001",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Add more keyword aliases.",
        risk="low",
        rationale="Most unknowns are short greetings.",
    )
    write_proposals(
        workspace, [proposal], _STATS, when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    )
    return derive_imp_id(
        proposal.pattern_detector, list(proposal.affected), proposal.change
    )


def test_approve_records_decision_and_queues_ct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    imp_id = _seed(cfg.aegis_home)
    rc = improvement_main([
        "--decide", imp_id,
        "--verdict", "approve",
        "--rationale", "looks safe",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"recorded approve on {imp_id}" in out
    assert "queued CT-001" in out
    decisions = load_decisions(cfg.aegis_home)
    assert len(decisions) == 1
    assert decisions[0].verdict == "approve"
    tasks = load_tasks(cfg.aegis_home)
    assert len(tasks) == 1
    assert tasks[0].ct_id == "CT-001"
    assert tasks[0].imp_id == imp_id


def test_reject_does_not_queue_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    imp_id = _seed(cfg.aegis_home)
    rc = improvement_main([
        "--decide", imp_id,
        "--verdict", "reject",
        "--rationale", "too risky",
    ])
    assert rc == 0
    capsys.readouterr()
    assert load_tasks(cfg.aegis_home) == []
    decisions = load_decisions(cfg.aegis_home)
    assert decisions[0].verdict == "reject"


def test_unknown_imp_id_returns_one(capsys: pytest.CaptureFixture[str]) -> None:
    _seed(get_config().aegis_home)
    rc = improvement_main([
        "--decide", "IMP-deadbeef",
        "--verdict", "approve",
        "--rationale", "x",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown IMP id" in err


def test_re_approve_does_not_duplicate_ct(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    imp_id = _seed(cfg.aegis_home)
    improvement_main([
        "--decide", imp_id, "--verdict", "approve", "--rationale", "first",
    ])
    capsys.readouterr()
    rc = improvement_main([
        "--decide", imp_id, "--verdict", "approve", "--rationale", "again",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already queued" in out
    assert len(load_tasks(cfg.aegis_home)) == 1
    decisions = load_decisions(cfg.aegis_home)
    assert len(decisions) == 2
    assert decisions[1].supersedes is not None


def test_list_mode_prints_pending(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed(get_config().aegis_home)
    rc = improvement_main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 pending" in out
    assert "unknown_intent" in out
