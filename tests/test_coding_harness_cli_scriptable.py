"""Scriptable modes of the coding harness CLI (--list, --task, --force)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness.cli import main as harness_main
from runtime.coding_harness.patch_writer import existing_drafts_for
from runtime.config import get_config
from runtime.improvement.cli import main as improvement_main
from runtime.improvement.proposal_loader import derive_imp_id
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.unit

_STATS = ReadStats(sessions=1, events=1, skipped=0)


def _seed_eligible(workspace: Path) -> str:
    """Write a proposal, approve it, and return the IMP-id."""
    proposal = Proposal(
        id="P-001",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Add aliases for 'hi'.",
        risk="low",
        rationale="r",
    )
    write_proposals(
        workspace, [proposal], _STATS, when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    )
    imp_id = derive_imp_id(
        proposal.pattern_detector, list(proposal.affected), proposal.change
    )
    improvement_main([
        "--decide", imp_id, "--verdict", "approve", "--rationale", "ok",
    ])
    return imp_id


def test_list_mode_prints_eligible(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_eligible(get_config().aegis_home)
    capsys.readouterr()  # drop improvement output
    rc = harness_main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 eligible of 1 queued" in out
    assert "CT-001" in out


def test_unknown_task_returns_one(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_eligible(get_config().aegis_home)
    capsys.readouterr()
    rc = harness_main(["--task", "CT-999"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "CT-999" in err
    assert "not eligible" in err


def test_default_run_writes_stub_when_no_client(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No OPENROUTER_API_KEY and no Ollama → stub patches written, exit 0."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")  # non-loopback → instant reject
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()
    rc = harness_main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 eligible CT(s)" in out
    assert "stub" in out
    drafts = existing_drafts_for(cfg.aegis_home, "CT-001")
    assert len(drafts) == 1
    text = drafts[0].read_text(encoding="utf-8")
    assert "status: stub" in text


def test_skip_when_prior_patch_exists(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")  # non-loopback → instant reject
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()
    harness_main([])
    capsys.readouterr()
    rc = harness_main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already drafted" in out
    assert len(existing_drafts_for(cfg.aegis_home, "CT-001")) == 1


def test_force_writes_second_patch_alongside(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")  # non-loopback → instant reject
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()
    harness_main([])
    capsys.readouterr()
    rc = harness_main(["--force"])
    assert rc == 0
    drafts = existing_drafts_for(cfg.aegis_home, "CT-001")
    assert len(drafts) == 2  # original kept; new one alongside
