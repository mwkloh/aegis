"""End-to-end Phase 4 — approve a proposal → harness writes a `.patch.md`."""
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

pytestmark = pytest.mark.e2e

_STATS = ReadStats(sessions=1, events=4, skipped=0)


def _seed_two_proposals(workspace: Path) -> tuple[str, str]:
    proposals = [
        Proposal(
            id="P-001",
            pattern_detector="unknown_intent",
            affected=["intent_classifier"],
            change="Add aliases for 'hi'.",
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


def test_e2e_only_approved_ct_gets_patch(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Approve one of two proposals → harness drafts only that CT."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")  # non-loopback → instant reject
    cfg = get_config()
    imp_a, imp_b = _seed_two_proposals(cfg.aegis_home)

    improvement_main([
        "--decide", imp_a, "--verdict", "approve", "--rationale", "ship",
    ])
    improvement_main([
        "--decide", imp_b, "--verdict", "reject", "--rationale", "drift",
    ])
    capsys.readouterr()

    rc = harness_main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 eligible CT(s)" in out
    assert "CT-001" in out
    assert "CT-002" not in out

    drafts = existing_drafts_for(cfg.aegis_home, "CT-001")
    assert len(drafts) == 1
    text = drafts[0].read_text(encoding="utf-8")
    assert imp_a in text
    assert "status: stub" in text  # no LLM client wired
