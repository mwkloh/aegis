"""Interactive review loop — drives _do_interactive with a stub input()."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.config import get_config
from runtime.improvement.cli import run_interactive
from runtime.improvement.coding_tasks import load_tasks
from runtime.improvement.decisions import load_decisions
from runtime.improvement.proposal_loader import load_proposals
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.unit

_STATS = ReadStats(sessions=1, events=1, skipped=0)


def _seed(workspace: Path) -> None:
    write_proposals(
        workspace,
        [
            Proposal(
                id="P-001",
                pattern_detector="unknown_intent",
                affected=["intent_classifier"],
                change="Add aliases.",
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
            Proposal(
                id="P-003",
                pattern_detector="tool_error",
                affected=["echo_tool"],
                change="Retry once on transient failure.",
                risk="low",
                rationale="r",
            ),
        ],
        _STATS,
        when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
    )


def test_walks_three_proposals_with_distinct_verdicts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    _seed(cfg.aegis_home)
    proposals = load_proposals(cfg.aegis_home)
    assert len(proposals) == 3
    # Order from load_proposals is dict insertion order (latest write per imp_id).
    # The script answers each (verdict, rationale) pair then quits.
    answers = iter([
        "a", "approving first",
        "r", "rejecting second",
        "d", "deferring third",
    ])
    rc = run_interactive(
        cfg, proposals, input_fn=lambda _prompt: next(answers)
    )
    assert rc == 0
    decisions = load_decisions(cfg.aegis_home)
    verdicts = [d.verdict for d in decisions]
    assert sorted(verdicts) == ["approve", "defer", "reject"]
    tasks = load_tasks(cfg.aegis_home)
    # Only the approve produces a CT.
    assert len(tasks) == 1
    out = capsys.readouterr().out
    assert "queued CT-001" in out


def test_quit_short_circuits_remaining_proposals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    _seed(cfg.aegis_home)
    proposals = load_proposals(cfg.aegis_home)
    answers = iter(["a", "ok", "q"])
    rc = run_interactive(
        cfg, proposals, input_fn=lambda _prompt: next(answers)
    )
    assert rc == 0
    decisions = load_decisions(cfg.aegis_home)
    assert len(decisions) == 1
    assert decisions[0].verdict == "approve"
    out = capsys.readouterr().out
    assert "quit — remaining proposals untouched." in out


def test_skip_records_no_decision(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = get_config()
    _seed(cfg.aegis_home)
    proposals = load_proposals(cfg.aegis_home)
    # 's' on every proposal — no rationale prompt fires for skip.
    answers = iter(["s", "s", "s"])
    rc = run_interactive(
        cfg, proposals, input_fn=lambda _prompt: next(answers)
    )
    assert rc == 0
    assert load_decisions(cfg.aegis_home) == []
    out = capsys.readouterr().out
    assert out.count("skipped") == 3


def test_unrecognized_input_re_prompts(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = get_config()
    _seed(cfg.aegis_home)
    proposals = load_proposals(cfg.aegis_home)[:1]
    # First input is garbage, second is 'a', then rationale, then we're done.
    answers = iter(["xyz", "a", "ok"])
    rc = run_interactive(
        cfg, proposals, input_fn=lambda _prompt: next(answers)
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "unrecognized" in out
    assert len(load_decisions(cfg.aegis_home)) == 1
