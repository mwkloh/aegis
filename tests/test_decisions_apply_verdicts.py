"""Phase 5 extension to ``Verdict``: applier outcomes get logged too.

Verifies that the new verdicts (``applied_clean``,
``applied_test_failed``, ``apply_conflict``, ``reverted``) round-trip
through record → load and that they correctly supersede a prior
``approve`` for the same ``IMP-id``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.improvement.decisions import (
    Decision,
    decisions_path,
    latest_by_imp,
    load_decisions,
    record_decision,
)

pytestmark = pytest.mark.unit

_IMP = "IMP-a86b087a"


@pytest.mark.parametrize(
    "verdict",
    ["applied_clean", "applied_test_failed", "apply_conflict", "reverted"],
)
def test_apply_verdicts_validate(verdict: str) -> None:
    d = Decision(
        imp_id=_IMP,
        verdict=verdict,  # type: ignore[arg-type]
        rationale="98 passed in 41.2s",
        decided_at=datetime(2026, 4, 19, 8, 30, tzinfo=UTC),
        supersedes="2026-04-18T12:00Z",
    )
    assert d.verdict == verdict


def test_unknown_verdict_still_rejected() -> None:
    with pytest.raises(ValidationError):
        Decision(
            imp_id=_IMP,
            verdict="exploded",  # type: ignore[arg-type]
            rationale="",
            decided_at=datetime(2026, 4, 19, tzinfo=UTC),
            supersedes=None,
        )


def test_apply_verdict_is_persisted_and_reloaded(tmp_path: Path) -> None:
    when = datetime(2026, 4, 19, 8, 30, tzinfo=UTC)
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="applied_clean",
        rationale="98 passed in 41.2s on aegis/CT-001-a86b087a",
        when=when,
    )
    text = decisions_path(tmp_path).read_text(encoding="utf-8")
    assert "## 2026-04-19T08:30Z — IMP-a86b087a — applied_clean" in text

    rows = load_decisions(tmp_path)
    assert len(rows) == 1
    assert rows[0].verdict == "applied_clean"
    assert "98 passed" in rows[0].rationale


def test_applied_test_failed_supersedes_prior_approve(tmp_path: Path) -> None:
    """A failing apply must overwrite an earlier ``approve`` for the same IMP."""
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="approve",
        rationale="LGTM",
        when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
    )
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="applied_test_failed",
        rationale="2 pytest failures on aegis/CT-001-a86b087a",
        when=datetime(2026, 4, 19, 8, 30, tzinfo=UTC),
    )

    rows = load_decisions(tmp_path)
    assert [r.verdict for r in rows] == ["approve", "applied_test_failed"]

    latest = latest_by_imp(rows)
    assert latest[_IMP].verdict == "applied_test_failed"
    assert latest[_IMP].supersedes == "2026-04-18T12:00Z"


def test_apply_conflict_round_trips_through_section_regex(tmp_path: Path) -> None:
    """Regression guard: the section regex must accept the new verbs verbatim."""
    when = datetime(2026, 4, 19, 9, 15, tzinfo=UTC)
    record_decision(
        tmp_path,
        imp_id=_IMP,
        verdict="apply_conflict",
        rationale="error: patch failed: runtime/foo.py:12",
        when=when,
    )
    rows = load_decisions(tmp_path)
    assert len(rows) == 1
    assert rows[0].verdict == "apply_conflict"
    assert rows[0].decided_at == when
