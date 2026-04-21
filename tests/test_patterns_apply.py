"""Phase 6 Track A — outcome-driven pattern detectors.

Pins the contract for the three detectors that close the recursive
loop by reading Plane 3 governance/harness events and turning them
into structural `PatternRecord`s the Reflection plane can cluster.

* ``detect_apply_failed_repeat`` — ≥2 non-clean apply verdicts for the
  same ``imp_id`` (high severity).
* ``detect_harness_refused_repeat`` — ≥2 ``harness.refused`` events for
  the same ``imp_id`` (medium).
* ``detect_context_mode_helps`` — ``harness_with_context`` followed by
  ``applied_clean`` for the same ``imp_id`` (low, positive signal).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from runtime.reflection.event_reader import Event
from runtime.reflection.patterns import (
    detect_all,
    detect_apply_failed_repeat,
    detect_context_mode_helps,
    detect_harness_refused_repeat,
)

pytestmark = pytest.mark.unit


def _evt(type_: str, session_id: str = "s", **payload: Any) -> Event:
    return Event(
        ts=datetime.now(tz=UTC),
        session_id=session_id,
        type=type_,
        payload=payload,
    )


# --- detect_apply_failed_repeat ---------------------------------------------


def test_apply_failed_repeat_below_threshold_returns_nothing() -> None:
    events = [
        _evt(
            "governance.decision",
            session_id="s1",
            imp_id="IMP-aaaaaaaa",
            verdict="applied_test_failed",
        )
    ]
    assert detect_apply_failed_repeat(events) == []


def test_apply_failed_repeat_at_threshold_emits_high_record() -> None:
    events = [
        _evt(
            "governance.decision", session_id=f"s{i}",
            imp_id="IMP-aaaaaaaa", verdict="applied_test_failed",
        )
        for i in range(2)
    ]
    out = detect_apply_failed_repeat(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.detector == "apply_failed_repeat"
    assert rec.severity == "high"
    assert rec.count == 2
    assert "IMP-aaaaaaaa" in rec.summary


def test_apply_failed_repeat_mixed_verdicts_count_only_non_clean() -> None:
    events = [
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_test_failed", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="apply_conflict", session_id="s2"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="reverted", session_id="s3"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_clean", session_id="s4"),  # ignored
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="approve", session_id="s5"),  # ignored (human verdict)
    ]
    out = detect_apply_failed_repeat(events)
    assert len(out) == 1
    assert out[0].count == 3


def test_apply_failed_repeat_groups_per_imp_id() -> None:
    events = [
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_test_failed", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_test_failed", session_id="s2"),
        _evt("governance.decision", imp_id="IMP-bbbbbbbb",
             verdict="apply_conflict", session_id="s3"),  # only 1 → skipped
    ]
    out = detect_apply_failed_repeat(events)
    assert len(out) == 1
    assert "IMP-aaaaaaaa" in out[0].summary


def test_apply_failed_repeat_skips_events_without_imp_id() -> None:
    events = [
        _evt("governance.decision", verdict="applied_test_failed", session_id="s1"),
        _evt("governance.decision", verdict="applied_test_failed", session_id="s2"),
    ]
    assert detect_apply_failed_repeat(events) == []


# --- detect_harness_refused_repeat ------------------------------------------


def test_harness_refused_repeat_below_threshold_returns_nothing() -> None:
    events = [
        _evt("harness.refused", imp_id="IMP-aaaaaaaa",
             ct_id="CT-001", reason="scope_touches_canon", session_id="s1"),
    ]
    assert detect_harness_refused_repeat(events) == []


def test_harness_refused_repeat_at_threshold_emits_medium_record() -> None:
    events = [
        _evt("harness.refused", imp_id="IMP-aaaaaaaa",
             ct_id="CT-001", reason="scope_touches_canon", session_id=f"s{i}")
        for i in range(2)
    ]
    out = detect_harness_refused_repeat(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.detector == "harness_refused_repeat"
    assert rec.severity == "medium"
    assert rec.count == 2
    assert "IMP-aaaaaaaa" in rec.summary


def test_harness_refused_repeat_per_imp_grouping() -> None:
    events = [
        _evt("harness.refused", imp_id="IMP-aaaaaaaa", session_id="s1"),
        _evt("harness.refused", imp_id="IMP-aaaaaaaa", session_id="s2"),
        _evt("harness.refused", imp_id="IMP-bbbbbbbb", session_id="s3"),  # 1 → skip
    ]
    out = detect_harness_refused_repeat(events)
    assert len(out) == 1
    assert "IMP-aaaaaaaa" in out[0].summary


def test_harness_refused_repeat_ignores_unrelated_events() -> None:
    events = [
        _evt("harness.refused", imp_id="IMP-aaaaaaaa", session_id="s1"),
        _evt("harness.refused", imp_id="IMP-aaaaaaaa", session_id="s2"),
        _evt("harness.draft.start", imp_id="IMP-aaaaaaaa", session_id="s3"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="reject", session_id="s4"),
    ]
    out = detect_harness_refused_repeat(events)
    assert len(out) == 1
    assert out[0].count == 2  # only the two refusals


# --- detect_context_mode_helps ----------------------------------------------


def test_context_mode_helps_emits_when_clean_apply_follows() -> None:
    events = [
        _evt(
            "pattern.observed",
            session_id="s1",
            pattern="harness_with_context",
            ct_id="CT-001", imp_id="IMP-aaaaaaaa",
            files=3, skills=1, total_bytes=4096, truncated=False,
        ),
        _evt(
            "governance.decision",
            session_id="s2",
            imp_id="IMP-aaaaaaaa",
            verdict="applied_clean",
        ),
    ]
    out = detect_context_mode_helps(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.detector == "context_mode_helps"
    assert rec.severity == "low"
    assert rec.count == 1
    assert "IMP-aaaaaaaa" in rec.summary
    assert "--with-context" in rec.summary


def test_context_mode_helps_silent_when_no_clean_apply() -> None:
    events = [
        _evt("pattern.observed", pattern="harness_with_context",
             imp_id="IMP-aaaaaaaa", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_test_failed", session_id="s2"),
    ]
    assert detect_context_mode_helps(events) == []


def test_context_mode_helps_silent_when_clean_apply_without_context_event() -> None:
    """An ``applied_clean`` without a matching context-mode event is not a hit."""
    events = [
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_clean", session_id="s1"),
    ]
    assert detect_context_mode_helps(events) == []


def test_context_mode_helps_ignores_other_pattern_observed_signals() -> None:
    """Only the ``harness_with_context`` pattern counts."""
    events = [
        _evt("pattern.observed", pattern="tier1_missing",
             imp_id="IMP-aaaaaaaa", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_clean", session_id="s2"),
    ]
    assert detect_context_mode_helps(events) == []


def test_context_mode_helps_one_record_per_imp_id() -> None:
    events = [
        _evt("pattern.observed", pattern="harness_with_context",
             imp_id="IMP-aaaaaaaa", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_clean", session_id="s2"),
        _evt("pattern.observed", pattern="harness_with_context",
             imp_id="IMP-bbbbbbbb", session_id="s3"),
        _evt("governance.decision", imp_id="IMP-bbbbbbbb",
             verdict="applied_clean", session_id="s4"),
    ]
    out = detect_context_mode_helps(events)
    assert len(out) == 2
    summaries = [r.summary for r in out]
    assert any("IMP-aaaaaaaa" in s for s in summaries)
    assert any("IMP-bbbbbbbb" in s for s in summaries)


# --- detect_all wiring ------------------------------------------------------


def test_detect_all_includes_phase_6_detectors() -> None:
    events = [
        # apply_failed_repeat trigger
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="applied_test_failed", session_id="s1"),
        _evt("governance.decision", imp_id="IMP-aaaaaaaa",
             verdict="apply_conflict", session_id="s2"),
        # harness_refused_repeat trigger
        _evt("harness.refused", imp_id="IMP-bbbbbbbb", session_id="s3"),
        _evt("harness.refused", imp_id="IMP-bbbbbbbb", session_id="s4"),
        # context_mode_helps trigger
        _evt("pattern.observed", pattern="harness_with_context",
             imp_id="IMP-cccccccc", session_id="s5"),
        _evt("governance.decision", imp_id="IMP-cccccccc",
             verdict="applied_clean", session_id="s6"),
    ]
    detectors = {r.detector for r in detect_all(events)}
    assert "apply_failed_repeat" in detectors
    assert "harness_refused_repeat" in detectors
    assert "context_mode_helps" in detectors
