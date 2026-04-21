"""Per-detector tests for `runtime.reflection.patterns`."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from runtime.reflection.event_reader import Event
from runtime.reflection.patterns import (
    detect_low_confidence,
    detect_model_latency,
    detect_tier1_unavailable,
    detect_tool_error,
    detect_unknown_intent,
)

pytestmark = pytest.mark.unit


def _evt(type_: str, session_id: str = "s", **payload: Any) -> Event:
    return Event(
        ts=datetime.now(tz=UTC),
        session_id=session_id,
        type=type_,
        payload=payload,
    )


def test_unknown_intent_below_threshold_returns_nothing() -> None:
    events = [_evt("intent.classified", intent="unknown") for _ in range(2)]
    assert detect_unknown_intent(events) == []


def test_unknown_intent_at_threshold_emits_one_record() -> None:
    events = [_evt("intent.classified", session_id=f"s{i}", intent="unknown") for i in range(3)]
    out = detect_unknown_intent(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.detector == "unknown_intent"
    assert rec.severity == "medium"
    assert rec.count == 3
    assert len(rec.sample_session_ids) == 3


def test_low_confidence_uses_cutoff_and_skips_high_conf() -> None:
    events = [
        _evt("intent.classified", confidence=0.4),
        _evt("intent.classified", confidence=0.9),  # ignored
        _evt("intent.classified", confidence=0.1),
        _evt("intent.classified", confidence=0.49),
    ]
    out = detect_low_confidence(events)
    assert len(out) == 1
    assert out[0].count == 3
    assert out[0].severity == "low"


def test_tool_error_groups_by_tool_name() -> None:
    events = [
        _evt("tool.invoked", session_id="a", tool="echo"),
        _evt("tool.result", session_id="a", status="error", error="boom"),
        _evt("tool.invoked", session_id="b", tool="echo"),
        _evt("tool.result", session_id="b", status="error", error="boom"),
        _evt("tool.invoked", session_id="c", tool="echo"),
        _evt("tool.result", session_id="c", status="error", error="boom"),
        _evt("tool.invoked", session_id="d", tool="time"),
        _evt("tool.result", session_id="d", status="error", error="zone"),  # below threshold
    ]
    out = detect_tool_error(events)
    assert len(out) == 1
    assert out[0].detector == "tool_error"
    assert out[0].severity == "high"
    assert "echo" in out[0].summary


def test_tier1_unavailable_fires_on_any_count() -> None:
    out = detect_tier1_unavailable([
        _evt("pattern.observed", pattern="tier1_missing", reason="no key"),
    ])
    assert len(out) == 1
    assert out[0].count == 1
    assert out[0].severity == "medium"


def test_model_latency_p95_threshold_breach() -> None:
    samples = [100, 150, 200, 250, 300, 6000, 7000, 9000]
    events = [
        _evt(
            "model.call.end",
            session_id=f"s{i}",
            tier="smart",
            status="ok",
            latency_ms=ms,
            tokens_in=10,
            tokens_out=10,
        )
        for i, ms in enumerate(samples)
    ]
    out = detect_model_latency(events)
    assert len(out) == 1
    assert out[0].detector == "model_latency"
    assert "smart" in out[0].summary


def test_model_latency_silent_below_threshold() -> None:
    events = [
        _evt(
            "model.call.end",
            session_id=f"s{i}",
            tier="fast",
            status="ok",
            latency_ms=200,
            tokens_in=5,
            tokens_out=5,
        )
        for i in range(8)
    ]
    assert detect_model_latency(events) == []


def test_payload_bodies_are_not_in_summary() -> None:
    """Defensive: detectors must not surface raw payload values."""
    events = [
        _evt("intent.classified", session_id=f"s{i}", intent="unknown",
             text="my secret credit card number")
        for i in range(3)
    ]
    out = detect_unknown_intent(events)
    assert "credit card" not in out[0].summary
