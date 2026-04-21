"""Deterministic pattern detectors over a window of `Event`s.

No LLM here. Each detector is a pure function: `(events) → list[PatternRecord]`.
Records carry counts + a handful of session ids — never raw user text or
model prompt bodies. That discipline matches the runtime instrumentation
rule set in Phase 1.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import quantiles
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .event_reader import Event

Severity = Literal["low", "medium", "high"]

_DEFAULT_THRESHOLD = 3
_LOW_CONFIDENCE_CUTOFF = 0.5
_LATENCY_P95_THRESHOLD_MS = 5000
_MIN_LATENCY_SAMPLES = 5
_MAX_SAMPLE_SESSIONS = 3

# Phase 6 — outcome-driven detectors over governance/harness streams.
_APPLY_FAIL_THRESHOLD = 2  # ≥2 non-clean apply verdicts for the same imp_id
_REFUSE_THRESHOLD = 2  # ≥2 harness.refused events for the same imp_id
_NON_CLEAN_APPLY_VERDICTS = frozenset({
    "applied_test_failed", "apply_conflict", "reverted",
})


class PatternRecord(BaseModel):
    """Structural summary of a recurring observation. No payload bodies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: str = Field(min_length=1, max_length=64)
    severity: Severity
    count: int = Field(ge=1)
    sample_session_ids: list[str] = Field(default_factory=list, max_length=_MAX_SAMPLE_SESSIONS)
    summary: str = Field(min_length=1, max_length=240)


def detect_all(events: Iterable[Event]) -> list[PatternRecord]:
    """Run every built-in detector and concatenate their findings."""
    materialized = list(events)
    out: list[PatternRecord] = []
    out.extend(detect_unknown_intent(materialized))
    out.extend(detect_low_confidence(materialized))
    out.extend(detect_tool_error(materialized))
    out.extend(detect_tier1_unavailable(materialized))
    out.extend(detect_model_latency(materialized))
    out.extend(detect_apply_failed_repeat(materialized))
    out.extend(detect_harness_refused_repeat(materialized))
    out.extend(detect_context_mode_helps(materialized))
    return out


def detect_unknown_intent(events: Iterable[Event]) -> list[PatternRecord]:
    sessions: list[str] = []
    for evt in events:
        if evt.type != "intent.classified":
            continue
        if evt.payload.get("intent") == "unknown":
            sessions.append(evt.session_id)
    if len(sessions) < _DEFAULT_THRESHOLD:
        return []
    return [
        PatternRecord(
            detector="unknown_intent",
            severity="medium",
            count=len(sessions),
            sample_session_ids=_first_unique(sessions),
            summary=f"Classifier returned 'unknown' {len(sessions)} times.",
        )
    ]


def detect_low_confidence(events: Iterable[Event]) -> list[PatternRecord]:
    sessions: list[str] = []
    for evt in events:
        if evt.type != "intent.classified":
            continue
        conf = _as_float(evt.payload.get("confidence"))
        if conf is None or conf >= _LOW_CONFIDENCE_CUTOFF:
            continue
        sessions.append(evt.session_id)
    if len(sessions) < _DEFAULT_THRESHOLD:
        return []
    return [
        PatternRecord(
            detector="low_confidence",
            severity="low",
            count=len(sessions),
            sample_session_ids=_first_unique(sessions),
            summary=(
                f"{len(sessions)} classifications below {_LOW_CONFIDENCE_CUTOFF:.2f} confidence."
            ),
        )
    ]


def detect_tool_error(events: Iterable[Event]) -> list[PatternRecord]:
    by_tool: dict[str, list[str]] = defaultdict(list)
    last_invoked: dict[str, str] = {}  # session_id → most recent tool name
    for evt in events:
        if evt.type == "tool.invoked":
            tool = str(evt.payload.get("tool", ""))
            if tool:
                last_invoked[evt.session_id] = tool
            continue
        if evt.type != "tool.result":
            continue
        if evt.payload.get("status") == "ok":
            continue
        tool = last_invoked.get(evt.session_id, "<unknown>")
        by_tool[tool].append(evt.session_id)
    out: list[PatternRecord] = []
    for tool, sessions in sorted(by_tool.items()):
        if len(sessions) < _DEFAULT_THRESHOLD:
            continue
        out.append(
            PatternRecord(
                detector="tool_error",
                severity="high",
                count=len(sessions),
                sample_session_ids=_first_unique(sessions),
                summary=f"Tool {tool!r} returned non-ok status {len(sessions)} times.",
            )
        )
    return out


def detect_tier1_unavailable(events: Iterable[Event]) -> list[PatternRecord]:
    sessions: list[str] = []
    for evt in events:
        if evt.type != "pattern.observed":
            continue
        if evt.payload.get("pattern") == "tier1_missing":
            sessions.append(evt.session_id)
    if not sessions:
        return []
    return [
        PatternRecord(
            detector="tier1_unavailable",
            severity="medium",
            count=len(sessions),
            sample_session_ids=_first_unique(sessions),
            summary=(
                "Tier 1 reasoner unavailable — frontier-only skills are degraded."
            ),
        )
    ]


def detect_model_latency(events: Iterable[Event]) -> list[PatternRecord]:
    by_tier: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for evt in events:
        if evt.type != "model.call.end":
            continue
        if evt.payload.get("status") != "ok":
            continue
        latency = _as_int(evt.payload.get("latency_ms"))
        tier = str(evt.payload.get("tier", ""))
        if latency is None or not tier:
            continue
        by_tier[tier].append((latency, evt.session_id))
    out: list[PatternRecord] = []
    for tier, samples in sorted(by_tier.items()):
        if len(samples) < _MIN_LATENCY_SAMPLES:
            continue
        latencies = sorted(s[0] for s in samples)
        p95 = _percentile_95(latencies)
        if p95 < _LATENCY_P95_THRESHOLD_MS:
            continue
        offenders = [sid for lat, sid in samples if lat >= p95]
        out.append(
            PatternRecord(
                detector="model_latency",
                severity="medium",
                count=len(offenders),
                sample_session_ids=_first_unique(offenders),
                summary=(
                    f"Tier {tier!r} p95 latency {p95} ms over {len(samples)} calls."
                ),
            )
        )
    return out


def detect_apply_failed_repeat(events: Iterable[Event]) -> list[PatternRecord]:
    """≥2 non-clean apply verdicts for the same `imp_id` → high signal.

    Reads `governance.decision` events emitted by the Plane 3 audit log.
    A non-clean verdict is any of `applied_test_failed | apply_conflict |
    reverted` — `approve|reject|defer` (human review) and `applied_clean`
    are ignored. One record per offending `imp_id`.
    """
    by_imp: dict[str, list[str]] = defaultdict(list)
    for evt in events:
        if evt.type != "governance.decision":
            continue
        verdict = str(evt.payload.get("verdict", ""))
        if verdict not in _NON_CLEAN_APPLY_VERDICTS:
            continue
        imp_id = str(evt.payload.get("imp_id", ""))
        if not imp_id:
            continue
        by_imp[imp_id].append(evt.session_id)
    out: list[PatternRecord] = []
    for imp_id, sessions in sorted(by_imp.items()):
        if len(sessions) < _APPLY_FAIL_THRESHOLD:
            continue
        out.append(
            PatternRecord(
                detector="apply_failed_repeat",
                severity="high",
                count=len(sessions),
                sample_session_ids=_first_unique(sessions),
                summary=(
                    f"{imp_id} failed apply {len(sessions)} times — "
                    "investigate scope or test fragility."
                ),
            )
        )
    return out


def detect_harness_refused_repeat(events: Iterable[Event]) -> list[PatternRecord]:
    """≥2 `harness.refused` events for the same `imp_id` → medium signal.

    Refusal = the harness rejected its own draft (scope/diff touches a
    canonical file). Repeated refusals for the same task suggest the
    scope description itself is wrong — a human needs to retitle or
    split the task.
    """
    by_imp: dict[str, list[str]] = defaultdict(list)
    for evt in events:
        if evt.type != "harness.refused":
            continue
        imp_id = str(evt.payload.get("imp_id", ""))
        if not imp_id:
            continue
        by_imp[imp_id].append(evt.session_id)
    out: list[PatternRecord] = []
    for imp_id, sessions in sorted(by_imp.items()):
        if len(sessions) < _REFUSE_THRESHOLD:
            continue
        out.append(
            PatternRecord(
                detector="harness_refused_repeat",
                severity="medium",
                count=len(sessions),
                sample_session_ids=_first_unique(sessions),
                summary=(
                    f"{imp_id} refused {len(sessions)} times — task scope likely touches canon."
                ),
            )
        )
    return out


def detect_context_mode_helps(events: Iterable[Event]) -> list[PatternRecord]:
    """Context-mode draft followed by `applied_clean` → low-but-positive signal.

    Pairs a `pattern.observed: harness_with_context` event with a later
    `governance.decision: applied_clean` for the same `imp_id`. One
    record per such `imp_id`. This detector celebrates wins — it tells
    the operator which tasks benefited from `--with-context` so they
    can keep using it for similar work.
    """
    context_imps: set[str] = set()
    sessions_by_imp: dict[str, list[str]] = defaultdict(list)
    for evt in events:
        if evt.type == "pattern.observed":
            if evt.payload.get("pattern") != "harness_with_context":
                continue
            imp_id = str(evt.payload.get("imp_id", ""))
            if imp_id:
                context_imps.add(imp_id)
                sessions_by_imp[imp_id].append(evt.session_id)
            continue
        if evt.type != "governance.decision":
            continue
        if evt.payload.get("verdict") != "applied_clean":
            continue
        imp_id = str(evt.payload.get("imp_id", ""))
        if imp_id and imp_id in context_imps:
            sessions_by_imp[imp_id].append(evt.session_id)
            sessions_by_imp[imp_id].append("__hit__")  # marker
    out: list[PatternRecord] = []
    for imp_id in sorted(context_imps):
        sessions = sessions_by_imp.get(imp_id, [])
        if "__hit__" not in sessions:
            continue
        clean_sessions = [s for s in sessions if s != "__hit__"]
        out.append(
            PatternRecord(
                detector="context_mode_helps",
                severity="low",
                count=1,
                sample_session_ids=_first_unique(clean_sessions),
                summary=(
                    f"{imp_id} drafted with context-mode and applied clean — "
                    "keep using --with-context here."
                ),
            )
        )
    return out


def _first_unique(sessions: list[str]) -> list[str]:
    counter = Counter(sessions)
    seen: list[str] = []
    for sid, _count in counter.most_common():
        if sid not in seen:
            seen.append(sid)
        if len(seen) >= _MAX_SAMPLE_SESSIONS:
            break
    return seen


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


_QUANTILES_MIN_SAMPLES = 2


def _percentile_95(sorted_values: list[int]) -> int:
    """Inclusive p95. `quantiles` requires ≥ 2 samples; caller enforces."""
    if len(sorted_values) < _QUANTILES_MIN_SAMPLES:
        return sorted_values[0]
    cuts = quantiles(sorted_values, n=20, method="inclusive")
    return int(cuts[-1])
