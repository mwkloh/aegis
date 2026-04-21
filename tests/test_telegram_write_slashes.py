"""Phase 7 §4.2 — `/approve` `/reject` `/defer` write-slash contracts.

Pins:

* Usage replies when args missing; `IMP-<8 hex>` canonicalization with
  a clear error on bad input.
* Validation gate: an imp_id without either a drafted proposal or a
  prior decision on file returns "Proposal IMP-... not found." — we
  refuse to create decisions for bare-string imp_ids.
* Idempotent: same verdict twice → "already {verdict}" reply, no
  second `DECISIONS.md` row written.
* Supersede: different verdict → new row with `supersedes` pointing
  at the prior decided_at.
* Case-insensitive imp_id accepted.
* Multi-token rationales joined with spaces (shlex quotes preserved
  by the dispatcher).
* `EventStream` injection causes a `governance.decision` event per
  successful write (but not for the idempotent no-op).
* Re-deciding on an imp with prior decision but no current proposal
  still succeeds — prior decisions keep an imp "known".
* `build_write_handlers` wires all three slashes with shared clock +
  events.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.telegram import (
    IncomingMessage,
    ParsedCommand,
    approve_handler,
    build_write_handlers,
    defer_handler,
    reject_handler,
)
from runtime.events import EventStream
from runtime.improvement.decisions import (
    decisions_path,
    latest_by_imp,
    load_decisions,
    record_decision,
)
from runtime.improvement.proposal_loader import derive_imp_id

pytestmark = pytest.mark.unit


def _msg(text: str) -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text=text)


def _write_proposals_md(
    workspace: Path,
    *,
    detector: str = "unknown_intent",
    change: str = "Add logging",
    affected: list[str] | None = None,
    risk: str = "low",
) -> str:
    """Write a minimal PROPOSALS.md; return the derived imp_id."""
    reflection = workspace / "reflection"
    reflection.mkdir(parents=True, exist_ok=True)
    aff = affected or []
    body = [
        "## 2026-04-19T12:00Z — sessions=1",
        f"### P-001 — {detector} (risk: {risk})",
        f"- **Affected:** {', '.join(aff) if aff else '—'}",
        f"- **Change:** {change}",
        "- **Rationale:** bootstrap",
    ]
    (reflection / "PROPOSALS.md").write_text("\n".join(body), encoding="utf-8")
    return derive_imp_id(detector, aff, change)


def _fixed_clock(dt: datetime):
    return lambda: dt


# --- usage / validation ------------------------------------------------


def test_approve_usage_without_args(tmp_path: Path) -> None:
    handler = approve_handler(tmp_path)
    out = handler(_msg("/approve"), ParsedCommand(name="/approve", args=()))
    assert "Usage" in out
    assert "/approve IMP-" in out


def test_reject_usage_without_args(tmp_path: Path) -> None:
    handler = reject_handler(tmp_path)
    out = handler(_msg("/reject"), ParsedCommand(name="/reject", args=()))
    assert "Usage" in out
    assert "/reject IMP-" in out


def test_defer_usage_without_args(tmp_path: Path) -> None:
    handler = defer_handler(tmp_path)
    out = handler(_msg("/defer"), ParsedCommand(name="/defer", args=()))
    assert "Usage" in out
    assert "/defer IMP-" in out


def test_approve_bad_imp_format(tmp_path: Path) -> None:
    handler = approve_handler(tmp_path)
    out = handler(
        _msg("/approve foo"),
        ParsedCommand(name="/approve", args=("foo",)),
    )
    assert "expects an IMP-<8 hex>" in out
    assert "'foo'" in out
    # No DECISIONS.md written
    assert not decisions_path(tmp_path).is_file()


def test_approve_unknown_imp_id_rejected(tmp_path: Path) -> None:
    """No proposal + no prior decision → not found, no write."""
    handler = approve_handler(tmp_path)
    out = handler(
        _msg("/approve IMP-deadbeef"),
        ParsedCommand(name="/approve", args=("IMP-deadbeef",)),
    )
    assert "not found" in out.lower()
    assert "IMP-deadbeef" in out
    assert not decisions_path(tmp_path).is_file()


# --- happy paths -------------------------------------------------------


def test_approve_records_decision(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    clock = _fixed_clock(datetime(2026, 4, 19, 14, 30, tzinfo=UTC))
    handler = approve_handler(tmp_path, clock=clock)
    out = handler(
        _msg(f"/approve {imp_id} ship it"),
        ParsedCommand(name="/approve", args=(imp_id, "ship", "it")),
    )
    assert f"Recorded approve for {imp_id}" in out
    assert "2026-04-19 14:30Z" in out
    assert "ship it" in out
    decisions = load_decisions(tmp_path)
    assert len(decisions) == 1
    assert decisions[0].imp_id == imp_id
    assert decisions[0].verdict == "approve"
    assert decisions[0].rationale == "ship it"
    assert decisions[0].supersedes is None


def test_reject_records_decision(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    handler = reject_handler(tmp_path)
    out = handler(
        _msg(f"/reject {imp_id} too risky"),
        ParsedCommand(name="/reject", args=(imp_id, "too", "risky")),
    )
    assert f"Recorded reject for {imp_id}" in out
    assert load_decisions(tmp_path)[0].verdict == "reject"


def test_defer_records_decision(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    handler = defer_handler(tmp_path)
    out = handler(
        _msg(f"/defer {imp_id}"),
        ParsedCommand(name="/defer", args=(imp_id,)),
    )
    assert f"Recorded defer for {imp_id}" in out
    assert "Rationale: —" in out  # no rationale provided
    d = load_decisions(tmp_path)[0]
    assert d.verdict == "defer"
    # On-disk round-trip keeps the em-dash sentinel for "no rationale".
    assert d.rationale in ("", "—")


# --- idempotency & supersede ------------------------------------------


def test_same_verdict_twice_is_idempotent(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    clock = _fixed_clock(datetime(2026, 4, 19, 10, 0, tzinfo=UTC))
    handler = approve_handler(tmp_path, clock=clock)
    first = handler(
        _msg(f"/approve {imp_id} one"),
        ParsedCommand(name="/approve", args=(imp_id, "one")),
    )
    assert "Recorded approve" in first
    second = handler(
        _msg(f"/approve {imp_id} two"),
        ParsedCommand(name="/approve", args=(imp_id, "two")),
    )
    assert "already approve" in second
    assert "No change recorded" in second
    # Still exactly one decision row on disk
    assert len(load_decisions(tmp_path)) == 1


def test_different_verdict_supersedes_prior(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    first_clock = _fixed_clock(datetime(2026, 4, 19, 10, 0, tzinfo=UTC))
    second_clock = _fixed_clock(datetime(2026, 4, 19, 12, 0, tzinfo=UTC))
    defer_h = defer_handler(tmp_path, clock=first_clock)
    approve_h = approve_handler(tmp_path, clock=second_clock)
    defer_h(
        _msg(f"/defer {imp_id}"),
        ParsedCommand(name="/defer", args=(imp_id,)),
    )
    out = approve_h(
        _msg(f"/approve {imp_id} now ready"),
        ParsedCommand(name="/approve", args=(imp_id, "now", "ready")),
    )
    assert "Supersedes: 2026-04-19T10:00Z" in out
    decisions = load_decisions(tmp_path)
    assert [d.verdict for d in decisions] == ["defer", "approve"]
    latest = latest_by_imp(decisions)[imp_id]
    assert latest.verdict == "approve"
    assert latest.supersedes == "2026-04-19T10:00Z"


# --- misc behaviors ----------------------------------------------------


def test_case_insensitive_imp_id(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    handler = approve_handler(tmp_path)
    out = handler(
        _msg(f"/approve {imp_id.lower()}"),
        ParsedCommand(name="/approve", args=(imp_id.lower(),)),
    )
    # Canonical (lowercase) form lands in DECISIONS.md
    assert f"Recorded approve for {imp_id}" in out
    assert load_decisions(tmp_path)[0].imp_id == imp_id


def test_prior_decision_keeps_imp_known_without_proposal(tmp_path: Path) -> None:
    """Re-deciding after a prior decision works even if PROPOSALS.md was deleted."""
    imp_id = "IMP-abcd1234"
    record_decision(
        tmp_path,
        imp_id=imp_id,
        verdict="defer",
        rationale="park it",
        when=datetime(2026, 4, 19, 8, 0, tzinfo=UTC),
    )
    # No PROPOSALS.md ever written — only the prior decision exists.
    handler = reject_handler(
        tmp_path, clock=_fixed_clock(datetime(2026, 4, 19, 11, 0, tzinfo=UTC))
    )
    out = handler(
        _msg(f"/reject {imp_id} killed"),
        ParsedCommand(name="/reject", args=(imp_id, "killed")),
    )
    assert "Recorded reject" in out
    assert load_decisions(tmp_path)[-1].verdict == "reject"


def test_event_emission_on_record(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    events = EventStream(sessions_dir)
    handler = approve_handler(tmp_path, events=events)
    handler(
        _msg(f"/approve {imp_id}"),
        ParsedCommand(name="/approve", args=(imp_id,)),
    )
    log = events.path.read_text(encoding="utf-8")
    assert '"type": "governance.decision"' in log
    assert '"verdict": "approve"' in log
    assert f'"imp_id": "{imp_id}"' in log


def test_idempotent_no_op_emits_no_event(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    events = EventStream(sessions_dir)
    handler = approve_handler(tmp_path, events=events)
    handler(
        _msg(f"/approve {imp_id}"),
        ParsedCommand(name="/approve", args=(imp_id,)),
    )
    # Baseline: one event emitted from first write
    baseline = events.path.read_text(encoding="utf-8").count(
        '"type": "governance.decision"'
    )
    # Repeat the same verdict
    handler(
        _msg(f"/approve {imp_id}"),
        ParsedCommand(name="/approve", args=(imp_id,)),
    )
    after = events.path.read_text(encoding="utf-8").count(
        '"type": "governance.decision"'
    )
    assert after == baseline  # no new event for the no-op


def test_build_write_handlers_registers_three_slashes(tmp_path: Path) -> None:
    handlers = build_write_handlers(tmp_path)
    assert set(handlers) == {"/approve", "/reject", "/defer"}


def test_build_write_handlers_shares_clock_and_events(tmp_path: Path) -> None:
    imp_id = _write_proposals_md(tmp_path)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    events = EventStream(sessions_dir)
    clock = _fixed_clock(datetime(2026, 4, 19, 9, 0, tzinfo=UTC))
    handlers = build_write_handlers(tmp_path, clock=clock, events=events)
    handlers["/defer"](
        _msg(f"/defer {imp_id}"),
        ParsedCommand(name="/defer", args=(imp_id,)),
    )
    handlers["/approve"](
        _msg(f"/approve {imp_id}"),
        ParsedCommand(name="/approve", args=(imp_id,)),
    )
    # Both went through the same injected clock — decided_at matches.
    decisions = load_decisions(tmp_path)
    assert len(decisions) == 2
    assert all(d.decided_at.strftime("%H:%M") == "09:00" for d in decisions)
    # Events stream accumulated both writes.
    log = events.path.read_text(encoding="utf-8")
    assert log.count('"type": "governance.decision"') == 2
