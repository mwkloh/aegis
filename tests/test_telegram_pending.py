"""Phase 7 §4.3 — `/pending` and `/proposal` handler contracts.

Pins:

* `/pending` lists loaded proposals joined against DECISIONS.md:
    - proposals with no decision → pending
    - proposals whose latest verdict is `defer` → still pending
      (annotated as "previously deferred")
    - proposals whose latest verdict is `approve`/`reject`/applier
      verdicts → dropped
* Empty corpus → explicit "No proposals drafted." reply.
* `/proposal <id>` accepts case-insensitive `IMP-<8 hex>`; garbage
  returns a usage hint; unknown id returns a "not found" reply.
* Proposal render includes latest decision annotation when one
  exists so the operator sees reject/defer history inline.
* Handlers never raise; they read the on-disk plane-3 state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.chat.telegram import (
    IncomingMessage,
    ParsedCommand,
    pending_handler,
    pending_proposals,
    proposal_handler,
    render_pending,
    render_proposal,
)
from runtime.improvement.decisions import load_decisions, record_decision
from runtime.improvement.proposal_loader import derive_imp_id, load_proposals

pytestmark = pytest.mark.unit


def _write_proposals_md(
    workspace: Path,
    *,
    run: str = "2026-04-19T12:00Z",
    entries: list[tuple[str, str, str, str, list[str]]],
) -> None:
    """Write a synthetic PROPOSALS.md file.

    entries = [(detector, risk, change, rationale, affected), ...]
    """
    reflection = workspace / "reflection"
    reflection.mkdir(parents=True, exist_ok=True)
    body = [f"## {run} — sessions=1"]
    for detector, risk, change, rationale, affected in entries:
        body.append(f"### P-001 — {detector} (risk: {risk})")
        body.append(f"- **Affected:** {', '.join(affected) if affected else '—'}")
        body.append(f"- **Change:** {change}")
        body.append(f"- **Rationale:** {rationale}")
        body.append("")
    (reflection / "PROPOSALS.md").write_text("\n".join(body), encoding="utf-8")


def _msg(text: str) -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text=text)


# --- /pending handler ---------------------------------------------------


def test_pending_empty_corpus(tmp_path: Path) -> None:
    handler = pending_handler(tmp_path)
    out = handler(_msg("/pending"), ParsedCommand(name="/pending", args=()))
    assert out == "No proposals drafted."


def test_pending_lists_undecided(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[
            ("unknown_intent", "low", "Add logging to classifier", "improve trail", []),
            ("tool_error", "medium", "Retry once on 429", "ratelimit", ["client.py"]),
        ],
    )
    handler = pending_handler(tmp_path)
    out = handler(_msg("/pending"), ParsedCommand(name="/pending", args=()))
    assert "Pending proposals" in out
    assert "unknown_intent" in out
    assert "tool_error" in out
    assert "[low]" in out
    assert "[medium]" in out


def test_pending_drops_approved(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[
            ("d1", "low", "c1", "r1", []),
            ("d2", "high", "c2", "r2", []),
        ],
    )
    # Approve the first one.
    imp_approved = derive_imp_id("d1", [], "c1")
    record_decision(
        tmp_path, imp_id=imp_approved, verdict="approve", rationale="shipped"
    )
    handler = pending_handler(tmp_path)
    out = handler(_msg("/pending"), ParsedCommand(name="/pending", args=()))
    assert imp_approved not in out
    assert "d2" in out


def test_pending_keeps_deferred(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[("d1", "low", "c1", "r1", [])],
    )
    imp_id = derive_imp_id("d1", [], "c1")
    record_decision(
        tmp_path, imp_id=imp_id, verdict="defer", rationale="later"
    )
    handler = pending_handler(tmp_path)
    out = handler(_msg("/pending"), ParsedCommand(name="/pending", args=()))
    assert imp_id in out
    assert "previously deferred" in out


def test_pending_proposals_unit_filter() -> None:
    """`pending_proposals` is the pure-function core — exercise directly."""
    # No proposals → no pending
    assert pending_proposals([], {}) == []


# --- /proposal handler --------------------------------------------------


def test_proposal_usage_without_args(tmp_path: Path) -> None:
    handler = proposal_handler(tmp_path)
    out = handler(_msg("/proposal"), ParsedCommand(name="/proposal", args=()))
    assert "Usage" in out
    assert "IMP-" in out


def test_proposal_bad_id_format(tmp_path: Path) -> None:
    handler = proposal_handler(tmp_path)
    out = handler(
        _msg("/proposal foo"),
        ParsedCommand(name="/proposal", args=("foo",)),
    )
    assert "expects an IMP-<8 hex>" in out


def test_proposal_unknown_id(tmp_path: Path) -> None:
    handler = proposal_handler(tmp_path)
    out = handler(
        _msg("/proposal IMP-deadbeef"),
        ParsedCommand(name="/proposal", args=("IMP-deadbeef",)),
    )
    assert "not found" in out.lower()
    assert "IMP-deadbeef" in out


def test_proposal_found_undecided(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[("unknown_intent", "low", "Add logging", "trail", ["cli.py"])],
    )
    imp_id = derive_imp_id("unknown_intent", ["cli.py"], "Add logging")
    handler = proposal_handler(tmp_path)
    out = handler(
        _msg(f"/proposal {imp_id}"),
        ParsedCommand(name="/proposal", args=(imp_id,)),
    )
    assert imp_id in out
    assert "unknown_intent" in out
    assert "risk: low" in out
    assert "cli.py" in out
    assert "drafted" in out  # no decision yet


def test_proposal_annotates_latest_decision(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[("d1", "medium", "c1", "r1", [])],
    )
    imp_id = derive_imp_id("d1", [], "c1")
    record_decision(
        tmp_path, imp_id=imp_id, verdict="reject", rationale="no thanks"
    )
    handler = proposal_handler(tmp_path)
    out = handler(
        _msg(f"/proposal {imp_id.lower()}"),  # case-insensitive
        ParsedCommand(name="/proposal", args=(imp_id.lower(),)),
    )
    assert "reject" in out
    assert "no thanks" in out


# --- formatter edge cases ----------------------------------------------


def test_render_pending_caps_at_max_rows(tmp_path: Path) -> None:
    entries = [
        (f"det{i}", "low", f"change {i}", f"r{i}", []) for i in range(30)
    ]
    _write_proposals_md(tmp_path, entries=entries)
    proposals = load_proposals(tmp_path)
    # all proposals are unique (detector varies), none decided
    out = render_pending(proposals, {})
    # "Pending proposals (25 of N)" — MAX_PENDING_ROWS cap
    assert "Pending proposals (25 of " in out


def test_render_proposal_missing_uses_label() -> None:
    out = render_proposal(None, None, imp_id="IMP-ffffffff")
    assert "IMP-ffffffff" in out
    assert "not found" in out


def test_render_proposal_empty_rationale_renders_dash(tmp_path: Path) -> None:
    _write_proposals_md(
        tmp_path,
        entries=[("d1", "low", "change only", "", [])],
    )
    proposals = load_proposals(tmp_path)
    assert proposals
    out = render_proposal(proposals[0], None)
    assert "Rationale: —" in out


def test_proposal_handler_survives_missing_file(tmp_path: Path) -> None:
    """No PROPOSALS.md on disk → 'not found', not an exception."""
    # no file written
    handler = proposal_handler(tmp_path)
    out = handler(
        _msg("/proposal IMP-aaaaaaaa"),
        ParsedCommand(name="/proposal", args=("IMP-aaaaaaaa",)),
    )
    assert "not found" in out.lower()
    # Confirm load_decisions also tolerates missing file
    assert load_decisions(tmp_path) == []
