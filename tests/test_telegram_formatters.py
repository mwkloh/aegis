"""Phase 7 §4.3 — `render_decisions` + `/decisions` handler contract.

Pins:

* Empty log → explicit "no decisions" string, not blank.
* Tail respects the requested N, clamps to MAX, floors to default
  on `N <= 0`.
* Rationale falls back to `—` when empty.
* `/decisions <non-int>` returns a usage hint, never an exception.
* `/decisions` reads only the on-disk DECISIONS.md; the handler is
  a pure function of workspace state.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.telegram import (
    MAX_DECISIONS_TAIL,
    IncomingMessage,
    ParsedCommand,
    decisions_handler,
    render_decisions,
)
from runtime.improvement.decisions import Decision, record_decision

pytestmark = pytest.mark.unit


def _decision(imp: str, verdict: str = "approve", rationale: str = "") -> Decision:
    return Decision(
        imp_id=imp,
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale,
        decided_at=datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
        supersedes=None,
    )


# --- render_decisions ---------------------------------------------------


def test_render_empty_log() -> None:
    assert render_decisions([]) == "No decisions recorded yet."


def test_render_tail_defaults() -> None:
    decisions = [_decision(f"IMP-{i:08x}") for i in range(25)]
    out = render_decisions(decisions)
    # default tail = 10
    assert out.startswith("Last 10 decision(s):")
    assert out.count("\n•") == 10


def test_render_tail_explicit() -> None:
    decisions = [_decision(f"IMP-{i:08x}") for i in range(20)]
    out = render_decisions(decisions, tail=3)
    assert out.startswith("Last 3 decision(s):")
    assert out.count("\n•") == 3


def test_render_tail_clamped_to_max() -> None:
    decisions = [_decision(f"IMP-{i:08x}") for i in range(100)]
    out = render_decisions(decisions, tail=9999)
    # MAX_DECISIONS_TAIL-bounded
    assert out.startswith(f"Last {MAX_DECISIONS_TAIL} decision(s):")


def test_render_tail_nonpositive_floors_to_default() -> None:
    decisions = [_decision(f"IMP-{i:08x}") for i in range(20)]
    out = render_decisions(decisions, tail=0)
    assert out.startswith("Last 10 decision(s):")
    out2 = render_decisions(decisions, tail=-5)
    assert out2.startswith("Last 10 decision(s):")


def test_render_empty_rationale_shows_dash() -> None:
    out = render_decisions([_decision("IMP-abcdef01", rationale="")])
    assert "— approve — —" in out


# --- /decisions handler -------------------------------------------------


def test_decisions_handler_reads_disk(tmp_path: Path) -> None:
    record_decision(
        tmp_path,
        imp_id="IMP-abcdef01",
        verdict="approve",
        rationale="ship it",
    )
    handler = decisions_handler(tmp_path)
    msg = IncomingMessage(chat_id=111, user_id=1, text="/decisions")
    out = handler(msg, ParsedCommand(name="/decisions", args=()))
    assert "IMP-abcdef01" in out
    assert "approve" in out
    assert "ship it" in out


def test_decisions_handler_tail_arg(tmp_path: Path) -> None:
    for i in range(5):
        record_decision(
            tmp_path,
            imp_id=f"IMP-aaaaaa{i:02x}",
            verdict="approve",
            rationale=f"r{i}",
        )
    handler = decisions_handler(tmp_path)
    msg = IncomingMessage(chat_id=111, user_id=1, text="/decisions 2")
    out = handler(msg, ParsedCommand(name="/decisions", args=("2",)))
    assert out.startswith("Last 2 decision(s):")
    assert out.count("\n•") == 2


def test_decisions_handler_non_int_arg_returns_usage(tmp_path: Path) -> None:
    handler = decisions_handler(tmp_path)
    msg = IncomingMessage(chat_id=111, user_id=1, text="/decisions zebra")
    out = handler(msg, ParsedCommand(name="/decisions", args=("zebra",)))
    assert "expects an integer" in out
    assert "Usage:" in out


def test_decisions_handler_empty_log(tmp_path: Path) -> None:
    handler = decisions_handler(tmp_path)
    msg = IncomingMessage(chat_id=111, user_id=1, text="/decisions")
    out = handler(msg, ParsedCommand(name="/decisions", args=()))
    assert out == "No decisions recorded yet."
