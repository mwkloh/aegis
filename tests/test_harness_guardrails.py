"""Shared deterministic guardrails (spec: 2026-10-06-harness-ablation-design.md, M3-M5)."""
from __future__ import annotations

import pytest

from runtime.harness.contract import ToolIntent, ToolResult

pytestmark = pytest.mark.unit


def _search(result: str) -> tuple[ToolIntent, ToolResult]:
    return (
        ToolIntent(tool="files_search", args={"pattern": "*CT-001*"}, skill_id="s"),
        ToolResult(status="ok", payload={"result": result}),
    )


def test_default_guardrails_are_todays_planner_behaviour() -> None:
    from runtime.harness.guardrails import Guardrails

    g = Guardrails()
    assert (g.autofill, g.claim_gate, g.evidence, g.plan_lock) == (True, True, True, True)
    assert g.synthesis is False


def test_with_evidence_without_a_summary_is_just_the_results() -> None:
    """Native mode: a model that spent its budget thinking leaves no text of its own."""
    from runtime.harness.guardrails import with_evidence

    out = with_evidence(
        "", [_search("/tmp/n/CT-001-notes.md")], per_result_chars=3500, turn_budget_chars=8000
    )
    assert out.startswith("Results:\n1. files_search(")
    assert "/tmp/n/CT-001-notes.md" in out


def test_with_evidence_appends_after_the_summary() -> None:
    from runtime.harness.guardrails import with_evidence

    out = with_evidence(
        "Found it.", [_search("/tmp/a.md")], per_result_chars=3500, turn_budget_chars=8000
    )
    assert out.startswith("Found it.\n\nResults:\n")


def test_autofill_replaces_a_wrong_guess_with_the_search_match() -> None:
    from runtime.harness.guardrails import autofill_path

    guess = ToolIntent(tool="files_read", args={"path": "/tmp/n/ct001.md"}, skill_id="s")
    fixed = autofill_path(guess, [_search("/tmp/n/CT-001-notes.md")])
    assert fixed.args["path"] == "/tmp/n/CT-001-notes.md"


def test_the_dispatcher_uses_the_shared_functions() -> None:
    """One implementation for both harness modes, not two copies that can drift."""
    from runtime.chat.telegram import harness_dispatcher as hd
    from runtime.harness import guardrails as g

    assert hd._render_chain is g.render_chain
    assert hd._history_verified_tools is g.history_verified_tools
    assert hd._autofill_path_from_prior_search is g.autofill_path
    assert hd.DESTRUCTIVE_TOOLS is g.DESTRUCTIVE_TOOLS
