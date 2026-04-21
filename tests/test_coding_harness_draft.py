"""Pydantic envelope for harness drafts."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from runtime.coding_harness.draft import Draft

pytestmark = pytest.mark.unit


def _base(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "ct_id": "CT-001",
        "imp_id": "IMP-a86b087a",
        "model": "openrouter:minimax/minimax-m2.7",
        "summary": "Add fallback retrieval.",
        "unified_diff": "--- a/x\n+++ b/x\n@@\n-1\n+2\n",
        "test_notes": "n/a",
        "rollback": "git revert HEAD",
        "drafted_at": datetime(2026, 4, 18, 12, 7, tzinfo=UTC),
        "status": "ok",
    }
    base.update(kwargs)
    return base


def test_minimum_fields_accept_ok_status() -> None:
    d = Draft(**_base())  # type: ignore[arg-type]
    assert d.status == "ok"
    assert d.reason == ""


def test_stub_status_keeps_reason() -> None:
    d = Draft(**_base(status="stub", unified_diff="", reason="LLM unavailable"))  # type: ignore[arg-type]
    assert d.status == "stub"
    assert d.reason == "LLM unavailable"


def test_refused_status_allowed() -> None:
    d = Draft(**_base(status="refused", unified_diff="", reason="canon scope"))  # type: ignore[arg-type]
    assert d.status == "refused"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Draft(**_base(extra="nope"))  # type: ignore[arg-type]


def test_diff_overflow_rejected() -> None:
    with pytest.raises(ValidationError):
        Draft(**_base(unified_diff="x" * 32_001))  # type: ignore[arg-type]


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Draft(**_base(status="weird"))  # type: ignore[arg-type]
