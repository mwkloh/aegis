"""`ApplyOutcome` envelope validation."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from runtime.coding_harness.apply_outcome import ApplyOutcome

pytestmark = pytest.mark.unit


def _outcome(**kwargs: object) -> ApplyOutcome:
    base: dict[str, object] = {
        "ct_id": "CT-001",
        "imp_id": "IMP-a86b087a",
        "verdict": "applied_clean",
        "branch": "aegis/CT-001-a86b087a",
        "patch_path": "CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md",
        "tests_exit_code": 0,
        "tests_duration_s": 41.2,
        "tests_stdout_tail": "98 passed, 0 failed in 41.2s",
        "applied_at": datetime(2026, 4, 19, 8, 30, tzinfo=UTC),
    }
    base.update(kwargs)
    return ApplyOutcome(**base)  # type: ignore[arg-type]


def test_minimal_valid_outcome() -> None:
    o = _outcome()
    assert o.verdict == "applied_clean"
    assert o.tests_exit_code == 0


@pytest.mark.parametrize(
    "verdict",
    ["applied_clean", "applied_test_failed", "apply_conflict", "precondition_failed"],
)
def test_all_verdicts_accepted(verdict: str) -> None:
    o = _outcome(verdict=verdict)
    assert o.verdict == verdict


def test_unknown_verdict_rejected() -> None:
    with pytest.raises(ValidationError):
        _outcome(verdict="exploded")


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ApplyOutcome(  # type: ignore[call-arg]
            ct_id="CT-001",
            imp_id="IMP-a86b087a",
            verdict="applied_clean",
            applied_at=datetime(2026, 4, 19, tzinfo=UTC),
            wat="nope",
        )


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        _outcome(tests_duration_s=-1.0)


def test_stdout_tail_capped_at_8kb() -> None:
    """Caller is responsible for truncating before construction."""
    with pytest.raises(ValidationError):
        _outcome(tests_stdout_tail="x" * 8193)


def test_precondition_failed_allows_no_test_fields() -> None:
    o = _outcome(
        verdict="precondition_failed",
        reason="working tree dirty",
        branch="",
        patch_path="",
        tests_exit_code=None,
        tests_duration_s=None,
        tests_stdout_tail="",
    )
    assert o.tests_exit_code is None
    assert o.tests_duration_s is None


def test_outcome_is_frozen() -> None:
    o = _outcome()
    with pytest.raises(ValidationError):
        o.ct_id = "CT-999"  # type: ignore[misc]
