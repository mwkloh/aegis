"""Result-file rendering and append-only behaviour."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness.apply_outcome import ApplyOutcome
from runtime.coding_harness.patch_writer import diffs_dir
from runtime.coding_harness.result_writer import (
    existing_results_for,
    latest_result_for,
    result_filename,
    write_result,
)

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
        "tests_stdout_tail": "98 passed, 0 failed",
        "applied_at": datetime(2026, 4, 19, 8, 30, tzinfo=UTC),
    }
    base.update(kwargs)
    return ApplyOutcome(**base)  # type: ignore[arg-type]


def test_filename_format() -> None:
    name = result_filename(_outcome())
    assert name == "CT-001__IMP-a86b087a__2026-04-19T0830Z.result.md"


def test_writes_alongside_patch_in_diffs_dir(tmp_path: Path) -> None:
    target = write_result(tmp_path, _outcome())
    assert target.exists()
    assert target.parent.name == "diffs"
    text = target.read_text(encoding="utf-8")
    assert "verdict: applied_clean" in text
    assert "test suite passed" in text
    assert "git diff main...aegis/CT-001-a86b087a" in text


def test_applied_test_failed_renders_failure_guidance(tmp_path: Path) -> None:
    target = write_result(
        tmp_path,
        _outcome(
            verdict="applied_test_failed",
            reason="2 pytest failures",
            tests_exit_code=1,
            tests_duration_s=37.4,
            tests_stdout_tail="FAILED tests/test_x.py::test_y",
        ),
    )
    text = target.read_text(encoding="utf-8")
    assert "verdict: applied_test_failed" in text
    assert "FAILED tests/test_x.py::test_y" in text
    assert "--force" in text


def test_apply_conflict_renders_inspection_guidance(tmp_path: Path) -> None:
    target = write_result(
        tmp_path,
        _outcome(
            verdict="apply_conflict",
            reason="error: patch failed: runtime/foo.py:12",
            tests_exit_code=None,
            tests_duration_s=None,
            tests_stdout_tail="",
        ),
    )
    text = target.read_text(encoding="utf-8")
    assert "verdict: apply_conflict" in text
    assert "error: patch failed" in text
    assert "git status" in text
    assert "Tests:** not run" in text


def test_precondition_failed_renders_no_branch(tmp_path: Path) -> None:
    target = write_result(
        tmp_path,
        _outcome(
            verdict="precondition_failed",
            reason="working tree dirty",
            branch="",
            patch_path="",
            tests_exit_code=None,
            tests_duration_s=None,
            tests_stdout_tail="",
        ),
    )
    text = target.read_text(encoding="utf-8")
    assert "verdict: precondition_failed" in text
    assert "working tree dirty" in text
    assert "Branch:** —" in text


def test_collision_in_same_minute_writes_suffixed(tmp_path: Path) -> None:
    first = write_result(tmp_path, _outcome())
    second = write_result(tmp_path, _outcome())
    assert first != second
    assert first.exists()
    assert second.exists()
    assert second.name.endswith("-02.result.md")


def test_latest_result_for_returns_most_recent(tmp_path: Path) -> None:
    earlier = _outcome(applied_at=datetime(2026, 4, 19, 8, 30, tzinfo=UTC))
    later = _outcome(applied_at=datetime(2026, 4, 19, 11, 0, tzinfo=UTC))
    write_result(tmp_path, earlier)
    write_result(tmp_path, later)
    latest = latest_result_for(tmp_path, "CT-001")
    assert latest is not None
    assert "1100Z" in latest.name


def test_latest_result_for_returns_none_when_empty(tmp_path: Path) -> None:
    assert latest_result_for(tmp_path, "CT-999") is None


def test_existing_results_does_not_pick_up_patch_files(tmp_path: Path) -> None:
    """Patch and result files share a directory — keep them separate."""
    write_result(tmp_path, _outcome())
    # Drop a stray .patch.md to confirm the glob is result-specific.
    fake_patch = diffs_dir(tmp_path) / "CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md"
    fake_patch.write_text("noop", encoding="utf-8")
    results = existing_results_for(tmp_path, "CT-001")
    assert len(results) == 1
    assert results[0].name.endswith(".result.md")
