"""Verdict tests for ``apply_patch`` using a scripted runner.

No real subprocess. Each test scripts the responses ``git`` and
``make`` would give and asserts the resulting ``ApplyOutcome``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness.applier import (
    GitResult,
    ScriptedRunner,
    apply_patch,
    has,
)

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 19, 8, 30, tzinfo=UTC)


def _clock() -> datetime:
    return _NOW


_PATCH_FILENAME = "CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md"
_BRANCH = "aegis/CT-001-a86b087a"

_OK_PATCH_TEXT = """# CT-001 — IMP-a86b087a (status: ok)

## Summary
Adds a thing.

## Unified diff
```diff
diff --git a/runtime/foo.py b/runtime/foo.py
--- a/runtime/foo.py
+++ b/runtime/foo.py
@@ -1,2 +1,3 @@
 def f():
+    return 1
     pass
```

## Test notes
—

## Rollback
—
"""


def _write_patch(tmp_path: Path, text: str = _OK_PATCH_TEXT) -> Path:
    p = tmp_path / _PATCH_FILENAME
    p.write_text(text, encoding="utf-8")
    return p


def _green_preconditions() -> list[tuple[object, GitResult]]:
    """Script entries that make preconditions + branch-availability pass."""
    return [
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, "")),
        (has("rev-parse", "--abbrev-ref"), GitResult(0, "feature/work\n")),
        (has("rev-parse", "--verify"), GitResult(1, "", "fatal: not a ref")),
    ]


# --- happy path --------------------------------------------------------------


def test_applied_clean_happy_path(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, "98 passed in 41.2s\n", duration_s=41.2)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "applied_clean"
    assert out.branch == _BRANCH
    assert out.tests_exit_code == 0
    assert out.tests_duration_s == 41.2
    assert "98 passed" in out.tests_stdout_tail
    assert out.applied_at == _NOW
    assert out.patch_path == str(patch)


# --- test verdicts -----------------------------------------------------------


def test_applied_test_failed_when_tests_nonzero(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED tests/test_x.py::test_y\n", duration_s=37.4)),
    ])
    out = apply_patch(
        tmp_path, patch, runner=runner, clock=_clock, auto_revert=False,
    )
    assert out.verdict == "applied_test_failed"
    assert out.branch == _BRANCH
    assert out.tests_exit_code == 1
    assert "FAILED" in out.tests_stdout_tail
    assert "exited 1" in out.reason


def test_applied_test_failed_on_test_timeout(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(-1, "still running...\n", timed_out=True)),
    ])
    out = apply_patch(
        tmp_path, patch, runner=runner, clock=_clock,
        test_timeout_secs=12.0, auto_revert=False,
    )
    assert out.verdict == "applied_test_failed"
    assert "timed out" in out.reason
    assert out.tests_exit_code is None
    assert out.tests_duration_s == 12.0


# --- apply_conflict ----------------------------------------------------------


def test_apply_conflict_on_check_failure_no_branch_created(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"),
         GitResult(1, "", "error: patch failed: runtime/foo.py:12")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "apply_conflict"
    assert out.branch == ""
    assert "patch failed" in out.reason
    # Make sure we did NOT continue past --check.
    invoked = [a for a, _ in runner.calls]
    assert not any("checkout" in a for a in invoked)


def test_apply_conflict_after_check_passed_keeps_branch(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)

    def _apply_without_check(argv: object) -> bool:
        argv_set = set(argv)  # type: ignore[arg-type]
        return "apply" in argv_set and "--check" not in argv_set

    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (_apply_without_check, GitResult(1, "", "error: applying corrupt patch")),
    ])
    out = apply_patch(
        tmp_path, patch, runner=runner, clock=_clock, auto_revert=False,
    )
    assert out.verdict == "apply_conflict"
    assert out.branch == _BRANCH  # branch was created before apply ran
    assert "after --check passed" in out.reason


# --- precondition_failed -----------------------------------------------------


def test_precondition_failed_dirty_tree(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, " M runtime/foo.py\n")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "precondition_failed"
    assert "dirty" in out.reason
    assert out.branch == ""


def test_precondition_failed_protected_branch(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, "")),
        (has("rev-parse", "--abbrev-ref"), GitResult(0, "main\n")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "precondition_failed"
    assert "protected branch" in out.reason
    assert "'main'" in out.reason


def test_precondition_failed_no_diff_in_patch(tmp_path: Path) -> None:
    refused = _OK_PATCH_TEXT.replace(
        "## Unified diff\n```diff",
        "## Unified diff\n_Refused — touches canonical file._\n\nIGNORED",
    )
    patch = _write_patch(tmp_path, refused)
    runner = ScriptedRunner()  # no calls expected past diff check
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "precondition_failed"
    assert "no applicable diff" in out.reason
    assert runner.calls == []  # we exit before any subprocess


def test_precondition_failed_branch_already_exists(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, "")),
        (has("rev-parse", "--abbrev-ref"), GitResult(0, "feature/work\n")),
        (has("rev-parse", "--verify"),
         GitResult(0, f"refs/heads/{_BRANCH}\n")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "precondition_failed"
    assert "already exists" in out.reason
    assert _BRANCH in out.reason
    assert "git branch -D" in out.reason


# --- options -----------------------------------------------------------------


def test_no_tests_skips_test_run_and_marks_clean(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock, run_tests=False)
    assert out.verdict == "applied_clean"
    assert "skipped" in out.reason
    assert out.tests_exit_code is None
    assert out.tests_duration_s is None
    invoked = [a for a, _ in runner.calls]
    assert not any("make" in a and "test" in a for a in invoked)


def test_diff_is_passed_to_apply_via_stdin(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, "ok\n", duration_s=1.0)),
    ])
    apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    apply_stdins = [
        stdin
        for argv, stdin in runner.calls
        if "apply" in argv
    ]
    assert all(stdin is not None and stdin.startswith("diff --git") for stdin in apply_stdins)
    assert len(apply_stdins) == 2  # --check and the real apply


def test_test_stdout_tail_truncated_to_8kb(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    huge = ("LINE_X\n" * 2000) + "FINAL_LINE\n"  # ~14 KB
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, huge, duration_s=2.0)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "applied_clean"
    assert len(out.tests_stdout_tail) <= 8192
    assert out.tests_stdout_tail.endswith("FINAL_LINE\n")
