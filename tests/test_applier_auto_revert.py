"""Phase 6 Track B — auto-revert behaviour in ``apply_patch``.

Pins the contract:

* Default ``auto_revert=True`` runs ``reset --hard HEAD`` →
  ``checkout <original>`` → ``branch -D <apply_branch>`` after a
  failure that left the apply branch in place.
* ``applied_test_failed`` and ``apply_conflict`` (after ``checkout
  -b``) trigger revert. ``apply_conflict`` *before* branch creation
  does NOT (no branch to revert).
* ``precondition_failed`` never reverts.
* ``applied_clean`` never reverts.
* Revert helper failure surfaces in ``reason`` as a soft suffix; the
  verdict is unchanged.
* ``auto_revert=False`` preserves Phase 5 behaviour exactly — no
  reset/checkout/branch -D calls are made.
"""
from __future__ import annotations

from collections.abc import Sequence
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
_PATCH_FILENAME = "CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md"
_BRANCH = "aegis/CT-001-a86b087a"
_ORIGINAL_BRANCH = "feature/work"

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


def _clock() -> datetime:
    return _NOW


def _write_patch(tmp_path: Path) -> Path:
    p = tmp_path / _PATCH_FILENAME
    p.write_text(_OK_PATCH_TEXT, encoding="utf-8")
    return p


def _green_preconditions() -> list[tuple[object, GitResult]]:
    return [
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, "")),
        (has("rev-parse", "--abbrev-ref"), GitResult(0, f"{_ORIGINAL_BRANCH}\n")),
        (has("rev-parse", "--verify"), GitResult(1, "", "fatal: not a ref")),
    ]


def _argvs(runner: ScriptedRunner) -> list[tuple[str, ...]]:
    return [argv for argv, _stdin in runner.calls]


def _has_call(runner: ScriptedRunner, *needles: str) -> bool:
    """True if any call's argv contains every needle (in any order)."""
    needle_set = set(needles)
    return any(needle_set.issubset(set(argv)) for argv in _argvs(runner))


# --- default revert ON triggers cleanup --------------------------------------


def test_test_failed_default_reverts_to_original_branch(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED tests/x.py\n", duration_s=2.0)),
        (has("reset", "--hard"), GitResult(0)),
        (has("checkout", _ORIGINAL_BRANCH), GitResult(0)),
        (has("branch", "-D", _BRANCH), GitResult(0)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "applied_test_failed"
    assert out.branch == _BRANCH
    assert out.tests_exit_code == 1
    assert "auto-reverted" in out.reason
    assert _ORIGINAL_BRANCH in out.reason
    assert _has_call(runner, "reset", "--hard", "HEAD")
    assert _has_call(runner, "checkout", _ORIGINAL_BRANCH)
    assert _has_call(runner, "branch", "-D", _BRANCH)


def test_test_failed_no_revert_flag_keeps_branch(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED\n", duration_s=2.0)),
    ])
    out = apply_patch(
        tmp_path, patch, runner=runner, clock=_clock, auto_revert=False,
    )
    assert out.verdict == "applied_test_failed"
    assert out.branch == _BRANCH
    assert "auto-revert" not in out.reason
    assert not _has_call(runner, "reset", "--hard", "HEAD")
    assert not _has_call(runner, "branch", "-D", _BRANCH)


def test_test_timeout_also_reverts(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(-1, "running\n", timed_out=True)),
        (has("reset", "--hard"), GitResult(0)),
        (has("checkout", _ORIGINAL_BRANCH), GitResult(0)),
        (has("branch", "-D", _BRANCH), GitResult(0)),
    ])
    out = apply_patch(
        tmp_path, patch, runner=runner, clock=_clock, test_timeout_secs=12.0,
    )
    assert out.verdict == "applied_test_failed"
    assert "timed out" in out.reason
    assert "auto-reverted" in out.reason
    assert _has_call(runner, "branch", "-D", _BRANCH)


def test_apply_conflict_after_branch_reverts(tmp_path: Path) -> None:
    """``git apply`` rejects after ``--check`` passed → branch was created → revert."""
    patch = _write_patch(tmp_path)

    def _apply_without_check(argv: Sequence[str]) -> bool:
        argv_set = set(argv)
        return "apply" in argv_set and "--check" not in argv_set

    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (_apply_without_check, GitResult(1, "", "corrupt patch")),
        (has("reset", "--hard"), GitResult(0)),
        (has("checkout", _ORIGINAL_BRANCH), GitResult(0)),
        (has("branch", "-D", _BRANCH), GitResult(0)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "apply_conflict"
    assert out.branch == _BRANCH
    assert "auto-reverted" in out.reason
    assert _has_call(runner, "branch", "-D", _BRANCH)


# --- revert is NOT triggered for these --------------------------------------


def test_apply_conflict_before_branch_does_not_revert(tmp_path: Path) -> None:
    """``git apply --check`` fails → no branch created → nothing to revert."""
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(1, "", "error: patch failed")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "apply_conflict"
    assert out.branch == ""
    assert "auto-revert" not in out.reason
    assert not _has_call(runner, "checkout", _ORIGINAL_BRANCH)
    assert not _has_call(runner, "branch", "-D", _BRANCH)


def test_precondition_failed_never_reverts(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, " M runtime/foo.py\n")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "precondition_failed"
    assert "auto-revert" not in out.reason
    assert not _has_call(runner, "branch", "-D", _BRANCH)


def test_applied_clean_never_reverts(tmp_path: Path) -> None:
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, "98 passed\n", duration_s=1.0)),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "applied_clean"
    assert out.branch == _BRANCH
    assert not _has_call(runner, "reset", "--hard", "HEAD")
    assert not _has_call(runner, "branch", "-D", _BRANCH)


# --- partial revert failure surfaces softly ---------------------------------


def test_revert_failure_surfaces_in_reason_but_verdict_unchanged(
    tmp_path: Path,
) -> None:
    """If ``checkout`` fails during revert, the test-failure verdict stands."""
    patch = _write_patch(tmp_path)
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED\n", duration_s=2.0)),
        (has("reset", "--hard"), GitResult(0)),
        (has("checkout", _ORIGINAL_BRANCH),
         GitResult(1, "", "error: cannot switch")),
    ])
    out = apply_patch(tmp_path, patch, runner=runner, clock=_clock)
    assert out.verdict == "applied_test_failed"  # verdict NOT masked
    assert out.tests_exit_code == 1
    assert "auto-revert failed" in out.reason
    assert "checkout failed" in out.reason
    # We never tried to delete the branch after the checkout failed.
    assert not _has_call(runner, "branch", "-D", _BRANCH)
