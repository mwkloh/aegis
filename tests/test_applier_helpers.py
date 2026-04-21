"""Pure-helper tests for the applier — preconditions and diff extraction.

These tests use a ``FakeGitRunner`` so no real subprocess is spawned.
The orchestrator (``apply_patch``) and its real subprocess runner land
in A3.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.coding_harness.applier import (
    FakeGitRunner,
    GitResult,
    _check_preconditions,
    _extract_unified_diff,
    _protected_branches,
)

pytestmark = pytest.mark.unit


# --- _extract_unified_diff ---------------------------------------------------


_OK_PATCH = """# CT-001 — IMP-a86b087a (status: ok)

- **Drafted:** 2026-04-18T12:07Z
- **Model:** test/test

## Summary
Adds a thing.

## Unified diff
```diff
diff --git a/runtime/foo.py b/runtime/foo.py
--- a/runtime/foo.py
+++ b/runtime/foo.py
@@ -1,3 +1,4 @@
 def f():
+    return 1
     pass
```

## Test notes
—

## Rollback
—
"""


_REFUSED_PATCH = """# CT-001 — IMP-a86b087a (status: refused)

## Summary
—

## Unified diff
_Refused — task scope or model output touches canonical files._

## Test notes
—

## Rollback
—
"""


_STUB_PATCH = """# CT-001 — IMP-a86b087a (status: stub)

## Summary
—

## Unified diff
_Stub — no diff produced. Reason: model unreachable._

## Test notes
—

## Rollback
—
"""


def test_extract_diff_from_ok_patch() -> None:
    diff = _extract_unified_diff(_OK_PATCH)
    assert diff is not None
    assert diff.startswith("diff --git a/runtime/foo.py")
    assert "+    return 1" in diff
    # `git apply` rejects a patch whose final line lacks a newline
    # ("error: corrupt patch at line N") — make the contract explicit.
    assert diff.endswith("\n")


def test_extract_diff_returns_none_for_refused() -> None:
    assert _extract_unified_diff(_REFUSED_PATCH) is None


def test_extract_diff_returns_none_for_stub() -> None:
    assert _extract_unified_diff(_STUB_PATCH) is None


def test_extract_diff_returns_none_when_section_missing() -> None:
    assert _extract_unified_diff("# CT-001\n\n## Summary\nnope\n") is None


def test_extract_diff_returns_none_when_diff_empty() -> None:
    text = "## Unified diff\n```diff\n\n```\n"
    assert _extract_unified_diff(text) is None


# --- _check_preconditions ----------------------------------------------------


def _runner(**canned: GitResult) -> FakeGitRunner:
    """Build a runner keyed on the meaningful suffix of each git call."""
    table: dict[tuple[str, ...], GitResult] = {}
    for key, val in canned.items():
        # Keys like 'is_inside_work_tree' map onto the actual argv tail.
        if key == "is_inside":
            table[("git", "-C", "/repo", "rev-parse", "--is-inside-work-tree")] = val
        elif key == "status":
            table[("git", "-C", "/repo", "status", "--porcelain")] = val
        elif key == "branch":
            table[("git", "-C", "/repo", "rev-parse", "--abbrev-ref", "HEAD")] = val
        else:
            raise AssertionError(f"unknown canned key: {key}")
    return FakeGitRunner(responses=table)


def test_preconditions_clean_repo_on_feature_branch_is_ok() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, ""),
        branch=GitResult(0, "feature/foo\n"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is None
    assert branch == "feature/foo"


def test_preconditions_not_a_repo() -> None:
    runner = _runner(
        is_inside=GitResult(128, "", "fatal: not a git repository"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert "not a git repository" in reason
    assert branch == ""


def test_preconditions_dirty_tree_refused() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, " M runtime/foo.py\n?? scratch.txt\n"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason == "working tree dirty (uncommitted changes)"
    assert branch == ""


def test_preconditions_protected_branch_refused() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, ""),
        branch=GitResult(0, "main\n"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert "protected branch" in reason
    assert "'main'" in reason
    assert branch == "main"


def test_preconditions_protected_master_refused() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, ""),
        branch=GitResult(0, "master\n"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert "master" in reason
    assert branch == "master"


def test_preconditions_protected_branches_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEGIS_PROTECTED_BRANCHES", "trunk, release/prod ")
    assert _protected_branches() == ("trunk", "release/prod")
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, ""),
        branch=GitResult(0, "trunk\n"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert "trunk" in reason
    assert branch == "trunk"


def test_preconditions_status_failure_surfaces_stderr() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(1, "", "fatal: bad object"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert reason.startswith("git status failed")
    assert "bad object" in reason
    assert branch == ""


def test_preconditions_branch_failure_surfaces_reason() -> None:
    runner = _runner(
        is_inside=GitResult(0, "true\n"),
        status=GitResult(0, ""),
        branch=GitResult(128, "", "fatal: not a valid object"),
    )
    reason, branch = _check_preconditions(Path("/repo"), git=runner)
    assert reason is not None
    assert "could not determine current branch" in reason
    assert branch == ""
