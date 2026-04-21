"""End-to-end Phase 5 — real ``git init`` repo, real ``git apply``, fake tests.

Stands the full apply flow up against a real working tree:

* a temp repo seeded by real ``git init`` + one commit on a non-protected
  branch (``feature/work``)
* a real ``.patch.md`` whose unified diff modifies the seeded source file
* a *hybrid* runner — real ``subprocess.run`` for every ``git ...``
  invocation (so ``git apply``, ``git checkout -b`` etc. actually
  mutate the repo) and a *scripted* response for ``make ... test`` (so
  the test suite never re-enters the host project's Makefile)

Asserts:
  * branch ``aegis/CT-001-a86b087a`` exists on disk
  * ``coding_harness/diffs/<...>.result.md`` written with ``applied_clean``
  * ``DECISIONS.md`` has one row, verdict ``applied_clean``
"""
from __future__ import annotations

import subprocess  # nosec
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness import apply_cli
from runtime.coding_harness.applier import GitResult
from runtime.coding_harness.patch_writer import diffs_dir
from runtime.coding_harness.result_writer import latest_result_for
from runtime.config import get_config
from runtime.improvement.decisions import load_decisions

pytestmark = pytest.mark.e2e


_NOW = datetime(2026, 4, 19, 9, 15, tzinfo=UTC)
_PATCH_FILENAME = "CT-001__IMP-a86b087a__2026-04-19T0915Z.patch.md"
_BRANCH = "aegis/CT-001-a86b087a"
_SOURCE_REL = "runtime/foo.py"

_INITIAL_SOURCE = "def f():\n    pass\n"

_PATCH_MD = """# CT-001 — IMP-a86b087a (status: ok)

## Summary
Adds a return value.

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
git branch -D aegis/CT-001-a86b087a
"""


def _clock() -> datetime:
    return _NOW


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Real git invocation — used only by the test harness (not the SUT)."""
    return subprocess.run(  # nosec
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True, check=True,
    )


def _seed_repo(repo_root: Path) -> None:
    """Stand up a fresh git repo on a non-protected branch with one commit."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec
        ["git", "init", "--initial-branch=feature/work", str(repo_root)],
        capture_output=True, text=True, check=True,
    )
    _git(repo_root, "config", "user.email", "aegis-test@example.com")
    _git(repo_root, "config", "user.name", "AEGIS Test")
    _git(repo_root, "config", "commit.gpgsign", "false")

    src = repo_root / _SOURCE_REL
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(_INITIAL_SOURCE, encoding="utf-8")
    _git(repo_root, "add", _SOURCE_REL)
    _git(repo_root, "commit", "-m", "seed")


def _seed_patch(workspace: Path) -> Path:
    target = diffs_dir(workspace) / _PATCH_FILENAME
    target.write_text(_PATCH_MD, encoding="utf-8")
    return target


def _hybrid_runner(make_result: GitResult) -> apply_cli.Runner:
    """Real subprocess for git, scripted for ``make ... test``."""

    def _run(
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> GitResult:
        argv_list = list(argv)
        if argv_list and argv_list[0] == "make":
            return make_result
        return apply_cli.real_runner(argv_list, stdin=stdin, timeout=timeout)

    return _run


def test_e2e_real_git_apply_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    repo_root = tmp_path / "repo"
    _seed_repo(repo_root)
    patch_path = _seed_patch(cfg.aegis_home)

    runner = _hybrid_runner(
        GitResult(exit_code=0, stdout="98 passed in 0.5s\n", duration_s=0.5),
    )

    rc = apply_cli.run(
        cfg,
        ct_id="CT-001",
        repo_root=repo_root,
        status=False,
        dry_run=False,
        run_tests=True,
        runner=runner,
        clock=_clock,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "applied_clean" in out
    assert _BRANCH in out

    # Branch exists on disk.
    branches = subprocess.run(  # nosec
        ["git", "-C", str(repo_root), "branch", "--list", _BRANCH],
        capture_output=True, text=True, check=True,
    )
    assert _BRANCH in branches.stdout

    # We're now on the new branch and the source file shows the patched line.
    head = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head == _BRANCH
    assert "return 1" in (repo_root / _SOURCE_REL).read_text(encoding="utf-8")

    # Result file written next to the patch.
    result_path = latest_result_for(cfg.aegis_home, "CT-001")
    assert result_path is not None
    assert result_path.parent == patch_path.parent
    body = result_path.read_text(encoding="utf-8")
    assert "applied_clean" in body
    assert _BRANCH in body

    # One decision row, mapped to the applier verdict.
    decisions = load_decisions(cfg.aegis_home)
    assert len(decisions) == 1
    assert decisions[0].imp_id == "IMP-a86b087a"
    assert decisions[0].verdict == "applied_clean"
    assert _BRANCH in decisions[0].rationale


def test_e2e_real_git_apply_test_failed_auto_reverts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Patch applies to disk, tests fail → default auto-revert restores HEAD.

    Verifies the Phase 6 Track B contract end-to-end:
    * back on the original branch (``feature/work``)
    * apply branch deleted
    * working tree clean and source file restored to pre-patch contents
    * verdict still ``applied_test_failed`` (revert never masks the failure)
    """
    cfg = get_config()
    repo_root = tmp_path / "repo"
    _seed_repo(repo_root)
    _seed_patch(cfg.aegis_home)

    runner = _hybrid_runner(
        GitResult(exit_code=1, stdout="FAILED tests/x.py::y\n", duration_s=2.4),
    )

    rc = apply_cli.run(
        cfg,
        ct_id="CT-001",
        repo_root=repo_root,
        status=False,
        dry_run=False,
        run_tests=True,
        runner=runner,
        clock=_clock,
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "applied_test_failed" in out
    assert "auto-reverted" in out

    # Back on the original branch.
    head = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head == "feature/work"

    # Apply branch is gone.
    branches = subprocess.run(  # nosec
        ["git", "-C", str(repo_root), "branch", "--list", _BRANCH],
        capture_output=True, text=True, check=True,
    )
    assert _BRANCH not in branches.stdout

    # Working tree clean and source restored.
    status = _git(repo_root, "status", "--porcelain").stdout
    assert status == ""
    assert (repo_root / _SOURCE_REL).read_text(encoding="utf-8") == _INITIAL_SOURCE

    decisions = load_decisions(cfg.aegis_home)
    assert decisions[-1].verdict == "applied_test_failed"


def test_e2e_real_git_apply_test_failed_no_revert_keeps_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--no-revert`` preserves Phase 5 behaviour — broken branch left for inspection."""
    cfg = get_config()
    repo_root = tmp_path / "repo"
    _seed_repo(repo_root)
    _seed_patch(cfg.aegis_home)

    runner = _hybrid_runner(
        GitResult(exit_code=1, stdout="FAILED tests/x.py::y\n", duration_s=2.4),
    )

    rc = apply_cli.run(
        cfg,
        ct_id="CT-001",
        repo_root=repo_root,
        status=False,
        dry_run=False,
        run_tests=True,
        auto_revert=False,
        runner=runner,
        clock=_clock,
    )

    assert rc == 1
    out = capsys.readouterr().out
    assert "applied_test_failed" in out
    assert "auto-revert" not in out

    # Branch still exists — we leave it for the human to inspect/delete.
    head = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert head == _BRANCH
    assert "return 1" in (repo_root / _SOURCE_REL).read_text(encoding="utf-8")

    decisions = load_decisions(cfg.aegis_home)
    assert decisions[-1].verdict == "applied_test_failed"
