"""Scriptable + ``--status`` modes of ``apply_cli``.

Covers the orchestration the CLI wraps around ``apply_patch``: locating
the latest patch, mapping verdicts onto the decisions enum, writing
``.result.md``, recording one ``DECISIONS.md`` row, exit codes for
each verdict, ``--dry-run`` (preconditions + ``git apply --check``),
``--no-tests`` (skip the test stage), and the no-patch error path.
No real subprocess — every git/make call is scripted.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness import apply_cli
from runtime.coding_harness.applier import GitResult, ScriptedRunner, has
from runtime.coding_harness.patch_writer import diffs_dir
from runtime.coding_harness.result_writer import latest_result_for
from runtime.config import get_config
from runtime.improvement.decisions import latest_by_imp, load_decisions

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 19, 8, 30, tzinfo=UTC)
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


def _clock() -> datetime:
    return _NOW


def _seed_patch(workspace: Path, text: str = _OK_PATCH_TEXT) -> Path:
    """Drop a patch file in the canonical workspace location."""
    target = diffs_dir(workspace) / _PATCH_FILENAME
    target.write_text(text, encoding="utf-8")
    return target


def _green_preconditions() -> list[tuple[object, GitResult]]:
    return [
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, "")),
        (has("rev-parse", "--abbrev-ref"), GitResult(0, "feature/work\n")),
        (has("rev-parse", "--verify"), GitResult(1, "", "fatal: not a ref")),
    ]


def _run(
    *,
    repo_root: Path,
    runner: ScriptedRunner,
    ct_id: str = "CT-001",
    status: bool = False,
    dry_run: bool = False,
    run_tests: bool = True,
    auto_revert: bool = False,
) -> int:
    return apply_cli.run(
        get_config(),
        ct_id=ct_id,
        repo_root=repo_root,
        status=status,
        dry_run=dry_run,
        run_tests=run_tests,
        auto_revert=auto_revert,
        runner=runner,
        clock=_clock,
    )


# --- happy path --------------------------------------------------------------


def test_apply_clean_returns_zero_writes_result_and_decision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, "98 passed in 41.2s\n", duration_s=41.2)),
    ])

    rc = _run(repo_root=repo_root, runner=runner)

    assert rc == 0
    out = capsys.readouterr().out
    assert "applied_clean" in out
    assert _BRANCH in out

    result_path = latest_result_for(cfg.aegis_home, "CT-001")
    assert result_path is not None
    assert "applied_clean" in result_path.read_text(encoding="utf-8")

    decisions = load_decisions(cfg.aegis_home)
    assert len(decisions) == 1
    assert decisions[0].verdict == "applied_clean"
    assert decisions[0].imp_id == "IMP-a86b087a"
    assert _BRANCH in decisions[0].rationale


def test_apply_test_failed_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED tests/x.py::y\n", duration_s=12.3)),
    ])

    rc = _run(repo_root=repo_root, runner=runner)

    assert rc == 1
    out = capsys.readouterr().out
    assert "applied_test_failed" in out
    assert "exit 1" in out

    decisions = load_decisions(cfg.aegis_home)
    assert latest_by_imp(decisions)["IMP-a86b087a"].verdict == "applied_test_failed"


def test_apply_conflict_maps_through_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"),
         GitResult(1, "", "error: patch failed: runtime/foo.py:12")),
    ])

    rc = _run(repo_root=repo_root, runner=runner)

    assert rc == 1
    decisions = load_decisions(cfg.aegis_home)
    assert decisions[-1].verdict == "apply_conflict"


def test_precondition_failed_maps_to_apply_conflict_in_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``precondition_failed`` is not a decisions enum value — it folds
    onto ``apply_conflict`` so the audit log only carries the four
    declared applier verdicts."""
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        (has("rev-parse", "--is-inside-work-tree"), GitResult(0, "true\n")),
        (has("status", "--porcelain"), GitResult(0, " M runtime/foo.py\n")),
    ])

    rc = _run(repo_root=repo_root, runner=runner)

    assert rc == 1
    decisions = load_decisions(cfg.aegis_home)
    assert decisions[-1].verdict == "apply_conflict"
    assert "dirty" in decisions[-1].rationale


# --- auto-revert plumbing (Phase 6 Track B) ---------------------------------


def test_auto_revert_default_runs_cleanup_on_test_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``auto_revert=True`` the CLI runs the revert sequence and
    surfaces ``auto-reverted`` in the printed note + the decision rationale."""
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(1, "FAILED\n", duration_s=2.0)),
        (has("reset", "--hard"), GitResult(0)),
        (has("checkout", "feature/work"), GitResult(0)),
        (has("branch", "-D", _BRANCH), GitResult(0)),
    ])

    rc = _run(repo_root=repo_root, runner=runner, auto_revert=True)

    assert rc == 1
    out = capsys.readouterr().out
    assert "applied_test_failed" in out
    assert "auto-reverted" in out

    decisions = load_decisions(cfg.aegis_home)
    assert decisions[-1].verdict == "applied_test_failed"
    assert "auto-reverted" in decisions[-1].rationale


def test_main_no_revert_flag_disables_revert(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``python -m apply_cli CT-001 --no-revert`` reaches ``apply_patch``
    with ``auto_revert=False`` — verified by capturing the kwarg."""
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    captured: dict[str, object] = {}

    def _fake_run(_cfg: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(apply_cli, "run", _fake_run)

    rc = apply_cli.main(["CT-001", "--no-revert", "--repo-root", str(repo_root)])

    assert rc == 0
    assert captured["auto_revert"] is False
    assert captured["ct_id"] == "CT-001"


def test_main_default_passes_auto_revert_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--no-revert`` the CLI defaults to ``auto_revert=True``."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    captured: dict[str, object] = {}

    def _fake_run(_cfg: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(apply_cli, "run", _fake_run)

    rc = apply_cli.main(["CT-001", "--repo-root", str(repo_root)])

    assert rc == 0
    assert captured["auto_revert"] is True


# --- options -----------------------------------------------------------------


def test_no_tests_skips_make_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
    ])

    rc = _run(repo_root=repo_root, runner=runner, run_tests=False)

    assert rc == 0
    invoked = [a for a, _ in runner.calls]
    assert not any("make" in a and "test" in a for a in invoked)
    out = capsys.readouterr().out
    assert "applied_clean" in out


def test_dry_run_passes_when_check_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
    ])

    rc = _run(repo_root=repo_root, runner=runner, dry_run=True)

    assert rc == 0
    out = capsys.readouterr().out
    assert "would apply cleanly" in out

    # No branch created, no decision recorded.
    invoked = [a for a, _ in runner.calls]
    assert not any("checkout" in a for a in invoked)
    assert load_decisions(cfg.aegis_home) == []


def test_dry_run_reports_conflict_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"),
         GitResult(1, "", "error: patch failed: runtime/foo.py:12")),
    ])

    rc = _run(repo_root=repo_root, runner=runner, dry_run=True)

    assert rc == 1
    err = capsys.readouterr().err
    assert "would CONFLICT" in err
    assert load_decisions(cfg.aegis_home) == []


def test_dry_run_refuses_patch_with_no_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    refused = _OK_PATCH_TEXT.replace(
        "## Unified diff\n```diff",
        "## Unified diff\n_Refused — touches canon._\n\nIGNORED",
    )
    _seed_patch(cfg.aegis_home, refused)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner()  # no scripted responses → never reached

    rc = _run(repo_root=repo_root, runner=runner, dry_run=True)

    assert rc == 1
    err = capsys.readouterr().err
    assert "no applicable diff" in err
    assert runner.calls == []


# --- error paths -------------------------------------------------------------


def test_missing_patch_returns_one_with_helpful_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner()

    rc = _run(repo_root=repo_root, runner=runner)

    assert rc == 1
    err = capsys.readouterr().err
    assert "no .patch.md" in err
    assert "CT-001" in err
    assert "make harness" in err
    assert runner.calls == []  # we exit before any subprocess


# --- --status ---------------------------------------------------------------


def test_status_prints_latest_result_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = ScriptedRunner(script=[
        *_green_preconditions(),
        (has("apply", "--check"), GitResult(0)),
        (has("checkout", "-b"), GitResult(0)),
        (has("apply"), GitResult(0)),
        (lambda a: "make" in a and "test" in a,
         GitResult(0, "98 passed in 41.2s\n", duration_s=41.2)),
    ])
    _run(repo_root=repo_root, runner=runner)
    capsys.readouterr()  # drop apply output

    rc = _run(repo_root=repo_root, runner=ScriptedRunner(), status=True)

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 patch(es)" in out
    assert "applied_clean" in out  # rendered .result.md is echoed inline


def test_status_without_patches_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    rc = _run(repo_root=repo_root, runner=ScriptedRunner(), status=True)

    assert rc == 1
    out = capsys.readouterr().out
    assert "no patches found" in out


def test_status_with_patch_but_no_apply_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = get_config()
    _seed_patch(cfg.aegis_home)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    rc = _run(repo_root=repo_root, runner=ScriptedRunner(), status=True)

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 patch(es)" in out
    assert "no apply attempts recorded yet" in out
