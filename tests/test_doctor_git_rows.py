"""Doctor rows that gate ``make apply`` on a working ``git`` install.

Two new rows under ``services:``:

| Label             | Severity if missing                                   |
| ----------------- | ----------------------------------------------------- |
| ``git:available`` | error — apply impossible without it                   |
| ``git:repo_clean``| warn — apply will refuse but doctor itself still ok   |

These tests stub ``shutil.which`` and ``subprocess.run`` so we never
spawn a real process — the goal is to pin the row contract, not to
re-test git itself.
"""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any

import pytest

from scripts import doctor

pytestmark = pytest.mark.unit


class _StubCompleted:
    """Drop-in stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _scripted_run(
    script: dict[tuple[str, ...], _StubCompleted],
) -> Any:
    """Build a ``subprocess.run`` replacement that dispatches by argv tuple."""

    def _run(argv: Sequence[str], **_kwargs: Any) -> _StubCompleted:
        key = tuple(argv[1:])  # drop the resolved git binary path
        if key in script:
            return script[key]
        raise AssertionError(f"unexpected git invocation: {tuple(argv)}")

    return _run


def test_git_missing_marks_error_and_skips_clean_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    rows = doctor._check_git()
    labels = [r[0] for r in rows]
    assert labels == ["git:available", "git:repo_clean"]
    assert rows[0][1] is False
    assert rows[0][3] == "error"
    assert "PATH" in rows[0][2]
    assert rows[1][3] == "warn"


def test_git_available_clean_repo_marks_both_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/git")
    script = {
        ("--version",): _StubCompleted(0, "git version 2.45.0\n"),
        ("rev-parse", "--is-inside-work-tree"): _StubCompleted(0, "true\n"),
        ("status", "--porcelain"): _StubCompleted(0, ""),
    }
    monkeypatch.setattr(doctor.subprocess, "run", _scripted_run(script))

    rows = doctor._check_git()

    assert [r[3] for r in rows] == ["ok", "ok"]
    assert "git version 2.45" in rows[0][2]
    assert rows[1][2] == "clean"


def test_git_available_dirty_tree_warns_only_on_clean_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/git")
    script = {
        ("--version",): _StubCompleted(0, "git version 2.45.0\n"),
        ("rev-parse", "--is-inside-work-tree"): _StubCompleted(0, "true\n"),
        ("status", "--porcelain"): _StubCompleted(
            0, " M runtime/foo.py\n?? notes.md\n"),
    }
    monkeypatch.setattr(doctor.subprocess, "run", _scripted_run(script))

    rows = doctor._check_git()

    assert rows[0][3] == "ok"  # availability still fine
    assert rows[1][3] == "warn"
    assert rows[1][1] is False
    assert "dirty" in rows[1][2]
    assert "2 change" in rows[1][2]


def test_git_outside_repo_warns_clean_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/git")
    script = {
        ("--version",): _StubCompleted(0, "git version 2.45.0\n"),
        ("rev-parse", "--is-inside-work-tree"): _StubCompleted(
            128, "", "fatal: not a git repository\n"),
    }
    monkeypatch.setattr(doctor.subprocess, "run", _scripted_run(script))

    rows = doctor._check_git()
    assert rows[0][3] == "ok"
    assert rows[1][3] == "warn"
    assert "not inside a git repo" in rows[1][2]


def test_git_version_probe_failure_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/git")

    def _boom(*_args: Any, **_kwargs: Any) -> _StubCompleted:
        raise OSError("permission denied")

    monkeypatch.setattr(doctor.subprocess, "run", _boom)

    rows = doctor._check_git()
    assert rows[0][3] == "error"
    assert "permission denied" in rows[0][2]
    assert rows[1][3] == "warn"


def test_git_status_timeout_warns_without_failing_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/git")

    def _run(argv: Sequence[str], **_kwargs: Any) -> _StubCompleted:
        key = tuple(argv[1:])
        if key == ("--version",):
            return _StubCompleted(0, "git version 2.45.0\n")
        if key == ("rev-parse", "--is-inside-work-tree"):
            return _StubCompleted(0, "true\n")
        if key == ("status", "--porcelain"):
            raise subprocess.TimeoutExpired(cmd=list(argv), timeout=2.0)
        raise AssertionError(f"unexpected: {tuple(argv)}")

    monkeypatch.setattr(doctor.subprocess, "run", _run)

    rows = doctor._check_git()
    assert rows[0][3] == "ok"
    assert rows[1][3] == "warn"
    assert "status probe failed" in rows[1][2]
