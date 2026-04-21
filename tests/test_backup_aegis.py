"""Unit tests for scripts/backup_aegis.py.

Covers:
* Exit 1 when AEGIS_BACKUP_DEST is unset.
* Returns 0 on a successful rsync run (subprocess mocked).
* Returns 1 when rsync exits non-zero.
* The rsync command includes the --delete flag.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.backup_aegis import main

pytestmark = pytest.mark.unit


def test_main_exits_1_when_backup_dest_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AEGIS_BACKUP_DEST", raising=False)
    result = main()
    assert result == 1


def test_main_runs_rsync_on_valid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aegis_root = tmp_path / "aegis"
    backup_dest = tmp_path / "backups"
    (aegis_root / "workspace").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AEGIS_ROOT", str(aegis_root))
    monkeypatch.setenv("AEGIS_BACKUP_DEST", str(backup_dest))

    mock_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = main()

    assert result == 0
    mock_run.assert_called_once()


def test_main_exits_1_on_rsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aegis_root = tmp_path / "aegis"
    backup_dest = tmp_path / "backups"
    (aegis_root / "workspace").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AEGIS_ROOT", str(aegis_root))
    monkeypatch.setenv("AEGIS_BACKUP_DEST", str(backup_dest))

    mock_result = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="rsync error: some failure"
    )
    with patch("subprocess.run", return_value=mock_result):
        result = main()

    assert result == 1


def test_rsync_command_includes_delete_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aegis_root = tmp_path / "aegis"
    backup_dest = tmp_path / "backups"
    (aegis_root / "workspace").mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AEGIS_ROOT", str(aegis_root))
    monkeypatch.setenv("AEGIS_BACKUP_DEST", str(backup_dest))

    mock_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        main()

    call_args = mock_run.call_args
    # First positional argument is the command list
    cmd = call_args[0][0]
    assert "--delete" in cmd
