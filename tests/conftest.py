"""Test-wide fixtures.

Every test runs with `AEGIS_HOME` and `AEGIS_ROOT` pointed at a `tmp_path`
sandbox so production canon at `~/.aegis/` is never touched.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from runtime.config import reset_config


@pytest.fixture(autouse=True)
def aegis_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "aegis"
    home = root / "workspace"
    (home / "memory").mkdir(parents=True)
    (home / "sessions").mkdir(parents=True)
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("AEGIS_ROOT", str(root))
    monkeypatch.setenv("AEGIS_HOME", str(home))
    # Drop the cached AegisConfig so each test re-resolves against the sandbox.
    reset_config()
    yield root
    reset_config()


@pytest.fixture
def env_value() -> str:
    return os.environ.get("AEGIS_HOME", "")
