"""Unit tests for the ``tier2_compress`` skill script.

Phase 10 Track D3c. Validates that the scheduled tier-2 maintenance CLI:

* Runs VACUUM + ANALYZE on the configured memory DB.
* Exits 0 silently when the DB doesn't exist yet (first boot before any
  chat turns hit tier-2).
* Exits 1 on a corrupted DB.
* Keeps stdout empty so the scheduler push layer pushes nothing.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runtime.skills.scripts import tier2_compress

pytestmark = pytest.mark.unit


class _FakeConfig:
    def __init__(self, *, memory_db: Path) -> None:
        self.storage = SimpleNamespace(memory_db=memory_db)


def _patch_config(
    monkeypatch: pytest.MonkeyPatch, cfg: _FakeConfig
) -> None:
    monkeypatch.setattr(tier2_compress, "get_config", lambda: cfg)


def _make_db(path: Path, *, bloat: bool = False) -> None:
    """Create a tiny SQLite DB. If `bloat=True`, insert + delete rows to
    leave freed pages VACUUM can reclaim.
    """
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT);")
        if bloat:
            conn.executemany(
                "INSERT INTO t (body) VALUES (?)",
                [("x" * 4096,) for _ in range(64)],
            )
            conn.execute("DELETE FROM t;")
        conn.commit()


def test_missing_db_is_silent_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _FakeConfig(memory_db=tmp_path / "missing.db")
    _patch_config(monkeypatch, cfg)

    rc = tier2_compress.main([])

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_vacuum_shrinks_bloated_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "mem.db"
    _make_db(db, bloat=True)
    before = db.stat().st_size
    cfg = _FakeConfig(memory_db=db)
    _patch_config(monkeypatch, cfg)

    rc = tier2_compress.main([])

    assert rc == 0
    after = db.stat().st_size
    assert after < before  # VACUUM reclaimed freed pages
    captured = capsys.readouterr()
    assert captured.out == ""  # silent-success contract


def test_healthy_db_stays_valid_after_maintenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "mem.db"
    _make_db(db, bloat=False)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO t (body) VALUES ('kept');")
        conn.commit()
    cfg = _FakeConfig(memory_db=db)
    _patch_config(monkeypatch, cfg)

    rc = tier2_compress.main([])

    assert rc == 0
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT body FROM t").fetchall()
    assert rows == [("kept",)]


def test_corrupted_db_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"this is not a sqlite database")
    cfg = _FakeConfig(memory_db=db)
    _patch_config(monkeypatch, cfg)

    rc = tier2_compress.main([])

    assert rc == 1


def test_unexpected_sqlite_error_surfaces_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "mem.db"
    _make_db(db, bloat=False)
    cfg = _FakeConfig(memory_db=db)
    _patch_config(monkeypatch, cfg)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise sqlite3.DatabaseError("disk I/O error")

    monkeypatch.setattr(tier2_compress, "_maintain", boom)

    rc = tier2_compress.main([])
    assert rc == 1
