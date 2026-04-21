"""Unit tests for the ``vault_reindex`` skill script.

Verifies the Phase 10 Track D3b argv-only skill convention:

* Disabled vault_indexing → exit 0 silently (scheduler pushes nothing).
* Enabled vault_indexing → walks configured sources, upserts, and prunes
  via ``VaultIndexer.reindex``; exit 0 when result.errors is empty.
* ``--label`` scopes the reindex to a single ``VaultSource.label``.
* ``result.errors`` on the reindex → exit 1 so the scheduler flags it.
* Silent-success contract: stdout is empty on exit 0.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runtime.chat.memory.vault_indexer import ReindexResult
from runtime.config import VaultIndexingConfig, VaultSource
from runtime.skills.scripts import vault_reindex

pytestmark = pytest.mark.unit


class _FakeConfig:
    def __init__(self, *, vault_indexing: VaultIndexingConfig, memory_db: Path) -> None:
        self.vault_indexing = vault_indexing
        self.storage = SimpleNamespace(memory_db=memory_db)


def _patch_config(
    monkeypatch: pytest.MonkeyPatch, cfg: _FakeConfig
) -> None:
    monkeypatch.setattr(vault_reindex, "get_config", lambda: cfg)


def _patch_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vault_reindex, "Tier2Store", lambda _db, _embedder: object()
    )


class _FakeIndexer:
    def __init__(self, result: ReindexResult, expected_label: str | None = None) -> None:
        self._result = result
        self._expected_label = expected_label
        self.calls: list[str | None] = []

    def __call__(self, **kwargs: Any) -> _FakeIndexer:
        # Called as the `VaultIndexer(...)` constructor.
        return self

    def reindex(self, *, only_label: str | None = None) -> ReindexResult:
        self.calls.append(only_label)
        return self._result


def _reindex_result(**overrides: Any) -> ReindexResult:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "started_at": now,
        "finished_at": now,
        "added": 0,
        "updated": 0,
        "skipped": 0,
        "pruned": 0,
        "errors": (),
    }
    base.update(overrides)
    return ReindexResult(**base)


def test_disabled_vault_returns_zero_silently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _FakeConfig(
        vault_indexing=VaultIndexingConfig(),  # vault_root=None → disabled
        memory_db=tmp_path / "mem.db",
    )
    _patch_config(monkeypatch, cfg)

    rc = vault_reindex.main([])

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""  # silent-success contract


def test_happy_path_runs_reindex_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _FakeConfig(
        vault_indexing=VaultIndexingConfig(
            vault_root=tmp_path,
            sources=(VaultSource(path="Docs", label="docs"),),
        ),
        memory_db=tmp_path / "mem.db",
    )
    _patch_config(monkeypatch, cfg)
    _patch_store(monkeypatch)
    fake = _FakeIndexer(_reindex_result(added=3, updated=1, skipped=12, pruned=0))
    monkeypatch.setattr(vault_reindex, "VaultIndexer", fake)

    rc = vault_reindex.main([])

    assert rc == 0
    assert fake.calls == [None]  # full reindex, no label filter
    captured = capsys.readouterr()
    assert captured.out == ""  # silent


def test_label_scopes_to_single_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _FakeConfig(
        vault_indexing=VaultIndexingConfig(
            vault_root=tmp_path,
            sources=(
                VaultSource(path="Docs", label="docs"),
                VaultSource(path="Research", label="research"),
            ),
        ),
        memory_db=tmp_path / "mem.db",
    )
    _patch_config(monkeypatch, cfg)
    _patch_store(monkeypatch)
    fake = _FakeIndexer(_reindex_result(added=1))
    monkeypatch.setattr(vault_reindex, "VaultIndexer", fake)

    rc = vault_reindex.main(["--label", "research"])

    assert rc == 0
    assert fake.calls == ["research"]


def test_errors_returned_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _FakeConfig(
        vault_indexing=VaultIndexingConfig(
            vault_root=tmp_path,
            sources=(VaultSource(path="Docs", label="docs"),),
        ),
        memory_db=tmp_path / "mem.db",
    )
    _patch_config(monkeypatch, cfg)
    _patch_store(monkeypatch)
    fake = _FakeIndexer(
        _reindex_result(errors=("read failed for 'x.md': OSError",)),
    )
    monkeypatch.setattr(vault_reindex, "VaultIndexer", fake)

    rc = vault_reindex.main([])

    assert rc == 1


def test_embedder_falls_back_to_fake_when_bgem3_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory.embeddings import FakeEmbedder, build_embedder

    monkeypatch.setattr(
        "memory.embeddings.Bgem3Embedder",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no ollama")),
    )
    embedder = build_embedder()
    assert isinstance(embedder, FakeEmbedder)


def test_embedder_uses_bgem3_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memory.embeddings import build_embedder

    sentinel = object()
    monkeypatch.setattr("memory.embeddings.Bgem3Embedder", lambda **_: sentinel)
    assert build_embedder() is sentinel
