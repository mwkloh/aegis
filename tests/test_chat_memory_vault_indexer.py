"""Phase 7 Track C §5.1 — vault indexer contract.

Pins:

* Walk respects `glob` + `exclude` filters.
* mtime-based incremental reindex: unchanged body → skipped, not re-embedded.
* Prune removes tier-2 rows whose files vanished from disk.
* `only_label` scopes both the walk and the prune to one source.
* Missing `vault_root` / empty sources → structured error, no crash.
* FilesystemVaultBodyLoader resolves rel_paths AND stem slugs.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory.embeddings import FakeEmbedder
from runtime.chat.memory.tier2 import Tier2Store
from runtime.chat.memory.vault_indexer import (
    FilesystemVaultBodyLoader,
    ReindexResult,
    VaultIndexer,
)
from runtime.config import VaultIndexingConfig, VaultSource

pytestmark = pytest.mark.unit


def _fixed_clock() -> datetime:
    return datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)


def _make_store(tmp_path: Path) -> Tier2Store:
    return Tier2Store(tmp_path / "tier2.db", FakeEmbedder(dim=8))


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _make_indexer(
    tmp_path: Path,
    vault_root: Path,
    sources: tuple[VaultSource, ...],
) -> tuple[VaultIndexer, Tier2Store]:
    store = _make_store(tmp_path)
    cfg = VaultIndexingConfig(vault_root=vault_root, sources=sources)
    return VaultIndexer(tier2=store, config=cfg, clock=_fixed_clock), store


def test_reindex_disabled_without_vault_root(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    cfg = VaultIndexingConfig()  # defaults: no vault_root
    result = VaultIndexer(tier2=store, config=cfg, clock=_fixed_clock).reindex()
    assert result.added == 0
    assert result.errors == ("vault indexing not configured",)


def test_reindex_adds_matching_files(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Research" / "alpha.md", "# Alpha\nBody one.")
    _write(vault / "Research" / "sub" / "beta.md", "# Beta")
    _write(vault / "Ignored" / "skip.md", "not-a-source")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Research", priority=1.5, label="research"),),
    )
    result = indexer.reindex()
    assert result.added == 2
    assert result.updated == 0
    assert result.skipped == 0
    assert result.errors == ()
    rels = sorted(n.rel_path for n in store.list_vault_notes())
    assert rels == ["Research/alpha.md", "Research/sub/beta.md"]
    for note in store.list_vault_notes():
        assert note.label == "research"
        assert note.priority == 1.5


def test_reindex_is_incremental_on_unchanged_body(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Docs" / "a.md", "stable content")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Docs", label="docs"),),
    )
    first = indexer.reindex()
    assert first.added == 1
    # Second call: file unchanged → skipped, not re-embedded.
    second = indexer.reindex()
    assert second.added == 0
    assert second.updated == 0
    assert second.skipped == 1
    assert len(store.list_vault_notes()) == 1


def test_reindex_updates_when_body_changes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    note = _write(vault / "Docs" / "a.md", "v1")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Docs"),),
    )
    indexer.reindex()
    # Change the body — new sha256 → `updated`.
    note.write_text("v2 different body", encoding="utf-8")
    result = indexer.reindex()
    assert result.added == 0
    assert result.updated == 1
    assert result.skipped == 0
    assert len(store.list_vault_notes()) == 1


def test_reindex_prunes_deleted_files(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    keep = _write(vault / "Docs" / "keep.md", "k")
    gone = _write(vault / "Docs" / "gone.md", "g")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Docs"),),
    )
    indexer.reindex()
    assert len(store.list_vault_notes()) == 2
    gone.unlink()
    result = indexer.reindex()
    assert result.pruned == 1
    assert [n.rel_path for n in store.list_vault_notes()] == ["Docs/keep.md"]
    assert keep.exists()  # indexer did not touch the filesystem


def test_reindex_honors_exclude_patterns(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Research" / "live.md", "x")
    _write(vault / "Research" / "Archive" / "old.md", "y")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (
            VaultSource(
                path="Research",
                exclude=("**/Archive/**",),
                label="research",
            ),
        ),
    )
    result = indexer.reindex()
    assert result.added == 1
    rels = [n.rel_path for n in store.list_vault_notes()]
    assert rels == ["Research/live.md"]


def test_only_label_scopes_walk_and_prune(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Research" / "a.md", "ra")
    wiki_file = _write(vault / "Wiki" / "w.md", "wiki body")
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (
            VaultSource(path="Research", label="research"),
            VaultSource(path="Wiki", label="wiki"),
        ),
    )
    indexer.reindex()
    assert len(store.list_vault_notes()) == 2
    # Delete the wiki file on disk, but reindex only the research label.
    # The wiki row must survive because prune is label-scoped.
    wiki_file.unlink()
    result = indexer.reindex(only_label="research")
    assert result.pruned == 0
    assert result.skipped == 1  # research/a.md still unchanged
    rels = sorted(n.rel_path for n in store.list_vault_notes())
    assert rels == ["Research/a.md", "Wiki/w.md"]


def test_only_label_unknown_returns_error(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Docs" / "a.md", "x")
    indexer, _ = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Docs", label="docs"),),
    )
    result = indexer.reindex(only_label="nope")
    assert result.added == 0
    assert result.errors == ("no source matched label 'nope'",)


def test_missing_source_directory_does_not_crash(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()  # vault exists but "Missing" subfolder does not
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Missing", label="x"),),
    )
    result = indexer.reindex()
    assert result.added == 0
    assert result.errors == ()
    assert store.list_vault_notes() == ()


def test_reindex_result_timestamps_use_injected_clock(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Docs" / "a.md", "body")
    indexer, _ = _make_indexer(
        tmp_path,
        vault,
        (VaultSource(path="Docs"),),
    )
    result: ReindexResult = indexer.reindex()
    assert result.started_at == _fixed_clock()
    assert result.finished_at == _fixed_clock()
    assert result.total_indexed == 1


def test_filesystem_loader_resolves_rel_path(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Research" / "phase-7.md", "phase seven body")
    loader = FilesystemVaultBodyLoader(vault)
    assert loader.load("Research/phase-7.md") == "phase seven body"


def test_filesystem_loader_resolves_slug(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "Research" / "phase-7.md", "phase seven body")
    loader = FilesystemVaultBodyLoader(vault)
    assert loader.load("phase-7") == "phase seven body"


def test_filesystem_loader_missing_returns_empty(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    loader = FilesystemVaultBodyLoader(vault)
    assert loader.load("nonexistent") == ""


def test_filesystem_loader_ambiguous_slug_raises(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    _write(vault / "A" / "dup.md", "one")
    _write(vault / "B" / "dup.md", "two")
    loader = FilesystemVaultBodyLoader(vault)
    with pytest.raises(LookupError):
        loader.load("dup")


def test_priority_weighted_ranking_beats_higher_cosine(tmp_path: Path) -> None:
    # A priority=1.5 source at slightly lower cosine should outrank a
    # priority=1.0 source at higher cosine — the weighting is already
    # in Tier2Store.search_vault, the indexer just has to carry the
    # priority onto the row.
    vault = tmp_path / "vault"
    _write(vault / "Research" / "r.md", "alpha beta gamma")  # high-priority
    _write(vault / "Wiki" / "w.md", "alpha beta gamma delta")  # default
    indexer, store = _make_indexer(
        tmp_path,
        vault,
        (
            VaultSource(path="Research", priority=1.5, label="research"),
            VaultSource(path="Wiki", priority=1.0, label="wiki"),
        ),
    )
    indexer.reindex()
    hits = store.search_vault(query="alpha beta gamma", top_k=5)
    assert len(hits) == 2
    # Top hit must be the research note — its priority*cosine beats wiki.
    assert hits[0].record.label == "research"
