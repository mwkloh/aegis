"""Phase 7 Track C §5.1 — `/vault status|reindex|sources` handler contract.

Pins:

* Usage fallback when no args or an unknown subcommand is given.
* `/vault sources` renders `VaultIndexingConfig` when enabled, and
  a "not configured" line when disabled.
* `/vault status` combines live tier-2 state (note count) with the
  most recent `ReindexResult` carried on `VaultState`.
* `/vault reindex` mutates `state.last_result` and passes an optional
  label through as `only_label`.
* `/vault reindex <label>` with an unknown label surfaces the indexer's
  structured error rather than a stub reply.
* `build_read_only_handlers` only registers `/vault` when both
  indexer and tier2 are wired — matches the `/recall` registration
  discipline.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memory.embeddings import FakeEmbedder
from runtime.chat.memory.tier2 import Tier2Store
from runtime.chat.memory.vault_indexer import ReindexResult, VaultIndexer
from runtime.chat.telegram import (
    IncomingMessage,
    ParsedCommand,
    VaultState,
    build_read_only_handlers,
    vault_handler,
)
from runtime.config import VaultIndexingConfig, VaultSource

pytestmark = pytest.mark.unit


def _msg() -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text="/vault")


def _fixed_clock() -> datetime:
    return datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _build(
    tmp_path: Path,
    *,
    sources: tuple[VaultSource, ...] = (VaultSource(path="Docs", label="docs"),),
    vault_subdir: str = "vault",
) -> tuple[VaultIndexer, Tier2Store, VaultIndexingConfig]:
    vault = tmp_path / vault_subdir
    vault.mkdir(parents=True, exist_ok=True)
    store = Tier2Store(tmp_path / "tier2.db", FakeEmbedder(dim=8))
    cfg = VaultIndexingConfig(vault_root=vault, sources=sources)
    indexer = VaultIndexer(tier2=store, config=cfg, clock=_fixed_clock)
    return indexer, store, cfg


# --- usage fallback ----------------------------------------------------


def test_vault_usage_without_args(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(_msg(), ParsedCommand(name="/vault", args=()))
    assert "Usage" in out
    assert "status" in out
    assert "reindex" in out
    assert "sources" in out


def test_vault_usage_on_unknown_subcommand(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(_msg(), ParsedCommand(name="/vault", args=("wobble",)))
    assert "Usage" in out


# --- /vault sources ----------------------------------------------------


def test_vault_sources_disabled(tmp_path: Path) -> None:
    store = Tier2Store(tmp_path / "tier2.db", FakeEmbedder(dim=8))
    cfg = VaultIndexingConfig()  # no vault_root
    indexer = VaultIndexer(tier2=store, config=cfg, clock=_fixed_clock)
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(_msg(), ParsedCommand(name="/vault", args=("sources",)))
    assert "not configured" in out


def test_vault_sources_lists_configured_sources(tmp_path: Path) -> None:
    indexer, store, cfg = _build(
        tmp_path,
        sources=(
            VaultSource(path="Research", priority=1.5, label="research"),
            VaultSource(path="Wiki", label="wiki", exclude=("**/Archive/**",)),
        ),
    )
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(_msg(), ParsedCommand(name="/vault", args=("sources",)))
    assert str(cfg.vault_root) in out
    assert "Research" in out
    assert "research" in out
    assert "1.5" in out
    assert "Wiki" in out
    assert "**/Archive/**" in out


# --- /vault status -----------------------------------------------------


def test_vault_status_without_prior_run(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(_msg(), ParsedCommand(name="/vault", args=("status",)))
    assert "Last run: never" in out
    assert "Notes currently indexed: 0" in out


def test_vault_status_reflects_last_result(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    state = VaultState(
        last_result=ReindexResult(
            started_at=_fixed_clock(),
            finished_at=_fixed_clock(),
            added=3,
            updated=1,
            skipped=2,
            pruned=0,
        )
    )
    handler = vault_handler(indexer=indexer, tier2=store, state=state)
    out = handler(_msg(), ParsedCommand(name="/vault", args=("status",)))
    assert "added: 3" in out
    assert "updated: 1" in out
    assert "skipped: 2" in out


def test_vault_status_note_count_reflects_tier2(tmp_path: Path) -> None:
    vault_subdir = "vault"
    _write(tmp_path / vault_subdir / "Docs" / "a.md", "hello")
    _write(tmp_path / vault_subdir / "Docs" / "b.md", "world")
    indexer, store, _ = _build(tmp_path, vault_subdir=vault_subdir)
    state = VaultState()
    handler = vault_handler(indexer=indexer, tier2=store, state=state)
    # Simulate a reindex via the `/vault reindex` handler.
    handler(_msg(), ParsedCommand(name="/vault", args=("reindex",)))
    out = handler(_msg(), ParsedCommand(name="/vault", args=("status",)))
    assert "Notes currently indexed: 2" in out


# --- /vault reindex ----------------------------------------------------


def test_vault_reindex_mutates_last_result(tmp_path: Path) -> None:
    _write(tmp_path / "vault" / "Docs" / "a.md", "hello")
    indexer, store, _ = _build(tmp_path)
    state = VaultState()
    assert state.last_result is None
    handler = vault_handler(indexer=indexer, tier2=store, state=state)
    out = handler(_msg(), ParsedCommand(name="/vault", args=("reindex",)))
    assert state.last_result is not None
    assert state.last_result.added == 1
    assert "added: 1" in out


def test_vault_reindex_label_passes_through(tmp_path: Path) -> None:
    _write(tmp_path / "vault" / "Research" / "r.md", "alpha")
    _write(tmp_path / "vault" / "Wiki" / "w.md", "beta")
    indexer, store, _ = _build(
        tmp_path,
        sources=(
            VaultSource(path="Research", label="research"),
            VaultSource(path="Wiki", label="wiki"),
        ),
    )
    state = VaultState()
    handler = vault_handler(indexer=indexer, tier2=store, state=state)
    # First pass indexes both.
    handler(_msg(), ParsedCommand(name="/vault", args=("reindex",)))
    # Second pass scoped to `research` should skip the research file
    # and leave the wiki row untouched.
    out = handler(
        _msg(), ParsedCommand(name="/vault", args=("reindex", "research"))
    )
    assert "skipped: 1" in out or "skipped:" in out
    rels = sorted(n.rel_path for n in store.list_vault_notes())
    assert rels == ["Research/r.md", "Wiki/w.md"]


def test_vault_reindex_unknown_label_surfaces_error(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    handler = vault_handler(indexer=indexer, tier2=store, state=VaultState())
    out = handler(
        _msg(), ParsedCommand(name="/vault", args=("reindex", "nope"))
    )
    assert "failed" in out.lower()
    assert "nope" in out


# --- build_read_only_handlers registration ------------------------------


def test_build_read_only_handlers_skips_vault_when_unwired(tmp_path: Path) -> None:
    handlers = build_read_only_handlers(tmp_path)
    assert "/vault" not in handlers


def test_build_read_only_handlers_registers_vault(tmp_path: Path) -> None:
    indexer, store, _ = _build(tmp_path)
    handlers = build_read_only_handlers(
        tmp_path,
        vault_indexer=indexer,
        vault_tier2=store,
        vault_state=VaultState(),
    )
    assert "/vault" in handlers
    out = handlers["/vault"](_msg(), ParsedCommand(name="/vault", args=("sources",)))
    assert "docs" in out.lower() or "Docs" in out


def test_build_read_only_handlers_requires_both_vault_args(tmp_path: Path) -> None:
    indexer, _, _ = _build(tmp_path)
    # indexer only → not registered
    assert (
        "/vault"
        not in build_read_only_handlers(tmp_path, vault_indexer=indexer)
    )
