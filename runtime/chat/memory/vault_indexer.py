"""Phase 7 Track C §5.1 — selective Obsidian-vault indexer.

Walks each configured `VaultSource` under `vault_root`, applies
`glob` + `exclude` filters, and upserts matching `.md` files into
`Tier2Store` with their `label` and `priority` carried on the row.
Retrieval ranking in `Tier2Store.search_vault` already multiplies
cosine similarity by `priority`, so "promoted" folders rank above
default notes.

Design invariants:

1. **Read-only vault.** The indexer never writes to `vault_root`.
   The file walk is pure `pathlib.Path.glob`; the only outbound
   writes are `Tier2Store.insert_vault_note` +
   `Tier2Store.delete_vault_note`.
2. **Incremental.** A file is re-embedded only if its mtime or
   sha256 changed since the last run. Embedding is the expensive
   step, so `last_indexed.body_sha256 == current_sha256` is the
   early-out gate.
3. **Prune on drift.** After the walk, every `vault_note` row
   whose `rel_path` is no longer on disk gets deleted (cascade
   removes the embedding). Files outside any configured source
   are also pruned — config changes (removing a source) shrink
   the corpus naturally on the next run.
4. **Never raise in the call path.** Per-file errors (bad UTF-8,
   missing permissions) accumulate into `ReindexResult.errors`
   and the walk continues. A missing `vault_root` returns an
   empty result with one `errors` entry rather than crashing.
5. **Deterministic test shape.** The `now` clock is injected so
   tests assert exact `ReindexResult.finished_at` values without
   freezegun gymnastics.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from runtime.chat.memory.tier2 import Tier2Store, VaultNote
from runtime.config import VaultIndexingConfig, VaultSource

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]

_MAX_BODY_BYTES = 512 * 1024  # hard cap; a single markdown note > 512 KB is pathological


@dataclass(frozen=True)
class _WalkedFile:
    """One markdown file picked up by the walk, pre-hash."""

    rel_path: str
    abs_path: Path
    mtime: datetime
    source: VaultSource


class ReindexResult(BaseModel):
    """Structured outcome of one `VaultIndexer.reindex()` call.

    Returned to `/vault status` and used by tests. Counters are
    plain ints — total files touched = added + updated + skipped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: datetime
    finished_at: datetime
    added: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    pruned: int = Field(default=0, ge=0)
    errors: tuple[str, ...] = ()

    @property
    def total_indexed(self) -> int:
        return self.added + self.updated + self.skipped


class VaultIndexer:
    """Drives incremental Obsidian-vault indexing into a `Tier2Store`.

    One instance per process. Callers invoke `reindex()` on startup
    and on the heartbeat cron (§5.1: `reindex_interval_hours`). The
    indexer is stateless between calls — all persistence lives in
    `Tier2Store`.
    """

    def __init__(
        self,
        *,
        tier2: Tier2Store,
        config: VaultIndexingConfig,
        clock: Clock | None = None,
    ) -> None:
        self._tier2 = tier2
        self._config = config
        self._clock: Clock = clock if clock is not None else _default_clock

    @property
    def config(self) -> VaultIndexingConfig:
        return self._config

    def reindex(self, *, only_label: str | None = None) -> ReindexResult:
        """Walk the vault, upsert changed files, prune stale rows.

        `only_label` scopes the walk to one source by its `label`
        — used by `/vault reindex <source>`. Pruning is ALSO scoped
        to that label so reindexing one source doesn't wipe notes
        from the other sources.
        """
        started = self._clock()
        if not self._config.is_enabled():
            return ReindexResult(
                started_at=started,
                finished_at=self._clock(),
                errors=("vault indexing not configured",),
            )
        sources = self._filter_sources(only_label)
        if not sources:
            return ReindexResult(
                started_at=started,
                finished_at=self._clock(),
                errors=(f"no source matched label {only_label!r}",),
            )
        errors: list[str] = []
        walked: dict[str, _WalkedFile] = {}
        for source in sources:
            try:
                for wf in self._walk_source(source):
                    walked[wf.rel_path] = wf
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"walk failed for {source.path!r}: {exc}")
        added, updated, skipped, embed_errors = self._upsert(walked)
        errors.extend(embed_errors)
        pruned = self._prune(walked.keys(), label=only_label)
        finished = self._clock()
        return ReindexResult(
            started_at=started,
            finished_at=finished,
            added=added,
            updated=updated,
            skipped=skipped,
            pruned=pruned,
            errors=tuple(errors),
        )

    def _filter_sources(self, only_label: str | None) -> tuple[VaultSource, ...]:
        if only_label is None:
            return self._config.sources
        return tuple(s for s in self._config.sources if s.label == only_label)

    def _walk_source(self, source: VaultSource) -> Iterator[_WalkedFile]:
        root = self._config.vault_root
        if root is None:
            return
        base = (root / source.path).expanduser()
        if not base.is_dir():
            return
        for match in base.glob(source.glob):
            if not match.is_file():
                continue
            if _is_excluded(match, base, source.exclude):
                continue
            rel = match.relative_to(root).as_posix()
            try:
                mtime = datetime.fromtimestamp(match.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            yield _WalkedFile(
                rel_path=rel, abs_path=match, mtime=mtime, source=source
            )

    def _upsert(
        self, walked: dict[str, _WalkedFile]
    ) -> tuple[int, int, int, list[str]]:
        existing = {
            note.rel_path: note for note in self._tier2.list_vault_notes()
        }
        added = updated = skipped = 0
        errors: list[str] = []
        for rel_path, wf in walked.items():
            try:
                body = _read_body(wf.abs_path)
            except Exception as exc:
                errors.append(f"read failed for {rel_path!r}: {exc}")
                continue
            sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            prior = existing.get(rel_path)
            if prior is not None and prior.body_sha256 == sha:
                skipped += 1
                continue
            note = VaultNote(
                rel_path=rel_path,
                label=wf.source.label,
                priority=wf.source.priority,
                mtime=wf.mtime,
                body_sha256=sha,
            )
            try:
                self._tier2.insert_vault_note(note, body_for_embedding=body)
            except Exception as exc:
                errors.append(f"embed failed for {rel_path!r}: {exc}")
                continue
            if prior is None:
                added += 1
            else:
                updated += 1
        return added, updated, skipped, errors

    def _prune(
        self, walked_paths: Iterable[str], *, label: str | None
    ) -> int:
        walked_set = set(walked_paths)
        pruned = 0
        for note in self._tier2.list_vault_notes(label=label):
            if note.rel_path in walked_set:
                continue
            if self._tier2.delete_vault_note(note.rel_path):
                pruned += 1
        return pruned


def _default_clock() -> datetime:
    return datetime.now(tz=UTC)


def _is_excluded(path: Path, base: Path, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    try:
        rel = path.relative_to(base).as_posix()
    except ValueError:
        return False
    return any(path.match(pat) or Path(rel).match(pat) for pat in patterns)


def _read_body(abs_path: Path) -> str:
    size = abs_path.stat().st_size
    if size > _MAX_BODY_BYTES:
        raise ValueError(f"note exceeds {_MAX_BODY_BYTES} bytes ({size} bytes)")
    return abs_path.read_text(encoding="utf-8")


class FilesystemVaultBodyLoader:
    """Resolves a vault `rel_path` (or slug) to its body text on disk.

    Implements the `VaultBodyLoader` protocol used by `RecallPolicy`
    and the `/recall vault:<slug>` handler. Accepts two key shapes:

    * `Research/Notes/phase-6-scope.md` — exact `rel_path` as stored
      in `vault_note.rel_path`.
    * `phase-6-scope` — a slug (filename stem). First matching file
      under `vault_root` wins; ambiguous slugs raise `LookupError`
      which the handler renders as "not found".

    Read-only: never writes, never modifies mtime. Body is returned
    decoded as UTF-8; decode errors surface as `UnicodeDecodeError`
    which `RecallPolicy` already swallows.
    """

    def __init__(self, vault_root: Path) -> None:
        self._root = Path(vault_root).expanduser()

    def load(self, rel_path: str) -> str:
        if not rel_path:
            return ""
        if not self._root.is_dir():
            return ""
        candidate = self._root / rel_path
        if candidate.is_file():
            return _read_body(candidate)
        # Slug fallback: match on filename stem.
        matches = [
            p for p in self._root.rglob("*.md") if p.stem == rel_path
        ]
        if not matches:
            return ""
        if len(matches) > 1:
            raise LookupError(
                f"ambiguous vault slug {rel_path!r}: {len(matches)} matches"
            )
        return _read_body(matches[0])


__all__ = [
    "FilesystemVaultBodyLoader",
    "ReindexResult",
    "VaultIndexer",
]
