"""Tier 2 chat memory — episodic summaries + vault notes (RAG corpus).

Phase 7 build-order step 2. Reads/writes the four sqlite tables
defined in `memory/store_sqlite.py` v2 (`episodic_memory`,
`vault_note`, `episodic_embedding`, `vault_embedding`).

Contract (per `docs/PLAN_PHASE_7_TELEGRAM.md` §3.1, §3.3, §3.5):

* **EpisodicMemory** — one row per compressed session slice. Carries
  a nullable `ColdRef` pointing back to the JSONL on disk for
  verbatim recall.
* **VaultNote** — one row per indexed Obsidian-vault file, with
  `priority` ∈ [0.5, 4.0] and an optional `label` (carries the
  `MemorySourceSchema.label` field from `VaultIndexingConfig`).
* **ColdRef** — pointer + sha256 to a slice of a session JSONL.
  Verified at read time so disk corruption surfaces as a typed error.

Search uses cosine similarity on float32 BLOB embeddings. At AEGIS
scale (<10k vectors) a full table scan is fine and avoids the
sqlite-vec extension load. The interface is shaped so we can swap
in sqlite-vec virtual tables later without touching callers.

Tenant isolation: episodic search is scoped to `chat_id`. Vault
search is global (the vault is one shared corpus per operator).
"""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory.embeddings import Embedder, blob_to_vec, vec_to_blob
from memory.store_sqlite import ensure as ensure_db

# --- Models ----------------------------------------------------------------


class ColdRef(BaseModel):
    """Pointer into the canonical session JSONL on disk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    jsonl_path: str = Field(min_length=1)
    turn_range: tuple[int, int]
    sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _range_is_valid(self) -> ColdRef:
        start, end = self.turn_range
        if start < 0:
            raise ValueError("turn_range start must be >= 0")
        if end <= start:
            raise ValueError("turn_range end must be strictly greater than start")
        return self


class EpisodicMemory(BaseModel):
    """One compressed session slice. `cold_ref=None` after retention rolloff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_id: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime
    summary: str = Field(min_length=1)
    decisions_cited: tuple[str, ...] = ()
    imp_ids_cited: tuple[str, ...] = ()
    cold_ref: ColdRef | None = None

    @model_validator(mode="after")
    def _interval_ok(self) -> EpisodicMemory:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")
        return self


class VaultNote(BaseModel):
    """One indexed Obsidian-vault note. `rel_path` is the unique key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rel_path: str = Field(min_length=1)
    label: str | None = None
    priority: float = Field(default=1.0, ge=0.5, le=4.0)
    mtime: datetime
    body_sha256: str = Field(min_length=64, max_length=64)


# --- Search-result shapes --------------------------------------------------


class EpisodicHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record: EpisodicMemory
    score: float


class VaultHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record: VaultNote
    score: float


# --- Store -----------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"cosine dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


class Tier2Store:
    """Sqlite-backed episodic + vault store with cosine-similarity search.

    Not thread-safe. Phase 7 keeps one in-flight command per chat
    (plan §4.2); a per-chat lock is added when the dispatcher moves
    to a worker pool.
    """

    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self._db_path = ensure_db(db_path)
        self._embedder = embedder

    # -- internal --------------------------------------------------------

    def _conn(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        @contextlib.contextmanager
        def _ctx() -> Iterator[sqlite3.Connection]:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                with conn:
                    yield conn
            finally:
                conn.close()
        return _ctx()

    @staticmethod
    def _row_to_episodic(row: tuple[Any, ...]) -> EpisodicMemory:
        (
            _id,
            chat_id,
            started_at,
            ended_at,
            summary,
            decisions_cited,
            imp_ids_cited,
            cold_session_id,
            cold_jsonl_path,
            cold_turn_start,
            cold_turn_end,
            cold_sha256,
        ) = row
        cold: ColdRef | None = None
        if cold_session_id is not None:
            cold = ColdRef(
                session_id=cold_session_id,
                jsonl_path=cold_jsonl_path,
                turn_range=(cold_turn_start, cold_turn_end),
                sha256=cold_sha256,
            )
        return EpisodicMemory(
            chat_id=chat_id,
            started_at=datetime.fromisoformat(started_at),
            ended_at=datetime.fromisoformat(ended_at),
            summary=summary,
            decisions_cited=tuple(json.loads(decisions_cited)),
            imp_ids_cited=tuple(json.loads(imp_ids_cited)),
            cold_ref=cold,
        )

    @staticmethod
    def _row_to_vault(row: tuple[Any, ...]) -> VaultNote:
        _id, rel_path, label, priority, mtime, body_sha256 = row
        return VaultNote(
            rel_path=rel_path,
            label=label,
            priority=priority,
            mtime=datetime.fromisoformat(mtime),
            body_sha256=body_sha256,
        )

    # -- writes ----------------------------------------------------------

    def insert_episodic(self, record: EpisodicMemory) -> int:
        """Insert one episodic memory + its embedding. Returns row id."""
        vec = self._embedder.embed(record.summary)
        if len(vec) != self._embedder.dim:
            raise ValueError(
                f"embedder returned dim {len(vec)} but advertises {self._embedder.dim}"
            )
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodic_memory (
                    chat_id, started_at, ended_at, summary,
                    decisions_cited, imp_ids_cited,
                    cold_session_id, cold_jsonl_path,
                    cold_turn_start, cold_turn_end, cold_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.chat_id,
                    record.started_at.isoformat(),
                    record.ended_at.isoformat(),
                    record.summary,
                    json.dumps(list(record.decisions_cited)),
                    json.dumps(list(record.imp_ids_cited)),
                    record.cold_ref.session_id if record.cold_ref else None,
                    record.cold_ref.jsonl_path if record.cold_ref else None,
                    record.cold_ref.turn_range[0] if record.cold_ref else None,
                    record.cold_ref.turn_range[1] if record.cold_ref else None,
                    record.cold_ref.sha256 if record.cold_ref else None,
                ),
            )
            row_id = int(cur.lastrowid or 0)
            conn.execute(
                "INSERT INTO episodic_embedding (episodic_id, dim, vec) VALUES (?, ?, ?)",
                (row_id, self._embedder.dim, vec_to_blob(vec)),
            )
            conn.commit()
        return row_id

    def insert_vault_note(self, note: VaultNote, *, body_for_embedding: str) -> int:
        """Insert or replace a vault note + its embedding by `rel_path`.

        `body_for_embedding` is the text to embed (typically the note
        body). Kept separate from the metadata model so we don't have
        to keep large bodies in memory once indexed.
        """
        vec = self._embedder.embed(body_for_embedding)
        if len(vec) != self._embedder.dim:
            raise ValueError(
                f"embedder returned dim {len(vec)} but advertises {self._embedder.dim}"
            )
        with self._conn() as conn:
            # Replace by rel_path. Cascade drops the stale embedding.
            conn.execute("DELETE FROM vault_note WHERE rel_path = ?", (note.rel_path,))
            cur = conn.execute(
                """
                INSERT INTO vault_note (rel_path, label, priority, mtime, body_sha256)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    note.rel_path,
                    note.label,
                    note.priority,
                    note.mtime.isoformat(),
                    note.body_sha256,
                ),
            )
            row_id = int(cur.lastrowid or 0)
            conn.execute(
                "INSERT INTO vault_embedding (vault_id, dim, vec) VALUES (?, ?, ?)",
                (row_id, self._embedder.dim, vec_to_blob(vec)),
            )
            conn.commit()
        return row_id

    # -- reads -----------------------------------------------------------

    def search_episodic(
        self, *, chat_id: str, query: str, top_k: int = 5
    ) -> tuple[EpisodicHit, ...]:
        """Cosine search over episodic memories for one chat. Newest-first on ties."""
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q_vec = self._embedder.embed(query)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    em.id, em.chat_id, em.started_at, em.ended_at, em.summary,
                    em.decisions_cited, em.imp_ids_cited,
                    em.cold_session_id, em.cold_jsonl_path,
                    em.cold_turn_start, em.cold_turn_end, em.cold_sha256,
                    ee.dim, ee.vec
                FROM episodic_memory em
                JOIN episodic_embedding ee ON ee.episodic_id = em.id
                WHERE em.chat_id = ?
                """,
                (chat_id,),
            ).fetchall()
        scored: list[tuple[float, datetime, EpisodicMemory]] = []
        for row in rows:
            dim = row[12]
            vec = blob_to_vec(row[13], dim)
            score = _cosine(q_vec, vec)
            record = self._row_to_episodic(row[:12])
            scored.append((score, record.ended_at, record))
        scored.sort(key=lambda t: (-t[0], -t[1].timestamp()))
        return tuple(EpisodicHit(record=r, score=s) for s, _, r in scored[:top_k])

    def search_vault(
        self,
        *,
        query: str,
        top_k: int = 5,
        label_filter: str | None = None,
    ) -> tuple[VaultHit, ...]:
        """Cosine search over vault notes. Score is `cosine * priority`."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q_vec = self._embedder.embed(query)
        sql = (
            "SELECT vn.id, vn.rel_path, vn.label, vn.priority, vn.mtime, "
            "vn.body_sha256, ve.dim, ve.vec "
            "FROM vault_note vn JOIN vault_embedding ve ON ve.vault_id = vn.id"
        )
        params: tuple[object, ...] = ()
        if label_filter is not None:
            sql += " WHERE vn.label = ?"
            params = (label_filter,)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        scored: list[tuple[float, VaultNote]] = []
        for row in rows:
            dim = row[6]
            vec = blob_to_vec(row[7], dim)
            cos = _cosine(q_vec, vec)
            record = self._row_to_vault(row[:6])
            scored.append((cos * record.priority, record))
        scored.sort(key=lambda t: -t[0])
        return tuple(VaultHit(record=r, score=s) for s, r in scored[:top_k])

    # -- maintenance -----------------------------------------------------

    def all_episodic(self, chat_id: str) -> tuple[EpisodicMemory, ...]:
        """Return every episodic record for one chat (debug / migration use)."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, chat_id, started_at, ended_at, summary,
                       decisions_cited, imp_ids_cited,
                       cold_session_id, cold_jsonl_path,
                       cold_turn_start, cold_turn_end, cold_sha256
                FROM episodic_memory WHERE chat_id = ?
                ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()
        return tuple(self._row_to_episodic(r) for r in rows)

    def chat_ids(self) -> Iterable[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT chat_id FROM episodic_memory ORDER BY chat_id"
            ).fetchall()
        return tuple(r[0] for r in rows)

    def list_vault_notes(
        self, *, label: str | None = None
    ) -> tuple[VaultNote, ...]:
        """Return every indexed vault note, optionally filtered by label.

        Ordered by `rel_path` so callers (the vault indexer's prune
        pass) get a stable diff against the filesystem walk.
        """
        sql = (
            "SELECT id, rel_path, label, priority, mtime, body_sha256 "
            "FROM vault_note"
        )
        params: tuple[object, ...] = ()
        if label is not None:
            sql += " WHERE label = ?"
            params = (label,)
        sql += " ORDER BY rel_path ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(self._row_to_vault(r) for r in rows)

    def delete_vault_note(self, rel_path: str) -> bool:
        """Remove one vault note (cascade drops its embedding). Idempotent."""
        if not rel_path:
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM vault_note WHERE rel_path = ?", (rel_path,)
            )
            conn.commit()
        return cur.rowcount > 0


__all__ = [
    "ColdRef",
    "EpisodicHit",
    "EpisodicMemory",
    "Tier2Store",
    "VaultHit",
    "VaultNote",
]
