"""Phase 7 step 2 — Tier 2 store contract.

Pins:

* Schema migration v0/v1 → v2 is idempotent.
* Pydantic models are frozen + `extra="forbid"` and reject malformed
  cold refs / time intervals.
* `FakeEmbedder` is deterministic and L2-normalized.
* Episodic insert + search round-trips and is scoped per `chat_id`.
* Vault insert + search round-trips, replaces by `rel_path`, and
  weights cosine by `priority` so `priority=2.0` outranks `priority=1.0`
  on equally-similar text.
* `label_filter` restricts vault search to one source label.
* Cold ref is fully optional and serialized with sha256 round-trip.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from memory.embeddings import (
    DEFAULT_DIM,
    Bgem3Embedder,
    Embedder,
    FakeEmbedder,
    blob_to_vec,
    vec_to_blob,
)
from memory.store_sqlite import SCHEMA_VERSION, ensure
from runtime.chat.memory import (
    ColdRef,
    EpisodicMemory,
    Tier2Store,
    VaultNote,
)

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
_SHA = "a" * 64


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "aegis-index.db"


@pytest.fixture
def store(db_path: Path) -> Tier2Store:
    return Tier2Store(db_path, embedder=FakeEmbedder(dim=DEFAULT_DIM))


# --- schema migration ------------------------------------------------------


def test_schema_version_is_three() -> None:
    assert SCHEMA_VERSION == 3


def test_ensure_creates_v2_tables(db_path: Path) -> None:
    ensure(db_path)
    with sqlite3.connect(db_path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {
        "schema_version",
        "episodic_memory",
        "vault_note",
        "episodic_embedding",
        "vault_embedding",
    }.issubset(names)


def test_ensure_is_idempotent(db_path: Path) -> None:
    ensure(db_path)
    ensure(db_path)
    ensure(db_path)
    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
    assert version == 3


def test_migrates_from_v1_dbs(db_path: Path) -> None:
    # Simulate an existing v1 DB (Phase 0 layout: only schema_version table).
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version(version) VALUES (1)")
        conn.commit()
    ensure(db_path)
    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert version == 3
    assert "episodic_memory" in names
    assert "scheduled_jobs" in names


# --- model invariants ------------------------------------------------------


def test_episodic_memory_is_frozen() -> None:
    em = EpisodicMemory(
        chat_id="c1",
        started_at=_NOW,
        ended_at=_NOW + timedelta(seconds=1),
        summary="hi",
    )
    with pytest.raises(ValidationError):
        em.summary = "mutated"  # type: ignore[misc]


def test_episodic_memory_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EpisodicMemory(  # type: ignore[call-arg]
            chat_id="c1",
            started_at=_NOW,
            ended_at=_NOW + timedelta(seconds=1),
            summary="hi",
            extra="nope",
        )


def test_episodic_memory_rejects_inverted_interval() -> None:
    with pytest.raises(ValidationError):
        EpisodicMemory(
            chat_id="c1",
            started_at=_NOW + timedelta(seconds=2),
            ended_at=_NOW,
            summary="hi",
        )


def test_cold_ref_rejects_bad_range() -> None:
    with pytest.raises(ValidationError):
        ColdRef(session_id="s1", jsonl_path="x.jsonl", turn_range=(5, 5), sha256=_SHA)
    with pytest.raises(ValidationError):
        ColdRef(session_id="s1", jsonl_path="x.jsonl", turn_range=(-1, 3), sha256=_SHA)


def test_cold_ref_requires_64_hex_sha() -> None:
    with pytest.raises(ValidationError):
        ColdRef(session_id="s1", jsonl_path="x.jsonl", turn_range=(0, 1), sha256="short")


def test_vault_note_rejects_priority_out_of_range() -> None:
    with pytest.raises(ValidationError):
        VaultNote(
            rel_path="a.md",
            mtime=_NOW,
            body_sha256=_sha256("body"),
            priority=0.4,
        )
    with pytest.raises(ValidationError):
        VaultNote(
            rel_path="a.md",
            mtime=_NOW,
            body_sha256=_sha256("body"),
            priority=4.5,
        )


# --- embedder contract -----------------------------------------------------


def test_fake_embedder_is_deterministic() -> None:
    e = FakeEmbedder(dim=32)
    assert e.embed("hello world") == e.embed("hello world")


def test_fake_embedder_diverges_on_different_text() -> None:
    e = FakeEmbedder(dim=32)
    assert e.embed("apples") != e.embed("oranges")


def test_fake_embedder_is_l2_normalized() -> None:
    vec = FakeEmbedder(dim=64).embed("anything")
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_fake_embedder_dim_matches_advertised() -> None:
    e = FakeEmbedder(dim=128)
    assert e.dim == 128
    assert len(e.embed("x")) == 128


def test_fake_embedder_rejects_zero_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        FakeEmbedder(dim=0)


def test_blob_round_trip() -> None:
    vec = [0.1, -0.2, 0.3, -0.4]
    blob = vec_to_blob(vec)
    assert len(blob) == len(vec) * 4
    out = blob_to_vec(blob, dim=4)
    for a, b in zip(vec, out, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_blob_to_vec_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="blob length"):
        blob_to_vec(b"\x00" * 12, dim=4)  # 12 bytes != 4*4


def test_embedder_protocol_recognizes_fake_and_bgem3() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(Bgem3Embedder(), Embedder)


# --- episodic insert + search ---------------------------------------------


def _episodic(
    *,
    chat_id: str,
    summary: str,
    cold: ColdRef | None = None,
    decisions: tuple[str, ...] = (),
    imps: tuple[str, ...] = (),
) -> EpisodicMemory:
    return EpisodicMemory(
        chat_id=chat_id,
        started_at=_NOW,
        ended_at=_NOW + timedelta(seconds=10),
        summary=summary,
        decisions_cited=decisions,
        imp_ids_cited=imps,
        cold_ref=cold,
    )


def test_episodic_round_trip_preserves_fields(store: Tier2Store) -> None:
    cold = ColdRef(
        session_id="sess-1",
        jsonl_path="2026-04-18/sess-1.jsonl",
        turn_range=(0, 12),
        sha256=_SHA,
    )
    record = _episodic(
        chat_id="c1",
        summary="planning the harness flow",
        cold=cold,
        decisions=("D-001", "D-002"),
        imps=("CT-007",),
    )
    store.insert_episodic(record)
    [out] = store.all_episodic("c1")
    assert out.summary == record.summary
    assert out.decisions_cited == ("D-001", "D-002")
    assert out.imp_ids_cited == ("CT-007",)
    assert out.cold_ref == cold


def test_episodic_search_finds_best_match(store: Tier2Store) -> None:
    store.insert_episodic(_episodic(chat_id="c1", summary="harness flow design"))
    store.insert_episodic(_episodic(chat_id="c1", summary="vault indexing notes"))
    hits = store.search_episodic(chat_id="c1", query="harness flow design")
    assert len(hits) == 2
    assert hits[0].record.summary == "harness flow design"
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[0].score > hits[1].score


def test_episodic_search_is_chat_scoped(store: Tier2Store) -> None:
    store.insert_episodic(_episodic(chat_id="alice", summary="alice secret"))
    store.insert_episodic(_episodic(chat_id="bob", summary="bob secret"))
    alice_hits = store.search_episodic(chat_id="alice", query="secret")
    bob_hits = store.search_episodic(chat_id="bob", query="secret")
    assert {h.record.summary for h in alice_hits} == {"alice secret"}
    assert {h.record.summary for h in bob_hits} == {"bob secret"}


def test_episodic_search_respects_top_k(store: Tier2Store) -> None:
    for i in range(7):
        store.insert_episodic(_episodic(chat_id="c1", summary=f"summary-{i}"))
    hits = store.search_episodic(chat_id="c1", query="summary-3", top_k=3)
    assert len(hits) == 3


def test_episodic_search_rejects_empty_chat_id(store: Tier2Store) -> None:
    with pytest.raises(ValueError, match="chat_id"):
        store.search_episodic(chat_id="", query="x")


def test_episodic_cold_ref_optional(store: Tier2Store) -> None:
    store.insert_episodic(_episodic(chat_id="c1", summary="no cold ref", cold=None))
    [out] = store.all_episodic("c1")
    assert out.cold_ref is None


# --- vault insert + search -------------------------------------------------


def _note(
    rel_path: str,
    *,
    label: str | None = None,
    priority: float = 1.0,
    body: str = "body",
) -> tuple[VaultNote, str]:
    note = VaultNote(
        rel_path=rel_path,
        label=label,
        priority=priority,
        mtime=_NOW,
        body_sha256=_sha256(body),
    )
    return note, body


def test_vault_round_trip(store: Tier2Store) -> None:
    note, body = _note("daily/2026-04-18.md", label="daily", priority=1.5)
    store.insert_vault_note(note, body_for_embedding=body)
    [out] = store.search_vault(query=body)
    assert out.record == note


def test_vault_search_label_filter(store: Tier2Store) -> None:
    n1, b1 = _note("daily/d1.md", label="daily", body="harness drafting flow")
    n2, b2 = _note("ref/r1.md", label="reference", body="harness drafting flow")
    store.insert_vault_note(n1, body_for_embedding=b1)
    store.insert_vault_note(n2, body_for_embedding=b2)
    daily_hits = store.search_vault(query="harness drafting flow", label_filter="daily")
    assert {h.record.label for h in daily_hits} == {"daily"}


def test_vault_priority_breaks_ties(store: Tier2Store) -> None:
    body = "shared body text"
    low, body_l = _note("a.md", label="a", priority=1.0, body=body)
    high, body_h = _note("b.md", label="b", priority=2.0, body=body)
    store.insert_vault_note(low, body_for_embedding=body_l)
    store.insert_vault_note(high, body_for_embedding=body_h)
    hits = store.search_vault(query=body)
    # Same cosine; priority 2.0 should rank first.
    assert hits[0].record.rel_path == "b.md"
    assert hits[1].record.rel_path == "a.md"


def test_vault_replace_by_rel_path(store: Tier2Store) -> None:
    note_v1, body = _note("a.md", priority=1.0, body="v1 body")
    store.insert_vault_note(note_v1, body_for_embedding=body)
    note_v2 = VaultNote(
        rel_path="a.md",
        label="updated",
        priority=2.5,
        mtime=_NOW + timedelta(hours=1),
        body_sha256=_sha256("v2 body"),
    )
    store.insert_vault_note(note_v2, body_for_embedding="v2 body")
    hits = store.search_vault(query="v2 body")
    assert len(hits) == 1
    assert hits[0].record.label == "updated"
    assert hits[0].record.priority == pytest.approx(2.5)


def test_vault_search_rejects_zero_top_k(store: Tier2Store) -> None:
    with pytest.raises(ValueError, match="top_k"):
        store.search_vault(query="x", top_k=0)


# --- chat_ids surface ------------------------------------------------------


def test_chat_ids_lists_all_distinct(store: Tier2Store) -> None:
    store.insert_episodic(_episodic(chat_id="alice", summary="a"))
    store.insert_episodic(_episodic(chat_id="alice", summary="aa"))
    store.insert_episodic(_episodic(chat_id="bob", summary="b"))
    assert tuple(store.chat_ids()) == ("alice", "bob")
