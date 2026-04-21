"""Phase 7 step 7 — `RecallPolicy` contract.

Pins:

* Empty corpus → `()`.
* Empty / whitespace-only user_text → `()` (no embed call).
* Episodic search scoped per chat_id; cross-chat hits never leak.
* Episodic + vault results merge and sort by score descending.
* Score clamped to >= 0 so `Lookup` validation never trips on a
  negative cosine.
* Request-path safety: tier 2 raises → recall returns `()`, never
  propagates the exception. Vault loader raises → that hit becomes
  the `[vault pointer: ...]` stub; other hits still return.
* Without a `VaultBodyLoader`, vault hits surface as pointer stubs
  but still carry their origin + score so the trail is visible.
* `vault_label` filter is forwarded to tier 2.
* `top_k=0` for a source silently disables it.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from memory.embeddings import DEFAULT_DIM, FakeEmbedder
from runtime.chat.memory import (
    ColdRef,
    EpisodicHit,
    EpisodicMemory,
    Lookup,
    RecallPolicy,
    Tier2Store,
    VaultBodyLoader,
    VaultHit,
    VaultNote,
)

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)
_SHA = "a" * 64


def _cold(session_id: str = "s1") -> ColdRef:
    return ColdRef(
        session_id=session_id,
        jsonl_path=f"/var/aegis/{session_id}.jsonl",
        turn_range=(0, 2),
        sha256=_SHA,
    )


@pytest.fixture
def tier2(tmp_path: Path) -> Tier2Store:
    return Tier2Store(
        tmp_path / "aegis-index.db",
        embedder=FakeEmbedder(dim=DEFAULT_DIM),
    )


def _insert_episodic(
    store: Tier2Store, chat_id: str, summary: str
) -> EpisodicMemory:
    rec = EpisodicMemory(
        chat_id=chat_id,
        started_at=_NOW,
        ended_at=_NOW,
        summary=summary,
        cold_ref=_cold(),
    )
    store.insert_episodic(rec)
    return rec


def _insert_vault(
    store: Tier2Store,
    rel_path: str,
    *,
    body: str,
    label: str | None = None,
    priority: float = 1.0,
) -> None:
    note = VaultNote(
        rel_path=rel_path,
        label=label,
        priority=priority,
        mtime=_NOW,
        body_sha256=_SHA,
    )
    store.insert_vault_note(note, body_for_embedding=body)


# --- empty paths ----------------------------------------------------------


def test_empty_user_text_returns_empty(tier2: Tier2Store) -> None:
    _insert_episodic(tier2, "c1", "past chat about parsers")
    policy = RecallPolicy(tier2=tier2)
    assert policy.recall("c1", "") == ()
    assert policy.recall("c1", "   ") == ()


def test_empty_corpus_returns_empty(tier2: Tier2Store) -> None:
    policy = RecallPolicy(tier2=tier2)
    assert policy.recall("c1", "anything at all") == ()


def test_empty_chat_id_returns_empty(tier2: Tier2Store) -> None:
    policy = RecallPolicy(tier2=tier2)
    assert policy.recall("", "query") == ()


# --- basic search ---------------------------------------------------------


def test_episodic_hit_becomes_lookup(tier2: Tier2Store) -> None:
    _insert_episodic(tier2, "c1", "we agreed on DEC-001")
    policy = RecallPolicy(tier2=tier2, top_vault=0)
    out = policy.recall("c1", "DEC-001 status")
    assert len(out) == 1
    lk = out[0]
    assert lk.kind == "episodic"
    assert lk.text == "we agreed on DEC-001"
    assert lk.score >= 0.0
    assert lk.origin.startswith("episodic:c1:")


def test_episodic_isolated_per_chat(tier2: Tier2Store) -> None:
    _insert_episodic(tier2, "c1", "chat one topic")
    _insert_episodic(tier2, "c2", "chat two topic")
    policy = RecallPolicy(tier2=tier2, top_vault=0)
    out = policy.recall("c1", "one topic")
    assert len(out) == 1
    assert out[0].origin.startswith("episodic:c1:")


def test_vault_hit_without_loader_is_pointer(tier2: Tier2Store) -> None:
    _insert_vault(tier2, "notes/foo.md", body="hello world")
    policy = RecallPolicy(tier2=tier2, top_episodic=0)
    out = policy.recall("c1", "hello world")
    assert len(out) == 1
    lk = out[0]
    assert lk.kind == "vault"
    assert lk.text == "[vault pointer: notes/foo.md]"
    assert lk.origin == "vault:notes/foo.md"


def test_vault_loader_returns_body(tier2: Tier2Store) -> None:
    _insert_vault(tier2, "notes/foo.md", body="hello world")

    class _Loader:
        def load(self, rel_path: str) -> str:
            assert rel_path == "notes/foo.md"
            return "loaded body"

    policy = RecallPolicy(
        tier2=tier2,
        vault_loader=cast(VaultBodyLoader, _Loader()),
        top_episodic=0,
    )
    out = policy.recall("c1", "hello world")
    assert out[0].text == "loaded body"


def test_vault_label_filter_is_forwarded(tier2: Tier2Store) -> None:
    _insert_vault(tier2, "a.md", body="alpha", label="stable")
    _insert_vault(tier2, "b.md", body="alpha", label="draft")
    policy = RecallPolicy(tier2=tier2, top_episodic=0)
    out = policy.recall("c1", "alpha", vault_label="stable")
    assert len(out) == 1
    assert out[0].origin == "vault:a.md"


# --- merge + ordering -----------------------------------------------------


def test_merged_results_sorted_by_score_desc(tier2: Tier2Store) -> None:
    _insert_episodic(tier2, "c1", "apple banana cherry")
    _insert_vault(tier2, "note.md", body="apple banana cherry")
    policy = RecallPolicy(tier2=tier2, top_episodic=3, top_vault=3)
    out = policy.recall("c1", "apple banana cherry")
    assert len(out) >= 2
    scores = [lk.score for lk in out]
    assert scores == sorted(scores, reverse=True)


# --- failure safety -------------------------------------------------------


def test_tier2_episodic_failure_returns_empty_not_raise(
    tier2: Tier2Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(**_kwargs: object) -> tuple[EpisodicHit, ...]:
        raise RuntimeError("sqlite is gone")

    monkeypatch.setattr(tier2, "search_episodic", _boom)
    policy = RecallPolicy(tier2=tier2, top_vault=0)
    assert policy.recall("c1", "query") == ()


def test_tier2_vault_failure_is_isolated_from_episodic(
    tier2: Tier2Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    _insert_episodic(tier2, "c1", "past chat about parsers")

    def _boom(**_kwargs: object) -> tuple[VaultHit, ...]:
        raise RuntimeError("sqlite is gone")

    monkeypatch.setattr(tier2, "search_vault", _boom)
    policy = RecallPolicy(tier2=tier2)
    out = policy.recall("c1", "parsers")
    assert len(out) == 1
    assert out[0].kind == "episodic"


def test_vault_loader_exception_degrades_to_pointer(tier2: Tier2Store) -> None:
    _insert_vault(tier2, "notes/foo.md", body="alpha")

    class _BoomLoader:
        def load(self, rel_path: str) -> str:
            raise OSError(f"disk flaky: {rel_path}")

    policy = RecallPolicy(
        tier2=tier2,
        vault_loader=cast(VaultBodyLoader, _BoomLoader()),
        top_episodic=0,
    )
    out = policy.recall("c1", "alpha")
    assert out[0].text == "[vault pointer: notes/foo.md]"


def test_vault_loader_empty_body_degrades_to_pointer(tier2: Tier2Store) -> None:
    _insert_vault(tier2, "notes/foo.md", body="alpha")

    class _EmptyLoader:
        def load(self, rel_path: str) -> str:
            del rel_path
            return ""

    policy = RecallPolicy(
        tier2=tier2,
        vault_loader=cast(VaultBodyLoader, _EmptyLoader()),
        top_episodic=0,
    )
    out = policy.recall("c1", "alpha")
    assert out[0].text == "[vault pointer: notes/foo.md]"


# --- score clamping -------------------------------------------------------


def test_negative_cosine_is_clamped_to_zero(
    tier2: Tier2Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Lookup.score` forbids negatives. Policy must clamp."""
    rec = _insert_episodic(tier2, "c1", "ignored")
    fake_hit = EpisodicHit(record=rec, score=-0.5)

    def _fake_search(**_kwargs: object) -> tuple[EpisodicHit, ...]:
        return (fake_hit,)

    monkeypatch.setattr(tier2, "search_episodic", _fake_search)
    policy = RecallPolicy(tier2=tier2, top_vault=0)
    out = policy.recall("c1", "anything")
    assert len(out) == 1
    assert isinstance(out[0], Lookup)
    assert out[0].score == 0.0


# --- knobs ---------------------------------------------------------------


def test_top_k_zero_disables_that_source(tier2: Tier2Store) -> None:
    _insert_episodic(tier2, "c1", "past chat")
    _insert_vault(tier2, "note.md", body="alpha")
    policy = RecallPolicy(tier2=tier2, top_episodic=0, top_vault=1)
    out = policy.recall("c1", "past chat alpha")
    for lk in out:
        assert lk.kind == "vault"


@pytest.mark.parametrize(("top_episodic", "top_vault"), [(-1, 0), (0, -1)])
def test_negative_top_k_is_rejected(
    tier2: Tier2Store, top_episodic: int, top_vault: int
) -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        RecallPolicy(
            tier2=tier2, top_episodic=top_episodic, top_vault=top_vault
        )


# --- integration with ContextBuilder -------------------------------------


def test_recall_output_plugs_into_context_builder_shape(
    tier2: Tier2Store,
) -> None:
    """The policy must emit types the context builder will accept.

    (The builder test suite covers budgeting; here we just confirm the
    shape handshake — all items are `Lookup` instances that satisfy
    its field validation.)
    """
    _insert_episodic(tier2, "c1", "past chat about DEC-007")
    policy = RecallPolicy(tier2=tier2, top_vault=0)
    lookups: Sequence[Lookup] = policy.recall("c1", "DEC-007 please")
    for lk in lookups:
        assert isinstance(lk, Lookup)
        assert lk.nbytes > 0
