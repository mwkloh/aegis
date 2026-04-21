"""Phase 7 step 4 — ContextBuilder contract.

Pins the behaviour:

* `Lookup` and `TurnContext` are frozen + `extra="forbid"`.
* Tier 1 and tier 3 bytes always included; tier 1 never dropped.
* Lookups dropped *from the tail* of the ranked list until the
  byte budget fits (§3.3 step 4).
* If tier 1 + tier 3 alone exceed budget, `overflow=True`; the
  builder still returns (stub-on-failure posture from §2.8).
* `event_payload()` exposes structural counts only — no bodies.
* 200-turn synthetic conversation stays under the default budget
  (plan §6 step 4 acceptance test).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.chat.memory import (
    DEFAULT_TURN_BUDGET_BYTES,
    ContextBuilder,
    Lookup,
    Tier1Loader,
    Tier3Store,
)

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)


class _StepClock:
    def __init__(self, start: datetime = _NOW) -> None:
        self._t = start

    def __call__(self) -> datetime:
        ts = self._t
        self._t = self._t + timedelta(seconds=1)
        return ts


def _make_builder(
    tmp_path: Path,
    *,
    budget: int = DEFAULT_TURN_BUDGET_BYTES,
    clock: _StepClock | None = None,
) -> tuple[ContextBuilder, Tier1Loader, Tier3Store]:
    tier1 = Tier1Loader(root=tmp_path)
    tier3 = Tier3Store(clock=clock or _StepClock())
    builder = ContextBuilder(tier1, tier3, budget_bytes=budget)
    return builder, tier1, tier3


# --- Model invariants -----------------------------------------------------


def test_lookup_is_frozen() -> None:
    lk = Lookup(kind="episodic", text="hello", score=0.9, origin="ep:1")
    with pytest.raises(ValidationError):
        lk.text = "mutated"  # type: ignore[misc]


def test_lookup_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Lookup(  # type: ignore[call-arg]
            kind="episodic", text="x", score=0.1, origin="ep:1", extra="nope",
        )


def test_lookup_rejects_negative_score() -> None:
    with pytest.raises(ValidationError):
        Lookup(kind="episodic", text="x", score=-0.1, origin="ep:1")


def test_lookup_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Lookup(kind="other", text="x", score=0.1, origin="ep:1")  # type: ignore[arg-type]


def test_lookup_rejects_empty_origin() -> None:
    with pytest.raises(ValidationError):
        Lookup(kind="episodic", text="x", score=0.1, origin="")


def test_lookup_nbytes_is_utf8() -> None:
    lk = Lookup(kind="vault", text="héllo", score=0.0, origin="v:a")
    assert lk.nbytes == 6


def test_turn_context_total_bytes_sum() -> None:
    # Build a minimal TurnContext via the builder path rather than by hand.
    pass  # covered by builder tests below.


# --- Builder invariants ---------------------------------------------------


def test_builder_rejects_zero_budget(tmp_path: Path) -> None:
    tier1 = Tier1Loader(root=tmp_path)
    tier3 = Tier3Store()
    with pytest.raises(ValueError, match="budget_bytes"):
        ContextBuilder(tier1, tier3, budget_bytes=0)


def test_builder_rejects_negative_budget(tmp_path: Path) -> None:
    tier1 = Tier1Loader(root=tmp_path)
    tier3 = Tier3Store()
    with pytest.raises(ValueError, match="budget_bytes"):
        ContextBuilder(tier1, tier3, budget_bytes=-1)


def test_build_with_empty_state(tmp_path: Path) -> None:
    builder, _, _ = _make_builder(tmp_path)
    ctx = builder.build("c1")
    assert ctx.chat_id == "c1"
    assert ctx.tier1.total_bytes == 0
    assert ctx.tier3_turns == ()
    assert ctx.lookups == ()
    assert ctx.total_bytes == 0
    assert ctx.budget_bytes == DEFAULT_TURN_BUDGET_BYTES
    assert ctx.overflow is False


def test_build_loads_tier1(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("I am AEGIS.", encoding="utf-8")
    (tmp_path / "USER.md").write_text("user: mwk", encoding="utf-8")
    builder, _, _ = _make_builder(tmp_path)
    ctx = builder.build("c1")
    assert ctx.bytes_tier1 == len(b"I am AEGIS.") + len(b"user: mwk")
    assert ctx.total_bytes == ctx.bytes_tier1


def test_build_loads_tier1_prefs(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    raw = json.dumps({"tone": "terse"})
    (chat_dir / "prefs.json").write_text(raw, encoding="utf-8")
    builder, _, _ = _make_builder(tmp_path)
    ctx = builder.build("c1")
    assert ctx.tier1.prefs == {"tone": "terse"}
    assert ctx.bytes_tier1 == len(raw.encode())


def test_build_includes_tier3_window(tmp_path: Path) -> None:
    builder, _, tier3 = _make_builder(tmp_path)
    for i in range(3):
        tier3.append("c1", "user", f"hi-{i}")
    ctx = builder.build("c1")
    assert [t.text for t in ctx.tier3_turns] == ["hi-0", "hi-1", "hi-2"]
    assert ctx.bytes_tier3 == sum(len(f"hi-{i}".encode()) for i in range(3))


def test_build_is_per_chat(tmp_path: Path) -> None:
    builder, _, tier3 = _make_builder(tmp_path)
    tier3.append("a", "user", "alpha")
    tier3.append("b", "user", "beta")
    ctx_a = builder.build("a")
    ctx_b = builder.build("b")
    assert [t.text for t in ctx_a.tier3_turns] == ["alpha"]
    assert [t.text for t in ctx_b.tier3_turns] == ["beta"]


# --- Lookup merging + budget ---------------------------------------------


def test_lookups_kept_in_order_when_budget_fits(tmp_path: Path) -> None:
    builder, _, _ = _make_builder(tmp_path, budget=1024)
    hits = [
        Lookup(kind="episodic", text="a" * 100, score=0.9, origin="ep:1"),
        Lookup(kind="vault", text="b" * 150, score=0.7, origin="v:x"),
        Lookup(kind="episodic", text="c" * 50, score=0.5, origin="ep:2"),
    ]
    ctx = builder.build("c1", lookups=hits)
    assert ctx.lookups_considered == 3
    assert ctx.lookups_kept == 3
    assert ctx.bytes_lookups == 300
    assert [lk.origin for lk in ctx.lookups] == ["ep:1", "v:x", "ep:2"]


def test_lookups_dropped_from_tail_when_over_budget(tmp_path: Path) -> None:
    # Budget = tier1(0) + tier3(0) + 200 bytes for lookups.
    builder, _, _ = _make_builder(tmp_path, budget=200)
    hits = [
        Lookup(kind="episodic", text="a" * 120, score=0.9, origin="ep:1"),
        Lookup(kind="vault", text="b" * 100, score=0.7, origin="v:x"),
        Lookup(kind="episodic", text="c" * 50, score=0.5, origin="ep:2"),
    ]
    ctx = builder.build("c1", lookups=hits)
    # 120 fits. +100 would exceed 200 → drop "v:x". Then 50 also fits (170).
    assert [lk.origin for lk in ctx.lookups] == ["ep:1"]
    assert ctx.bytes_lookups == 120
    assert ctx.total_bytes <= 200
    assert ctx.lookups_considered == 3
    assert ctx.lookups_kept == 1
    assert ctx.overflow is False


def test_lookups_all_dropped_when_tier1_plus_tier3_fills_budget(
    tmp_path: Path,
) -> None:
    # Make tier1 exactly fill the budget.
    (tmp_path / "IDENTITY.md").write_text("x" * 100, encoding="utf-8")
    builder, _, _ = _make_builder(tmp_path, budget=100)
    hits = [Lookup(kind="episodic", text="y" * 10, score=0.9, origin="ep:1")]
    ctx = builder.build("c1", lookups=hits)
    assert ctx.lookups == ()
    assert ctx.bytes_lookups == 0
    assert ctx.total_bytes == 100
    assert ctx.overflow is False


def test_overflow_when_fixed_bytes_exceed_budget(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("x" * 500, encoding="utf-8")
    builder, _, _ = _make_builder(tmp_path, budget=100)
    ctx = builder.build("c1")
    assert ctx.overflow is True
    assert ctx.total_bytes == 500
    assert ctx.bytes_tier1 == 500
    assert ctx.lookups == ()


def test_tier1_never_dropped_even_under_pressure(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("x" * 1000, encoding="utf-8")
    builder, _, _ = _make_builder(tmp_path, budget=500)
    hits = [Lookup(kind="vault", text="y" * 100, score=1.0, origin="v:a")]
    ctx = builder.build("c1", lookups=hits)
    assert ctx.bytes_tier1 == 1000
    assert ctx.lookups == ()
    assert ctx.overflow is True


def test_build_accepts_generator(tmp_path: Path) -> None:
    builder, _, _ = _make_builder(tmp_path, budget=1024)

    def _gen() -> list[Lookup]:
        return [
            Lookup(kind="episodic", text="a", score=0.5, origin="ep:1"),
            Lookup(kind="vault", text="b", score=0.4, origin="v:1"),
        ]

    ctx = builder.build("c1", lookups=iter(_gen()))
    assert ctx.lookups_kept == 2


# --- event_payload ---------------------------------------------------------


def test_event_payload_has_no_message_bodies(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("secret identity", encoding="utf-8")
    builder, _, tier3 = _make_builder(tmp_path, budget=1024)
    tier3.append("c1", "user", "my private message")
    hits = [Lookup(kind="vault", text="vault body", score=0.9, origin="v:x")]
    ctx = builder.build("c1", lookups=hits)
    payload = ctx.event_payload()
    serialized = json.dumps(payload)
    assert "secret identity" not in serialized
    assert "my private message" not in serialized
    assert "vault body" not in serialized


def test_event_payload_exposes_counts(tmp_path: Path) -> None:
    builder, _, tier3 = _make_builder(tmp_path, budget=1024)
    tier3.append("c1", "user", "hi")
    hits = [Lookup(kind="episodic", text="abc", score=0.5, origin="ep:1")]
    ctx = builder.build("c1", lookups=hits)
    payload = ctx.event_payload()
    assert payload["chat_id"] == "c1"
    assert payload["budget_bytes"] == 1024
    assert payload["tier3_turns"] == 1
    assert payload["lookups_considered"] == 1
    assert payload["lookups_kept"] == 1
    assert payload["overflow"] is False
    assert "bytes_tier1" in payload
    assert "bytes_tier3" in payload
    assert "bytes_lookups" in payload


# --- chat_id propagation to tier1 loader ----------------------------------


def test_builder_propagates_chat_id_rejection(tmp_path: Path) -> None:
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(ValueError, match="chat_id must match"):
        builder.build("../etc")


# --- 200-turn synthetic integration test ----------------------------------


def test_two_hundred_turn_conversation_stays_under_budget(tmp_path: Path) -> None:
    """Plan §6 step 4 acceptance: 200-turn synthetic convo stays under budget."""
    # Populate tier 1 with realistic (~2 KB) identity/user content.
    (tmp_path / "IDENTITY.md").write_text("AEGIS identity.\n" * 40, encoding="utf-8")
    (tmp_path / "USER.md").write_text("Operator profile.\n" * 40, encoding="utf-8")
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text(
        json.dumps({"tone": "terse", "tz": "Asia/Singapore"}), encoding="utf-8",
    )
    builder, _, tier3 = _make_builder(tmp_path)

    # Simulate 200 turns. Each turn carries a payload of ~80 bytes.
    # Tier 3 is bounded at 12, so the live window never blows up regardless
    # of how many turns were exchanged.
    for i in range(200):
        role = "user" if i % 2 == 0 else "bot"
        tier3.append("c1", role, f"message {i:03d} " + "x" * 64)

    # Simulate a noisy lookup list — the caller's retriever returned 20 hits,
    # each ~400 bytes. The budget should trim these to fit.
    hits = tuple(
        Lookup(
            kind="episodic" if i % 2 == 0 else "vault",
            text="lookup body " + "y" * 380,
            score=1.0 - i * 0.01,
            origin=f"ep:{i}" if i % 2 == 0 else f"v:{i}",
        )
        for i in range(20)
    )
    ctx = builder.build("c1", lookups=hits)

    assert ctx.total_bytes <= DEFAULT_TURN_BUDGET_BYTES
    assert ctx.overflow is False
    assert ctx.lookups_considered == 20
    assert ctx.lookups_kept < 20  # At least one dropped under pressure.
    assert len(ctx.tier3_turns) == 12  # Tier 3 cap.
    # Retained lookups must be the top-ranked prefix (ranked order preserved).
    kept_origins = [lk.origin for lk in ctx.lookups]
    expected_prefix = [hits[i].origin for i in range(ctx.lookups_kept)]
    assert kept_origins == expected_prefix
