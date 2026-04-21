"""Phase 7 step 1 — Tier 3 rolling-window store contract.

Pins the behaviour:

* `Turn` is frozen + `extra="forbid"`.
* `append` assigns monotonic `turn_idx` per `chat_id`.
* Live window is bounded by `keep_turns` (default 12).
* Overflow turns move to a per-chat eviction queue; nothing is
  silently dropped.
* `drain_evicted` is destructive; `peek_evicted` is not.
* Multi-tenant isolation — chat A's turns are invisible to chat B.
* `clear(chat_id)` resets all three internal maps for that chat.
* `clock` is injectable so tests are deterministic.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from runtime.chat.memory import TIER3_KEEP_TURNS, Tier3Store, Turn

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)


class _StepClock:
    """Deterministic clock that advances 1 second per call."""

    def __init__(self, start: datetime = _NOW) -> None:
        self._t = start

    def __call__(self) -> datetime:
        ts = self._t
        self._t = self._t + timedelta(seconds=1)
        return ts


# --- Turn model invariants --------------------------------------------------


def test_turn_is_frozen() -> None:
    turn = Turn(chat_id="c1", turn_idx=0, role="user", text="hi", ts=_NOW)
    with pytest.raises(ValidationError):
        turn.text = "mutated"  # type: ignore[misc]


def test_turn_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Turn(  # type: ignore[call-arg]
            chat_id="c1", turn_idx=0, role="user", text="hi", ts=_NOW, extra="nope",
        )


def test_turn_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Turn(chat_id="c1", turn_idx=0, role="system", text="hi", ts=_NOW)  # type: ignore[arg-type]


def test_turn_rejects_negative_idx() -> None:
    with pytest.raises(ValidationError):
        Turn(chat_id="c1", turn_idx=-1, role="user", text="hi", ts=_NOW)


def test_turn_rejects_empty_chat_id() -> None:
    with pytest.raises(ValidationError):
        Turn(chat_id="", turn_idx=0, role="user", text="hi", ts=_NOW)


# --- append + recent --------------------------------------------------------


def test_default_keep_turns_matches_plan() -> None:
    assert TIER3_KEEP_TURNS == 12


def test_append_assigns_monotonic_idx() -> None:
    store = Tier3Store(clock=_StepClock())
    t0 = store.append("c1", "user", "first")
    t1 = store.append("c1", "bot", "reply")
    t2 = store.append("c1", "user", "second")
    assert (t0.turn_idx, t1.turn_idx, t2.turn_idx) == (0, 1, 2)


def test_append_uses_clock() -> None:
    clock = _StepClock(start=_NOW)
    store = Tier3Store(clock=clock)
    t0 = store.append("c1", "user", "first")
    t1 = store.append("c1", "bot", "second")
    assert t0.ts == _NOW
    assert t1.ts == _NOW + timedelta(seconds=1)


def test_recent_oldest_to_newest() -> None:
    store = Tier3Store(clock=_StepClock())
    for i in range(3):
        store.append("c1", "user", f"msg-{i}")
    recent = store.recent("c1")
    assert [t.text for t in recent] == ["msg-0", "msg-1", "msg-2"]


def test_append_rejects_empty_chat_id() -> None:
    store = Tier3Store()
    with pytest.raises(ValueError, match="chat_id"):
        store.append("", "user", "hi")


def test_constructor_rejects_zero_keep_turns() -> None:
    with pytest.raises(ValueError, match="keep_turns"):
        Tier3Store(keep_turns=0)


# --- bounded window + eviction ---------------------------------------------


def test_no_eviction_below_threshold() -> None:
    store = Tier3Store(keep_turns=5, clock=_StepClock())
    for i in range(5):
        store.append("c1", "user", f"msg-{i}")
    assert store.chat_size("c1") == 5
    assert store.peek_evicted("c1") == ()


def test_one_eviction_at_threshold_plus_one() -> None:
    store = Tier3Store(keep_turns=5, clock=_StepClock())
    for i in range(6):
        store.append("c1", "user", f"msg-{i}")
    assert store.chat_size("c1") == 5
    evicted = store.peek_evicted("c1")
    assert len(evicted) == 1
    assert evicted[0].text == "msg-0"
    # Live window holds the newest 5.
    assert [t.text for t in store.recent("c1")] == [f"msg-{i}" for i in range(1, 6)]


def test_many_evictions_preserve_order() -> None:
    store = Tier3Store(keep_turns=3, clock=_StepClock())
    for i in range(10):
        store.append("c1", "user", f"msg-{i}")
    # Live: last 3. Evicted: first 7, in original order.
    assert [t.text for t in store.recent("c1")] == ["msg-7", "msg-8", "msg-9"]
    assert [t.text for t in store.peek_evicted("c1")] == [f"msg-{i}" for i in range(7)]


def test_evicted_turns_keep_their_original_idx() -> None:
    store = Tier3Store(keep_turns=2, clock=_StepClock())
    for i in range(5):
        store.append("c1", "user", f"msg-{i}")
    evicted = store.peek_evicted("c1")
    assert [t.turn_idx for t in evicted] == [0, 1, 2]
    assert [t.turn_idx for t in store.recent("c1")] == [3, 4]


# --- drain semantics --------------------------------------------------------


def test_drain_evicted_returns_then_clears() -> None:
    store = Tier3Store(keep_turns=2, clock=_StepClock())
    for i in range(5):
        store.append("c1", "user", f"msg-{i}")
    drained = store.drain_evicted("c1")
    assert [t.text for t in drained] == ["msg-0", "msg-1", "msg-2"]
    assert store.peek_evicted("c1") == ()
    # Second drain is empty.
    assert store.drain_evicted("c1") == ()


def test_drain_evicted_unknown_chat_returns_empty() -> None:
    store = Tier3Store()
    assert store.drain_evicted("never-seen") == ()


def test_drain_does_not_disturb_live_window() -> None:
    store = Tier3Store(keep_turns=2, clock=_StepClock())
    for i in range(5):
        store.append("c1", "user", f"msg-{i}")
    store.drain_evicted("c1")
    assert [t.text for t in store.recent("c1")] == ["msg-3", "msg-4"]


# --- multi-tenant isolation -------------------------------------------------


def test_chats_are_isolated() -> None:
    store = Tier3Store(keep_turns=3, clock=_StepClock())
    store.append("alice", "user", "alice-1")
    store.append("bob", "user", "bob-1")
    store.append("alice", "user", "alice-2")
    assert [t.text for t in store.recent("alice")] == ["alice-1", "alice-2"]
    assert [t.text for t in store.recent("bob")] == ["bob-1"]


def test_turn_idx_is_per_chat() -> None:
    store = Tier3Store(clock=_StepClock())
    a0 = store.append("alice", "user", "a")
    b0 = store.append("bob", "user", "b")
    a1 = store.append("alice", "bot", "a-reply")
    assert (a0.turn_idx, a1.turn_idx) == (0, 1)
    assert b0.turn_idx == 0


def test_eviction_in_one_chat_does_not_affect_another() -> None:
    store = Tier3Store(keep_turns=2, clock=_StepClock())
    for i in range(5):
        store.append("alice", "user", f"a-{i}")
    store.append("bob", "user", "b-only")
    assert store.peek_evicted("bob") == ()
    assert len(store.peek_evicted("alice")) == 3


# --- cardinality + clear ----------------------------------------------------


def test_len_sums_live_across_chats() -> None:
    store = Tier3Store(keep_turns=10, clock=_StepClock())
    for i in range(3):
        store.append("a", "user", str(i))
    for i in range(2):
        store.append("b", "user", str(i))
    assert len(store) == 5


def test_clear_wipes_chat_state() -> None:
    store = Tier3Store(keep_turns=2, clock=_StepClock())
    for i in range(5):
        store.append("c1", "user", f"msg-{i}")
    store.clear("c1")
    assert store.recent("c1") == ()
    assert store.peek_evicted("c1") == ()
    # And turn_idx restarts from 0 after clear.
    new_turn = store.append("c1", "user", "fresh")
    assert new_turn.turn_idx == 0


def test_clear_unknown_chat_is_a_noop() -> None:
    store = Tier3Store()
    store.clear("never-seen")  # must not raise
