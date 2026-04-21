"""Phase 7 step 5 — Compressor contract.

Pins:

* Drains tier-3 evictions into one episodic row per batch.
* No-op when there are no evicted turns.
* Rejects mismatched `session_log.chat_id` — silent acceptance would
  poison tier 2 with the wrong cold pointer.
* On summarizer failure (exception or empty result) the batch is
  archived verbatim with `degraded=True`.
* Citation extraction picks up `DEC-###` and `IMP-<hex>` tokens from
  the summary and the turn bodies, deduped in first-seen order.
* ColdRef points at the canonical JSONL path + turn range + sha256
  that matches the source slice byte-for-byte.
* Acceptance test (plan §6 step 5): after >= 12 turns age out,
  compression archives them to tier 2, tier 3 eviction queue is empty,
  every new `EpisodicMemory` has a populated `cold_ref`, and the
  sha256 matches the on-disk slice byte-for-byte.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory.embeddings import DEFAULT_DIM, FakeEmbedder
from runtime.chat.memory import (
    CompressionResult,
    Compressor,
    Tier2Store,
    Tier3Store,
    Turn,
)
from runtime.chat.session_log import ChatSessionLog

pytestmark = pytest.mark.unit


class _FakeClock:
    def __init__(self) -> None:
        self._t = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        cur = self._t
        self._t += timedelta(seconds=1)
        return cur


class _StubSummarizer:
    def __init__(self, body: str = "summary body") -> None:
        self.body = body
        self.calls: list[Sequence[Turn]] = []

    def summarize(self, turns: Sequence[Turn]) -> str:
        self.calls.append(tuple(turns))
        return self.body


class _RaisingSummarizer:
    def summarize(self, turns: Sequence[Turn]) -> str:
        raise RuntimeError("local model is down")


class _EmptySummarizer:
    def summarize(self, turns: Sequence[Turn]) -> str:
        return "   "


@pytest.fixture
def tier2(tmp_path: Path) -> Tier2Store:
    return Tier2Store(
        tmp_path / "aegis-index.db",
        embedder=FakeEmbedder(dim=DEFAULT_DIM),
    )


@pytest.fixture
def tier3() -> Tier3Store:
    return Tier3Store(clock=_FakeClock())


@pytest.fixture
def session_log(tmp_path: Path) -> ChatSessionLog:
    return ChatSessionLog(tmp_path / "sessions", chat_id="c1", session_id="s1")


def _fill_live_window(t3: Tier3Store, chat_id: str, n: int) -> None:
    for i in range(n):
        t3.append(chat_id, "user" if i % 2 == 0 else "bot", f"turn-{i}")


def _mirror_to_log(evicted: Sequence[Turn], log: ChatSessionLog) -> None:
    for turn in evicted:
        log.append(turn)


# --- happy path ------------------------------------------------------------


def test_no_evicted_is_no_op(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer())
    result = c.compress_chat("c1", session_log)
    assert result == CompressionResult(
        chat_id="c1", batches=0, turns_compressed=0, degraded=False
    )
    assert tier2.all_episodic("c1") == ()


def test_mismatched_session_log_chat_id_is_rejected(
    tier2: Tier2Store, tier3: Tier3Store, tmp_path: Path
) -> None:
    log = ChatSessionLog(tmp_path / "sessions", chat_id="other", session_id="s1")
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer())
    with pytest.raises(ValueError, match="does not match compress target"):
        c.compress_chat("c1", log)


def test_compress_writes_one_episodic_with_cold_ref(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    _fill_live_window(tier3, "c1", 13)  # one eviction
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    summary = "DEC-001 noted; IMP-a1b2 applied"
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer(summary))
    result = c.compress_chat("c1", session_log)
    assert result.batches == 1
    assert result.turns_compressed == 1
    assert result.degraded is False
    assert len(result.episodic_ids) == 1
    records = tier2.all_episodic("c1")
    assert len(records) == 1
    rec = records[0]
    assert rec.summary == summary
    assert rec.decisions_cited == ("DEC-001",)
    assert rec.imp_ids_cited == ("IMP-a1b2",)
    assert rec.cold_ref is not None
    assert rec.cold_ref.session_id == "s1"
    assert rec.cold_ref.jsonl_path == str(session_log.path)
    assert rec.cold_ref.turn_range == (0, 1)
    expected_sha = hashlib.sha256(session_log.path.read_bytes()).hexdigest()
    assert rec.cold_ref.sha256 == expected_sha


def test_drain_is_cleared_after_compress(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    _fill_live_window(tier3, "c1", 15)
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer())
    c.compress_chat("c1", session_log)
    assert tier3.peek_evicted("c1") == ()


# --- degraded paths --------------------------------------------------------


def test_summarizer_exception_triggers_verbatim_archive(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    _fill_live_window(tier3, "c1", 14)  # 2 evictions
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_RaisingSummarizer())
    result = c.compress_chat("c1", session_log)
    assert result.degraded is True
    assert result.turns_compressed == 2
    rec = tier2.all_episodic("c1")[0]
    assert "user: turn-0" in rec.summary
    assert "bot: turn-1" in rec.summary
    assert rec.cold_ref is not None


def test_empty_summary_triggers_verbatim_archive(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    _fill_live_window(tier3, "c1", 13)
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_EmptySummarizer())
    result = c.compress_chat("c1", session_log)
    assert result.degraded is True
    rec = tier2.all_episodic("c1")[0]
    assert "user: turn-0" in rec.summary


# --- citation extraction ---------------------------------------------------


def test_citations_extracted_from_turn_bodies(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    # Make the first (and only) evicted turn carry citations in its body.
    tier3.append("c1", "user", "refs DEC-042 and IMP-deadbeef")
    for i in range(12):
        tier3.append("c1", "bot", f"filler {i}")
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    c = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer("no cites"))
    c.compress_chat("c1", session_log)
    rec = tier2.all_episodic("c1")[0]
    assert rec.decisions_cited == ("DEC-042",)
    assert rec.imp_ids_cited == ("IMP-deadbeef",)


def test_citations_deduped_first_seen_order(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    # Two evictions carrying the same IMP twice; DEC appears in summary only.
    tier3.append("c1", "user", "IMP-aaaa first")
    tier3.append("c1", "user", "IMP-aaaa and IMP-bbbb second")
    for i in range(12):
        tier3.append("c1", "bot", f"filler {i}")
    _mirror_to_log(tier3.peek_evicted("c1"), session_log)
    c = Compressor(
        tier3=tier3, tier2=tier2, summarizer=_StubSummarizer("DEC-007 closed")
    )
    c.compress_chat("c1", session_log)
    rec = tier2.all_episodic("c1")[0]
    assert rec.decisions_cited == ("DEC-007",)
    assert rec.imp_ids_cited == ("IMP-aaaa", "IMP-bbbb")


# --- acceptance: plan §6 step 5 -------------------------------------------


def test_acceptance_age_ge_12_turns_archives_byte_for_byte(
    tier2: Tier2Store, tier3: Tier3Store, session_log: ChatSessionLog
) -> None:
    # Append 24 turns so 12 get evicted from the live window.
    for i in range(24):
        tier3.append("c1", "user" if i % 2 == 0 else "bot", f"turn-{i}")
    evicted = tier3.peek_evicted("c1")
    assert len(evicted) == 12
    _mirror_to_log(evicted, session_log)
    c = Compressor(
        tier3=tier3, tier2=tier2, summarizer=_StubSummarizer("condensed"),
    )
    result = c.compress_chat("c1", session_log)

    # Drain is now empty — compressor consumed all pending.
    assert tier3.peek_evicted("c1") == ()
    assert result.turns_compressed == 12
    assert result.batches == 1
    assert result.degraded is False

    records = tier2.all_episodic("c1")
    assert len(records) == 1
    rec = records[0]
    assert rec.cold_ref is not None, "every episodic must carry a ColdRef"
    # Byte-for-byte verification over the exact on-disk slice.
    raw = session_log.path.read_bytes()
    expected_sha = hashlib.sha256(raw).hexdigest()
    assert rec.cold_ref.sha256 == expected_sha
    assert rec.cold_ref.turn_range == (0, 12)
    # And the live window still holds the remaining 12 newest turns.
    remaining = tier3.recent("c1")
    assert len(remaining) == 12
    assert remaining[0].turn_idx == 12
    assert remaining[-1].turn_idx == 23
