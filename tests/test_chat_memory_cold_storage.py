"""Phase 7 step 6 — `ColdStorageReader` contract.

Pins:

* Round-trip: compressor-archived turns → tier 2 ColdRef →
  `ColdStorageReader.read(ref)` returns the identical sequence
  (chat_id, turn_idx, role, text, ts).
* Typed errors:
    - missing file → `ColdStorageMissing`
    - sha256 drift → `ColdStorageMismatch`
    - missing turn_idx in range → `ColdStorageMismatch`
    - malformed JSONL line → `ColdStorageMismatch`
* The reader refuses to return partial success — any anomaly raises.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memory.embeddings import DEFAULT_DIM, FakeEmbedder
from runtime.chat.memory import (
    ColdRef,
    ColdStorageMismatch,
    ColdStorageMissing,
    ColdStorageRead,
    ColdStorageReader,
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
    def summarize(self, turns: Sequence[Turn]) -> str:
        del turns
        return "summary"


@pytest.fixture
def reader() -> ColdStorageReader:
    return ColdStorageReader()


@pytest.fixture
def tier2(tmp_path: Path) -> Tier2Store:
    return Tier2Store(
        tmp_path / "aegis-index.db",
        embedder=FakeEmbedder(dim=DEFAULT_DIM),
    )


@pytest.fixture
def session_log(tmp_path: Path) -> ChatSessionLog:
    return ChatSessionLog(tmp_path / "sessions", chat_id="c1", session_id="s1")


def _archive_turns(
    tier2: Tier2Store, session_log: ChatSessionLog, n_turns: int = 14
) -> ColdRef:
    """Append `n_turns`, mirror to disk, compress, return the stored ColdRef."""
    tier3 = Tier3Store(clock=_FakeClock())
    for i in range(n_turns):
        tier3.append("c1", "user" if i % 2 == 0 else "bot", f"turn-{i}")
    for turn in tier3.peek_evicted("c1"):
        session_log.append(turn)
    compressor = Compressor(tier3=tier3, tier2=tier2, summarizer=_StubSummarizer())
    compressor.compress_chat("c1", session_log)
    ref = tier2.all_episodic("c1")[0].cold_ref
    assert ref is not None
    return ref


# --- round-trip -----------------------------------------------------------


def test_round_trip_returns_archived_turns(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=14)
    read = reader.read(ref)
    assert isinstance(read, ColdStorageRead)
    assert read.ref == ref
    assert len(read.turns) == ref.turn_range[1] - ref.turn_range[0]
    for i, turn in enumerate(read.turns):
        assert turn.chat_id == "c1"
        assert turn.turn_idx == ref.turn_range[0] + i
        assert turn.text == f"turn-{turn.turn_idx}"
        assert turn.role == ("user" if turn.turn_idx % 2 == 0 else "bot")
        assert isinstance(turn.ts, datetime)


def test_turns_returned_sorted_by_turn_idx(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=24)
    read = reader.read(ref)
    indices = [t.turn_idx for t in read.turns]
    assert indices == sorted(indices)
    assert indices[0] == ref.turn_range[0]
    assert indices[-1] == ref.turn_range[1] - 1


def test_read_preserves_role_and_ts_types(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=14)
    read = reader.read(ref)
    for turn in read.turns:
        assert isinstance(turn, Turn)
        assert turn.role in {"user", "bot"}
        assert turn.ts.tzinfo is not None


# --- missing file ---------------------------------------------------------


def test_missing_file_raises_cold_storage_missing(
    reader: ColdStorageReader,
) -> None:
    ref = ColdRef(
        session_id="s1",
        jsonl_path="/nonexistent/path/does-not-exist.jsonl",
        turn_range=(0, 1),
        sha256="a" * 64,
    )
    with pytest.raises(ColdStorageMissing, match="cold JSONL not found"):
        reader.read(ref)


# --- sha mismatch ---------------------------------------------------------


def test_sha_drift_raises_mismatch(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=14)
    # Corrupt one line (but preserve turn_idx & structure) so the count
    # check passes and we reach the sha comparison.
    path = Path(ref.jsonl_path)
    raw = path.read_text(encoding="utf-8")
    corrupted = raw.replace('"turn-0"', '"turn-0-TAMPERED"')
    assert corrupted != raw, "test setup: replace must have taken effect"
    path.write_text(corrupted, encoding="utf-8")
    with pytest.raises(ColdStorageMismatch, match="sha256 mismatch"):
        reader.read(ref)


def test_missing_turn_idx_raises_mismatch(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=14)
    path = Path(ref.jsonl_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Drop the first line entirely — range still looks partly valid but
    # one turn_idx is gone.
    path.write_text("".join(lines[1:]), encoding="utf-8")
    with pytest.raises(ColdStorageMismatch, match="expects 2 records, found 1"):
        reader.read(ref)


def test_gap_in_range_raises_mismatch(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=24)
    path = Path(ref.jsonl_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Drop a middle line so count mismatches.
    kept = lines[:3] + lines[4:]
    path.write_text("".join(kept), encoding="utf-8")
    with pytest.raises(ColdStorageMismatch, match="expects"):
        reader.read(ref)


# --- malformed JSONL ------------------------------------------------------


def test_malformed_line_raises_mismatch(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    ref = _archive_turns(tier2, session_log, n_turns=14)
    path = Path(ref.jsonl_path)
    path.write_bytes(b"{not-json\n")
    with pytest.raises(ColdStorageMismatch, match="malformed JSONL"):
        reader.read(ref)


# --- non-conflicting foreign lines ----------------------------------------


def test_foreign_turn_idx_outside_range_is_ignored(
    reader: ColdStorageReader, tier2: Tier2Store, session_log: ChatSessionLog
) -> None:
    """A JSONL that also contains lines outside the archived range still verifies.

    This happens in production: the session log keeps growing after the
    compressor snapshots. The reader must scope to `turn_range` and
    ignore everything else — so long as all in-range indices are present
    and the in-range bytes hash correctly.
    """
    ref = _archive_turns(tier2, session_log, n_turns=14)
    # session_log only has the archived turn right now (we only mirrored
    # evicted turns). Append a foreign-looking line at the end.
    with Path(ref.jsonl_path).open("ab") as fh:
        fh.write(b'{"chat_id": "c1", "role": "user", "text": "later", '
                 b'"ts": "2027-01-01T00:00:00+00:00", "turn_idx": 999}\n')
    read = reader.read(ref)
    assert len(read.turns) == ref.turn_range[1] - ref.turn_range[0]
