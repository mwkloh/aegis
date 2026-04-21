"""Phase 7 step 5 — background compression of tier-3 evictions into tier 2.

Runs out-of-band (cron + `make compress`), never inside a turn. Per
`docs/PLAN_PHASE_7_TELEGRAM.md` §3.2 and §3.5:

* Drains `Tier3Store.drain_evicted(chat_id)` in FIFO order. Evicted
  turns are contiguous by construction (tier 3 pops the head).
* Calls an injected `Summarizer` (local model only — never frontier).
  On any exception the batch is archived verbatim with
  `degraded=True`; we never lose turns to a flaky summarizer.
* Extracts `DEC-###` and `IMP-<hex>` citations from the summary + the
  raw turn bodies so a later recall can trace the claim.
* Builds a `ColdRef` by hashing the exact JSONL slice on disk via
  `ChatSessionLog.slice_sha256` — byte-for-byte reproducible across
  OSes because the writer forces `"\\n"` terminators.
* Writes one `EpisodicMemory` per batch via `Tier2Store.insert_episodic`.

Single-threaded; callers serialize per chat_id. No events emitted here
(the dispatcher logs `chat.turn.compressed` around the call).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from runtime.chat.memory.tier2 import ColdRef, EpisodicMemory, Tier2Store
from runtime.chat.memory.tier3 import Tier3Store, Turn
from runtime.chat.session_log import ChatSessionLog

_DEC_RE = re.compile(r"\bDEC-\d+\b")
_IMP_RE = re.compile(r"\bIMP-[A-Fa-f0-9]+\b")


class Summarizer(Protocol):
    """Callable that condenses a batch of turns to a single summary string.

    Implementations MUST be local-tier models per plan §3.2. Raising
    any exception triggers the degraded verbatim-archive path; the
    compressor never retries inline.
    """

    def summarize(self, turns: Sequence[Turn]) -> str: ...


class CompressionResult(BaseModel):
    """Outcome of one `compress_chat` call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_id: str = Field(min_length=1)
    batches: int = Field(ge=0)
    turns_compressed: int = Field(ge=0)
    degraded: bool
    episodic_ids: tuple[int, ...] = ()


class Compressor:
    """Drains evicted turns from tier 3 → writes episodic rows to tier 2."""

    def __init__(
        self,
        *,
        tier3: Tier3Store,
        tier2: Tier2Store,
        summarizer: Summarizer,
    ) -> None:
        self._tier3 = tier3
        self._tier2 = tier2
        self._summarizer = summarizer

    def compress_chat(
        self, chat_id: str, session_log: ChatSessionLog
    ) -> CompressionResult:
        """Compress every pending-evicted turn for `chat_id` into one batch.

        The caller supplies the active `ChatSessionLog` for this chat
        because the compressor computes `ColdRef.sha256` over the
        canonical on-disk JSONL slice. Mismatched `session_log.chat_id`
        is rejected — silent acceptance would poison the tier 2 row
        with the wrong cold pointer.
        """
        if session_log.chat_id != chat_id:
            raise ValueError(
                f"session_log.chat_id={session_log.chat_id!r} does not match "
                f"compress target chat_id={chat_id!r}"
            )
        turns = self._tier3.drain_evicted(chat_id)
        if not turns:
            return CompressionResult(
                chat_id=chat_id,
                batches=0,
                turns_compressed=0,
                degraded=False,
            )
        degraded, summary = self._summarize_or_verbatim(turns)
        start_idx = turns[0].turn_idx
        end_idx = turns[-1].turn_idx + 1
        nbytes, sha = session_log.slice_sha256(start_idx, end_idx)
        if nbytes == 0:
            raise ValueError("session log slice is empty — refusing to archive")
        cold = ColdRef(
            session_id=session_log.session_id,
            jsonl_path=str(session_log.path),
            turn_range=(start_idx, end_idx),
            sha256=sha,
        )
        decisions, imp_ids = _extract_citations(summary, turns)
        record = EpisodicMemory(
            chat_id=chat_id,
            started_at=turns[0].ts,
            ended_at=turns[-1].ts,
            summary=summary,
            decisions_cited=decisions,
            imp_ids_cited=imp_ids,
            cold_ref=cold,
        )
        episodic_id = self._tier2.insert_episodic(record)
        return CompressionResult(
            chat_id=chat_id,
            batches=1,
            turns_compressed=len(turns),
            degraded=degraded,
            episodic_ids=(episodic_id,),
        )

    def _summarize_or_verbatim(
        self, turns: Sequence[Turn]
    ) -> tuple[bool, str]:
        try:
            summary = self._summarizer.summarize(turns)
        except Exception:
            return True, _verbatim(turns)
        if not summary.strip():
            return True, _verbatim(turns)
        return False, summary


def _verbatim(turns: Sequence[Turn]) -> str:
    """Degraded archive body. One line per turn, `role: text`."""
    return "\n".join(f"{t.role}: {t.text}" for t in turns)


def _extract_citations(
    summary: str, turns: Sequence[Turn]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Unique, order-preserving `DEC-###` and `IMP-<hex>` ids from summary + turns."""
    decisions: list[str] = []
    imps: list[str] = []
    dec_seen: set[str] = set()
    imp_seen: set[str] = set()
    sources = [summary, *(t.text for t in turns)]
    for src in sources:
        for match in _DEC_RE.findall(src):
            if match not in dec_seen:
                dec_seen.add(match)
                decisions.append(match)
        for match in _IMP_RE.findall(src):
            if match not in imp_seen:
                imp_seen.add(match)
                imps.append(match)
    return tuple(decisions), tuple(imps)


__all__ = [
    "CompressionResult",
    "Compressor",
    "Summarizer",
]
