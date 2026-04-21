"""Tier 3 chat memory — rolling window of recent raw turns per chat_id.

Phase 7 build-order step 1. In-memory only; the sqlite-backed variant
lands later when the bot is wired up. Keeps the API surface small so
the swap is a drop-in.

Contract (per `docs/PLAN_PHASE_7_TELEGRAM.md` §3.1-§3.2):

* Each `chat_id` has its own bounded queue of `Turn` records.
* When a turn lands as #(`TIER3_KEEP_TURNS` + 1), the oldest turn is
  evicted from the live window into a per-chat eviction queue. The
  compressor (step 5) drains that queue, summarizes the evicted
  turns, and writes them to tier 2.
* Tenant isolation — `chat_id`s never see each other's turns.
* `Turn` is frozen + `extra="forbid"`. Cannot be mutated after
  construction; cannot be subclassed by accident.
* No PII in events. This module emits no events directly; callers
  in the context builder log structural counts only.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TIER3_KEEP_TURNS = 12
"""Maximum live turns per chat_id. Default lifted from §3.2."""

Role = Literal["user", "bot"]


class Turn(BaseModel):
    """One exchange in a chat. Immutable once constructed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_id: str = Field(min_length=1)
    turn_idx: int = Field(ge=0)
    role: Role
    text: str
    ts: datetime


def _default_clock() -> datetime:
    return datetime.now(UTC)


class Tier3Store:
    """In-memory rolling-window store, keyed by `chat_id`.

    Not thread-safe — Phase 7 runs the Telegram dispatcher
    single-threaded per chat (one in-flight command per chat_id, see
    plan §4.2). When we move to a multi-chat worker pool, wrap this
    in a per-chat lock.
    """

    def __init__(
        self,
        *,
        keep_turns: int = TIER3_KEEP_TURNS,
        clock: Callable[[], datetime] = _default_clock,
    ) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be >= 1")
        self._keep = keep_turns
        self._clock = clock
        self._live: dict[str, list[Turn]] = {}
        self._evicted: dict[str, list[Turn]] = {}
        self._next_idx: dict[str, int] = {}

    def append(self, chat_id: str, role: Role, text: str) -> Turn:
        """Record a turn. Evicts the oldest live turn if window is full."""
        if not chat_id:
            raise ValueError("chat_id must be non-empty")
        idx = self._next_idx.get(chat_id, 0)
        turn = Turn(
            chat_id=chat_id,
            turn_idx=idx,
            role=role,
            text=text,
            ts=self._clock(),
        )
        live = self._live.setdefault(chat_id, [])
        live.append(turn)
        self._next_idx[chat_id] = idx + 1
        if len(live) > self._keep:
            self._evicted.setdefault(chat_id, []).append(live.pop(0))
        return turn

    def recent(self, chat_id: str) -> tuple[Turn, ...]:
        """Snapshot of the live window, oldest → newest."""
        return tuple(self._live.get(chat_id, ()))

    def drain_evicted(self, chat_id: str) -> tuple[Turn, ...]:
        """Return + clear all turns evicted since the last drain.

        The compressor calls this when the live window crosses the
        threshold. Drained turns are gone from this store — the
        compressor is responsible for promoting them to tier 2 (or
        archiving verbatim if the summarizer is down).
        """
        evicted = self._evicted.pop(chat_id, [])
        return tuple(evicted)

    def peek_evicted(self, chat_id: str) -> tuple[Turn, ...]:
        """Snapshot of pending-evicted turns without draining. Tests + observability."""
        return tuple(self._evicted.get(chat_id, ()))

    def clear(self, chat_id: str) -> None:
        """Wipe a single chat — for tests and tenant deletion."""
        self._live.pop(chat_id, None)
        self._evicted.pop(chat_id, None)
        self._next_idx.pop(chat_id, None)

    def __len__(self) -> int:
        """Total live turns across all chats."""
        return sum(len(turns) for turns in self._live.values())

    def chat_size(self, chat_id: str) -> int:
        """Live turn count for one chat."""
        return len(self._live.get(chat_id, ()))
