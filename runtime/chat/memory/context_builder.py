"""Per-turn context builder — merges tier 1, tier 3, and supplied tier-2
lookups under a byte budget.

Phase 7 build-order step 4. Pure assembly — intent classification and
tier-2 lookup live outside this module. Callers pass a ranked list of
`Lookup` objects; the builder loads tier 1 + tier 3 and enforces
`TELEGRAM_TURN_TOKEN_BUDGET` (default 8 KB) per plan §3.3.

Budget rule:

1. Tier 1 bytes always included (`Tier1Loader.load`).
2. Tier 3 bytes always included (`Tier3Store.recent`).
3. Lookups are iterated in the order given (caller sorts by
   relevance desc). Dropped *from the tail* until total fits.
4. If tier 1 + tier 3 alone exceed budget, `overflow=True`. The
   turn still proceeds with what we have — the bot never silently
   stalls (stub-on-failure posture from §2.8).

`TurnContext` carries byte counts for the `chat.turn.context`
telemetry event. `event_payload()` returns the no-bodies dict
the caller wires into the event bus — structural counts only,
per §3.3 step 5.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .tier1 import Tier1Loader, Tier1Snapshot
from .tier3 import Tier3Store, Turn

DEFAULT_TURN_BUDGET_BYTES = 8 * 1024
"""8 KB default, matching `TELEGRAM_TURN_TOKEN_BUDGET` in plan §2.3."""

LookupKind = Literal["episodic", "vault"]


class Lookup(BaseModel):
    """One materialized tier-2 lookup ready to splice into the prompt.

    `origin` is a short opaque tag for the telemetry event — never a
    message body — so logs can join back to the source record.
    Shape maps 1:1 onto a tier-2 hit; the caller adapts
    `EpisodicHit` / `VaultHit` into `Lookup` before passing in.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: LookupKind
    text: str
    score: float = Field(ge=0.0)
    origin: str = Field(min_length=1)

    @property
    def nbytes(self) -> int:
        return len(self.text.encode("utf-8"))


class TurnContext(BaseModel):
    """Assembled per-turn view, ready for prompt templating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_id: str = Field(min_length=1)
    tier1: Tier1Snapshot
    tier3_turns: tuple[Turn, ...]
    lookups: tuple[Lookup, ...]
    bytes_tier1: int = Field(ge=0)
    bytes_tier3: int = Field(ge=0)
    bytes_lookups: int = Field(ge=0)
    budget_bytes: int = Field(gt=0)
    lookups_considered: int = Field(ge=0)
    lookups_kept: int = Field(ge=0)
    overflow: bool

    @property
    def total_bytes(self) -> int:
        return self.bytes_tier1 + self.bytes_tier3 + self.bytes_lookups

    def event_payload(self) -> dict[str, Any]:
        """`chat.turn.context` event body — structural counts only.

        Never includes message bodies. Safe to persist or forward
        into Plane 1 event storage (see §2.7 / §3.3 step 5).
        """
        return {
            "chat_id": self.chat_id,
            "budget_bytes": self.budget_bytes,
            "total_bytes": self.total_bytes,
            "bytes_tier1": self.bytes_tier1,
            "bytes_tier3": self.bytes_tier3,
            "bytes_lookups": self.bytes_lookups,
            "tier3_turns": len(self.tier3_turns),
            "lookups_considered": self.lookups_considered,
            "lookups_kept": self.lookups_kept,
            "overflow": self.overflow,
        }


class ContextBuilder:
    """Assembles per-turn context under a byte budget."""

    def __init__(
        self,
        tier1: Tier1Loader,
        tier3: Tier3Store,
        *,
        budget_bytes: int = DEFAULT_TURN_BUDGET_BYTES,
    ) -> None:
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be > 0")
        self._tier1 = tier1
        self._tier3 = tier3
        self._budget = budget_bytes

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def build(
        self,
        chat_id: str,
        *,
        lookups: Iterable[Lookup] = (),
    ) -> TurnContext:
        """Build the turn context. Lookups must be ranked best → worst."""
        snap = self._tier1.load(chat_id)
        turns = self._tier3.recent(chat_id)
        bytes_tier3 = sum(len(t.text.encode("utf-8")) for t in turns)

        ranked = tuple(lookups)
        bytes_fixed = snap.total_bytes + bytes_tier3
        remaining = self._budget - bytes_fixed

        kept: list[Lookup] = []
        used = 0
        if remaining > 0:
            for r in ranked:
                if used + r.nbytes > remaining:
                    break
                kept.append(r)
                used += r.nbytes

        total = bytes_fixed + used
        overflow = total > self._budget

        return TurnContext(
            chat_id=chat_id,
            tier1=snap,
            tier3_turns=turns,
            lookups=tuple(kept),
            bytes_tier1=snap.total_bytes,
            bytes_tier3=bytes_tier3,
            bytes_lookups=used,
            budget_bytes=self._budget,
            lookups_considered=len(ranked),
            lookups_kept=len(kept),
            overflow=overflow,
        )


__all__ = [
    "DEFAULT_TURN_BUDGET_BYTES",
    "ContextBuilder",
    "Lookup",
    "LookupKind",
    "TurnContext",
]
