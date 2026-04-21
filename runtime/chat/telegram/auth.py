"""Phase 7 §4.3 — `chat_id` allow-list.

Non-negotiables (`docs/PLAN_PHASE_7_TELEGRAM.md` §2):

* One operator per deployment in Phase 7. Any `chat_id` outside the
  configured allow-list is rejected with a typed decision — the
  dispatcher never looks at the message body for a denied chat.
* Empty allow-list = **deny all**, not "allow all". This is the
  opposite of the usual default so a misconfigured bot cannot
  accidentally accept the world on its first boot.
* The authorizer is a pure function over `(chat_id, allowlist)` —
  no global state, no config reads, no event writes. Callers wire
  it once at startup with the allowlist pulled from `AegisConfig`.

Denied requests never reach dispatch; callers MUST emit the
`governance.denied` audit event themselves so the Reflection plane
sees the attempt without the dispatcher having to know about events.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DenyReason = Literal["not_allowed", "empty_allowlist"]


class AuthDecision(BaseModel):
    """Authorization verdict for one `chat_id`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool
    chat_id: int
    reason: DenyReason | None = Field(default=None)

    def __bool__(self) -> bool:  # truthy iff allowed
        return self.allowed


class Authorizer:
    """Immutable allow-list check over Telegram `chat_id` values."""

    def __init__(self, allowlist: tuple[int, ...]) -> None:
        self._allowlist = tuple(dict.fromkeys(allowlist))

    @property
    def allowlist(self) -> tuple[int, ...]:
        return self._allowlist

    def check(self, chat_id: int) -> AuthDecision:
        if not self._allowlist:
            return AuthDecision(
                allowed=False, chat_id=chat_id, reason="empty_allowlist"
            )
        if chat_id in self._allowlist:
            return AuthDecision(allowed=True, chat_id=chat_id, reason=None)
        return AuthDecision(
            allowed=False, chat_id=chat_id, reason="not_allowed"
        )


__all__ = [
    "AuthDecision",
    "Authorizer",
    "DenyReason",
]
