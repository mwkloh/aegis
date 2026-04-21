"""Phase 7 §4.3 — slash-command routing.

Design constraints from `docs/PLAN_PHASE_7_TELEGRAM.md`:

* **No network in tests.** The dispatcher accepts a plain
  `IncomingMessage` dataclass — not a `telegram.Update`. The real
  entrypoint (`bot.py`, built last) converts one to the other.
* **Authorized first, parsed second.** Denied chats never reach a
  handler, so even a malformed `/approve` from an unauthorized
  operator produces a single audit-emitting deny reply — never a
  usage-hint that would leak the command surface.
* **One handler per slash, registered at construction.** Dispatch
  is a dict lookup; there is no regex fallback, no fuzzy match. An
  unknown slash returns a typed `unknown_command` reply — the
  operator surface is closed by default.
* **Never raises.** Handler exceptions are caught and converted
  into a typed `handler_error` reply, the same way the recall
  policy catches tier 2 exceptions. The caller decides what to
  log; dispatch only returns a `DispatchOutcome`.
"""
from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from runtime.chat.telegram.auth import Authorizer

OutcomeKind = Literal[
    "ok",
    "denied",
    "unknown_command",
    "not_a_command",
    "handler_error",
]


class IncomingMessage(BaseModel):
    """Minimal Telegram message shape the dispatcher actually needs.

    `bot.py` maps `telegram.Update` → this; tests construct it
    directly. Intentionally tiny — any field not listed here is
    out of scope for Phase 7.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chat_id: int
    user_id: int
    text: str = Field(min_length=0, max_length=4096)


class ParsedCommand(BaseModel):
    """A `/slash arg1 arg2` split into name + shell-style args."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    args: tuple[str, ...] = Field(default_factory=tuple)


class DispatchOutcome(BaseModel):
    """Everything the caller (bot.py) needs to send a reply + audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: OutcomeKind
    chat_id: int
    reply: str
    command: str | None = Field(default=None)
    error: str | None = Field(default=None)


Handler = Callable[[IncomingMessage, ParsedCommand], str]


class Dispatcher:
    """Slash-command router.

    Wire handlers at construction. The dispatcher never mutates its
    registry after `__init__` — safe to share across threads / async
    tasks.
    """

    def __init__(
        self,
        *,
        authorizer: Authorizer,
        handlers: dict[str, Handler],
    ) -> None:
        self._auth = authorizer
        # Copy + normalize to leading-slash form so callers can register
        # either `"pending"` or `"/pending"` without surprise.
        self._handlers: dict[str, Handler] = {
            _normalize(name): handler for name, handler in handlers.items()
        }

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    @property
    def authorizer(self) -> Authorizer:
        """Expose the authorizer so async callers (bot.py long-running
        surface) can auth-check without re-plumbing a separate param."""
        return self._auth

    def handle(self, message: IncomingMessage) -> DispatchOutcome:
        decision = self._auth.check(message.chat_id)
        if not decision:
            return DispatchOutcome(
                kind="denied",
                chat_id=message.chat_id,
                reply="unauthorized",
                error=decision.reason,
            )
        parsed = parse_command(message.text)
        if parsed is None:
            return DispatchOutcome(
                kind="not_a_command",
                chat_id=message.chat_id,
                reply=(
                    "Phase 7 accepts slash commands only. "
                    "Try /pending, /status, or /decisions."
                ),
            )
        handler = self._handlers.get(parsed.name)
        if handler is None:
            return DispatchOutcome(
                kind="unknown_command",
                chat_id=message.chat_id,
                reply=f"unknown command: {parsed.name}",
                command=parsed.name,
            )
        try:
            reply = handler(message, parsed)
        except Exception as exc:  # stub-on-failure (§2.8)
            return DispatchOutcome(
                kind="handler_error",
                chat_id=message.chat_id,
                reply=f"internal error running {parsed.name}",
                command=parsed.name,
                error=str(exc),
            )
        return DispatchOutcome(
            kind="ok",
            chat_id=message.chat_id,
            reply=reply,
            command=parsed.name,
        )


def parse_command(text: str) -> ParsedCommand | None:
    """Parse a slash command. Returns `None` for non-commands.

    Shell-style argument splitting so operators can quote rationales:
    `/approve IMP-abc12345 "ship it — tests green"`. Unclosed quotes
    fall back to a simple whitespace split so a malformed line still
    reaches the handler instead of raising.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    # Drop leading slash, split head from body
    head, _, body = stripped[1:].partition(" ")
    if not head:
        return None
    name = _normalize(head.split("@", maxsplit=1)[0])
    try:
        args = tuple(shlex.split(body)) if body else ()
    except ValueError:
        args = tuple(body.split())
    return ParsedCommand(name=name, args=args)


def _normalize(name: str) -> str:
    n = name.strip().lower()
    if not n.startswith("/"):
        n = "/" + n
    return n


__all__ = [
    "DispatchOutcome",
    "Dispatcher",
    "Handler",
    "IncomingMessage",
    "OutcomeKind",
    "ParsedCommand",
    "parse_command",
]
