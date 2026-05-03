"""Phase 7 §6 step 13 — conversational turn pipeline.

Composes the pieces built across Track A + Track B into a single
async entrypoint the Telegram `route_chat` handler can call with a
raw user message and get a reply string back:

    Tier1Loader  ─┐
    Tier3Store   ─┤
    RecallPolicy ─┼─→ ChatPipeline.turn(chat_id, user_text) → reply
    ContextBuilder┤
    ModelClient  ─┘

Design pins (from `docs/PLAN_PHASE_7_TELEGRAM.md` §3.3, §3.6, §6):

* **Never raises.** LLM failures, tier-2 outages, embedder errors —
  all surface as a typed "degraded" reply. The pipeline always
  emits a `chat.turn.context` event (with `overflow`/`error` flags)
  and always appends the user+bot turns to tier 3 so the next turn
  has full history. This mirrors the stub-on-failure posture in
  `memory/recall.py` and the coding harness (§2.8).
* **No message bodies in events.** `chat.turn.context` carries only
  structural counts (bytes per tier, retrieval counts). `chat.turn.reply`
  carries the reply *byte count* and `error` flag, never the text.
* **Tier 3 write happens AFTER the LLM call succeeds OR fails.** Even
  a stub reply is recorded so the conversation state is
  self-consistent. If we didn't write a stub reply, the next turn's
  tier-3 history would have a dangling user message with no answer.
* **ModelClient is injectable.** No provider branching in this module.
  Production wiring lives in `bot.py` next to the SDK imports.

The intent classifier from plan §3.3 step 2 and §3.6 is a future
enhancement — this v1 runs the recall policy on every turn and lets
the context-builder's byte budget drop low-signal hits. That's a
correctness-preserving shortcut: the policy already score-ranks and
the builder already enforces the budget, so wasted recall cycles are
the only cost. Intent gating is a follow-up optimization.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from runtime.chat.memory.context_builder import ContextBuilder, Lookup, TurnContext
from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store, Turn
from runtime.chat.reply_verdict import annotate_unverified_claim
from runtime.events import EventStream, EventType
from runtime.llm.clients.base import ChatMessage, ChatRequest, ModelClient
from runtime.tools.record import load_tool_calls


class RecallPort(Protocol):
    """Anything that turns `(chat_id, user_text)` into ranked lookups.

    `runtime.chat.memory.recall.RecallPolicy` satisfies this structurally.
    A Protocol keeps `pipeline.py` testable without standing up a real
    `Tier2Store` (sqlite + embedder) in unit tests.
    """

    def recall(
        self, chat_id: str, user_text: str
    ) -> tuple[Lookup, ...]: ...

DEFAULT_SYSTEM_PROMPT = (
    "You are AEGIS, an operator-facing assistant. "
    "Prefer concise, direct answers. When unsure, ask a clarifying question. "
    "Never fabricate IMP or CT ids — if a referenced id is missing from "
    "context, say so plainly."
)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2
STUB_REPLY = (
    "AEGIS is temporarily unable to reach the model. "
    "Your message is still recorded; please try again in a moment."
)

logger = logging.getLogger(__name__)


def _role_to_chat(role: str) -> str:
    # Tier3 Role = "user" | "bot"; ModelClient expects "user" | "assistant".
    return "assistant" if role == "bot" else "user"


def _format_tier1(snap: Any) -> str:
    """Render `Tier1Snapshot` into a single text block for the system prompt.

    Skips empty sections so a fresh deployment (no IDENTITY/USER yet)
    doesn't waste budget on empty headers.
    """
    parts: list[str] = []
    if snap.identity:
        parts.append(f"## Identity\n{snap.identity}")
    if snap.user:
        parts.append(f"## Operator\n{snap.user}")
    prefs = getattr(snap, "prefs", None)
    if prefs:
        # prefs is a dict; render as a simple k=v list. Truthiness on dict
        # is False when empty so we skip the noise.
        rendered = "\n".join(f"{k}: {v}" for k, v in sorted(prefs.items()))
        parts.append(f"## Preferences\n{rendered}")
    return "\n\n".join(parts)


def _format_lookups(lookups: tuple[Lookup, ...]) -> str:
    if not lookups:
        return ""
    chunks = []
    for lk in lookups:
        chunks.append(f"### {lk.kind}:{lk.origin}\n{lk.text}")
    return "## Retrieved context\n" + "\n\n".join(chunks)


def _compose_system_prompt(
    *,
    system_prefix: str,
    tier1_text: str,
    lookups_text: str,
) -> str:
    parts = [system_prefix]
    if tier1_text:
        parts.append(tier1_text)
    if lookups_text:
        parts.append(lookups_text)
    return "\n\n".join(parts)


def _history_messages(turns: tuple[Turn, ...]) -> list[ChatMessage]:
    return [
        ChatMessage(role=_role_to_chat(t.role), content=t.text)  # type: ignore[arg-type]
        for t in turns
    ]


class ChatPipeline:
    """Async turn pipeline composing memory + recall + LLM.

    Constructed once per bot process (all state lives in the injected
    stores). `turn()` is coroutine-safe as long as the underlying
    stores are — the in-memory `Tier3Store` is single-writer per
    chat_id (plan §4.2 one-in-flight guard).
    """

    def __init__(
        self,
        *,
        tier1: Tier1Loader,
        tier3: Tier3Store,
        recall: RecallPort,
        builder: ContextBuilder,
        model: ModelClient,
        model_name: str,
        events: EventStream | None = None,
        system_prefix: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        self._tier1 = tier1
        self._tier3 = tier3
        self._recall = recall
        self._builder = builder
        self._model = model
        self._model_name = model_name
        self._events = events
        self._system_prefix = system_prefix
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def turn(self, chat_id: str, user_text: str) -> str:
        """Run one conversational turn. Never raises."""
        if not chat_id or not user_text.strip():
            return ""

        lookups = self._safe_recall(chat_id, user_text)
        ctx = self._builder.build(chat_id, lookups=lookups)

        system_text = _compose_system_prompt(
            system_prefix=self._system_prefix,
            tier1_text=_format_tier1(ctx.tier1),
            lookups_text=_format_lookups(ctx.lookups),
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_text),
            *_history_messages(ctx.tier3_turns),
            ChatMessage(role="user", content=user_text),
        ]

        reply, errored = await self._safe_chat(messages)

        final_reply = self._gate_reply(reply)

        # Record the turn in tier 3 regardless of success — the
        # stub reply is still a reply from the operator's POV. We
        # store the annotated reply so tier 3 history reflects what
        # the operator actually saw.
        self._tier3.append(chat_id, "user", user_text)
        self._tier3.append(chat_id, "bot", final_reply)

        self._emit_context(ctx, errored=errored)
        self._emit_reply(chat_id=chat_id, reply=final_reply, errored=errored)
        return final_reply

    # ---- internals --------------------------------------------------

    def _gate_reply(self, reply: str) -> str:
        """Annotate the reply if it claims an action the audit trail can't confirm.

        Builds the set of ``tool.invoked`` tool ids on the current
        session shard with ``verdict='verified'``. The gate uses this
        set to per-tool match against claim phrasings (§D2). When the
        shard isn't reachable (no events wired, IO error), we skip
        the gate rather than over-annotate — chat must never crash on
        a log read failure.
        """
        if self._events is None:
            return reply
        try:
            verified = {
                r.tool for r in load_tool_calls(self._events) if r.verdict == "verified"
            }
        except Exception:
            logger.exception("chat.pipeline.verdict_gate_read_failed")
            return reply
        outcome = annotate_unverified_claim(reply, verified_tools=verified)
        if outcome.was_flagged:
            logger.info(
                "chat.pipeline.unverified_tool_claim",
                extra={"phrases": list(outcome.flagged_phrases)},
            )
        return outcome.annotated_reply

    def _safe_recall(self, chat_id: str, user_text: str) -> tuple[Lookup, ...]:
        try:
            return self._recall.recall(chat_id, user_text)
        except Exception:
            logger.exception("chat.pipeline.recall_failed", extra={"chat_id": chat_id})
            return ()

    async def _safe_chat(self, messages: list[ChatMessage]) -> tuple[str, bool]:
        req = ChatRequest(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        t0 = time.monotonic()
        try:
            resp = await self._model.chat(req)
        except Exception:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.exception(
                "chat.pipeline.model_failed",
                extra={"model": self._model_name, "latency_ms": elapsed_ms},
            )
            return (STUB_REPLY, True)
        text = resp.content or ""
        return (text if text.strip() else STUB_REPLY, not bool(text.strip()))

    def _emit_context(self, ctx: TurnContext, *, errored: bool) -> None:
        if self._events is None:
            return
        payload = ctx.event_payload()
        payload["errored"] = errored
        try:
            self._events.append(EventType.CHAT_TURN_CONTEXT, payload)
        except Exception:
            logger.exception("chat.pipeline.emit_context_failed")

    def _emit_reply(self, *, chat_id: str, reply: str, errored: bool) -> None:
        if self._events is None:
            return
        try:
            self._events.append(
                EventType.CHAT_TURN_REPLY,
                {
                    "chat_id": chat_id,
                    "reply_bytes": len(reply.encode("utf-8")),
                    "errored": errored,
                    "model": self._model_name,
                },
            )
        except Exception:
            logger.exception("chat.pipeline.emit_reply_failed")


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "STUB_REPLY",
    "ChatPipeline",
    "RecallPort",
]
