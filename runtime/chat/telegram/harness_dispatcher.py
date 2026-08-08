"""HarnessDispatcher — pre-pipeline tool-use layer for Telegram free-form chat."""
from __future__ import annotations

import enum
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.reply_verdict import annotate_unverified_claim
from runtime.events import EventStream, EventType
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.llm.clients.base import ChatMessage, ChatRequest, ModelClient
from runtime.reasoning.skill_runner import SkillRunner
from runtime.skills.registry import SkillDescriptor, SkillRegistry
from runtime.tools.record import (
    compute_argv_hash,
    load_tool_calls,
    record_tool_call,
    verdict_for_result,
)

logger = logging.getLogger(__name__)

HARNESS_CONFIDENCE_THRESHOLD = 0.7
_MAX_REPLY_CHARS = 3500

# Tools that mutate filesystem state. The multi-step loop intercepts any
# planner step proposing one of these BEYOND the first tool call and routes
# the operator through a deterministic confirmation path (no LLM rephrasing).
# A destructive intent at step 1 is allowed — it's the operator's explicit
# opening request. Defence-in-depth: these names are not in DEFAULT_TOOLS
# today, but a hallucinating planner could still propose them.
# See docs/PLAN_MULTI_STEP_AGENT_LOOP.md §5 decision 5.
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "files_delete",
    "files_move",
    "files_write",
})


@dataclass(frozen=True)
class _PendingConfirmation:
    """A destructive intent the guard intercepted, held for one follow-up turn.

    `user_text` is the ORIGINAL request that produced `intent` (not the
    later "yes") — `_execute_confirmed` needs it to feed `_synthesize`'s
    "the operator asked" prompt slot and to mirror the single-shot path's
    tier3 write. In-memory only, keyed by chat_id on the dispatcher
    instance — single-operator scope, lost on restart by design.
    """

    intent: ToolIntent
    skill_id: str
    user_text: str
    created_at: datetime


_CONFIRMATION_TTL_S = 120
_AFFIRMATIVES = frozenset({"yes", "y", "confirm", "do it", "go ahead"})

_SYNTHESIS_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "reasoning"
    / "prompts"
    / "tool_synthesis.txt"
)
_SYNTHESIS_CHAIN_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "reasoning"
    / "prompts"
    / "tool_synthesis_chain.txt"
)
_MAX_CHAIN_RESULT_CHARS = 1024

__all__ = ["DESTRUCTIVE_TOOLS", "DispatchOutcome", "HarnessDispatcher"]


class DispatchOutcome(enum.Enum):
    FIRED = "fired"
    CLARIFY = "clarify"
    PASS = "pass"


@dataclass(frozen=True)
class _ChainResult:
    """Outcome of `_run_multi_step`. `guarded_intent` is set when the
    destructive guard intercepted a planner step beyond step 1; `history`
    holds whatever ran successfully BEFORE the interception.
    `completion_summary` is set when the planner emitted `task_complete`
    with non-empty history — it carries the gate-checked summary text
    (Track C, C2), and its presence tells the caller to skip chain
    synthesis and reply with this text directly."""

    history: list[tuple[ToolIntent, ToolResult]] = field(default_factory=list)
    guarded_intent: ToolIntent | None = None
    completion_summary: str | None = None


def _format_destructive_confirmation(intent: ToolIntent) -> str:
    """Deterministic operator-facing message for the destructive-guard path.

    Routed directly through `_send` — never through the synthesizer. The
    point of the guard is to bypass the model on destructive paths so the
    operator sees the exact tool id and args, not an LLM paraphrase.
    """
    return (
        f"⚠️ I'd like to run `{intent.tool}` with args {intent.args!r} to "
        f"finish this — please confirm with a follow-up message before I proceed."
    )


def _clarify_question(descriptor: SkillDescriptor) -> str:
    if descriptor.id == "list_files":
        return "Which folder should I list? (e.g. ~/Downloads)"
    if descriptor.id == "read_file":
        return "Which file should I read? Please give the full path."
    if descriptor.id == "search_files":
        return "Which folder and pattern should I search? (e.g. ~/Downloads *.pdf)"
    if descriptor.id == "file_info":
        return "Which file or directory should I get info for? Please give the full path."
    schema = descriptor.args_schema
    required: list[str] = schema.get("required", []) if isinstance(schema, dict) else []
    if required:
        fields = " and ".join(required)
        return f"Could you clarify the {fields} for: {descriptor.description}"
    return f"Could you provide more details for: {descriptor.description}"


def _render_chain(history: list[tuple[ToolIntent, ToolResult]]) -> str:
    """Render the (call, result) chain into the chain-synthesis prompt slot.

    Each result payload is clipped to bound prompt length; even a long chain
    of fat results stays within `max_tokens`.
    """
    if not history:
        return "(no tools ran on this turn)"
    lines: list[str] = []
    for idx, (call, result) in enumerate(history, start=1):
        args_blob = _format_args(call.args)
        if result.status == "error":
            payload = f"ERROR: {result.error or '(no detail)'}"
        else:
            payload = str(result.payload) if result.payload is not None else "(empty)"
        if len(payload) > _MAX_CHAIN_RESULT_CHARS:
            payload = payload[:_MAX_CHAIN_RESULT_CHARS] + "…(truncated)"
        lines.append(f"{idx}. {call.tool}({args_blob}) → {payload}")
    return "\n".join(lines)


def _format_args(args: dict[str, Any]) -> str:
    """Compact args repr — sorted keys for stable prompts."""
    if not args:
        return ""
    parts = [f"{k}={args[k]!r}" for k in sorted(args)]
    return ", ".join(parts)


def _last_payload_text(history: list[tuple[ToolIntent, ToolResult]]) -> str:
    """Synthesis-failure fallback text — most-recent state the operator wants."""
    if not history:
        return "(no tools ran)"
    _, last = history[-1]
    if last.status == "error" and last.error:
        return last.error
    if last.payload is not None:
        return str(last.payload)
    return "(empty)"


def _history_verified_tools(history: list[tuple[ToolIntent, ToolResult]]) -> set[str]:
    """Tools that actually succeeded on this turn, from in-memory history.

    Shared by `_gate_completion` (both its no-ledger and ledger-read-failed
    branches) and `dispatch()`'s respond-path verdict gate — a failed tool
    must never count as "verified" regardless of which of those call sites
    is asking (Phase 11 whole-branch review, C3/I3).

    Uses `verdict_for_result` rather than a bare `res.status == "ok"` check
    (Phase 11 review follow-up) — some tools report a soft failure inside
    an "ok" status (e.g. `run_command`'s `payload["verdict"] ==
    "exit_nonzero"` for a non-zero exit). A raw status check would still
    count that tool as verified here even though C4 already teaches the
    ledger path not to.
    """
    return {
        call.tool for call, res in history if verdict_for_result(res) == "verified"
    }


def _clip(text: str) -> str:
    """Bound raw tool output used as a synthesis-failure fallback.

    Successful synthesis replies are NOT clipped — `_chunk()` in
    `bot.py` splits long replies into multiple Telegram messages at
    the 4096-char per-message limit. This function only guards the
    fallback path where we ship `str(tool_result.payload)` directly.
    """
    return text[:_MAX_REPLY_CHARS]


class HarnessDispatcher:
    def __init__(
        self,
        *,
        classifier: Any,
        registry: SkillRegistry,
        runner: SkillRunner,
        harness: HarnessAdapter,
        synthesizer: ModelClient,
        tier3: Tier3Store,
        tier1_loader: Tier1Loader,
        synthesis_model: str,
        multi_step: bool = False,
        max_steps: int = 5,
        events: EventStream | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._classifier = classifier
        self._registry = registry
        self._runner = runner
        self._harness = harness
        self._synthesizer = synthesizer
        self._tier3 = tier3
        self._tier1_loader = tier1_loader
        self._synthesis_model = synthesis_model
        # multi_step is the Step-1 scaffold for the multi-step agent loop. The
        # Step-2 loop body, the verdict-gate refactor (set-based), and the
        # destructive-tool guard all hang off this flag — keep it gated until
        # those land. See docs/PLAN_MULTI_STEP_AGENT_LOOP.md.
        self._multi_step = multi_step
        self._max_steps = max_steps
        # Evidence ledger (Phase 11 Track B). `None` keeps every existing
        # call site/test working unmodified — recording is opt-in.
        self._events = events
        self._clock = clock
        # Destructive-guard confirmation state (Phase 11 Track D, D1). In-memory
        # only, keyed by chat_id: single-operator scope, lost on restart by
        # design. Popped-before-checked in `dispatch()` so a pending intent is
        # consumed exactly once regardless of outcome.
        self._pending: dict[int, _PendingConfirmation] = {}

    def _now(self) -> datetime:
        return self._clock() if self._clock is not None else datetime.now(tz=UTC)

    def _append_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Log one structural event, if an EventStream is wired.

        `EventStream.append` does unguarded file I/O and can raise (disk
        full, permissions, ...). Some call sites fire AFTER a destructive
        tool has already mutated disk (e.g. HARNESS_CONFIRMATION_ACCEPTED
        in `_execute_confirmed`) — an unguarded raise there would propagate
        out of `dispatch()` and leave the operator with no reply about an
        action that DID happen. Telemetry must never be able to break a
        chat turn, so this is wrapped total, like `_record_tool_call`.
        """
        if self._events is None:
            return
        try:
            self._events.append(event_type, payload)
        except Exception:
            logger.warning("harness_dispatcher.event_append_failed", exc_info=True)

    def _record_tool_call(
        self,
        *,
        turn_id: str,
        skill_id: str,
        tool_intent: ToolIntent,
        result: ToolResult,
    ) -> None:
        """Log one executed tool call to the evidence ledger, if wired.

        Structural only — argv_hash and outcome_bytes, never args contents
        or payload bodies. `record_tool_call` is idempotent on its
        composite key and does not raise on malformed prior shard lines.
        Recording is telemetry, not the product: `args`/`payload` are
        `dict[str, Any]` and may hold values `json.dumps` can't natively
        serialize (sets, `Path`, `datetime`, ...), so serialization here
        uses `default=str` to stay total, and the whole body is wrapped so
        no future recording failure can ever break a chat turn.
        """
        if self._events is None:
            return
        try:
            payload_json = json.dumps(
                result.payload, ensure_ascii=False, default=str
            )
            record_tool_call(
                self._events,
                imp_id=turn_id,
                skill=skill_id,
                tool=tool_intent.tool,
                argv_hash=compute_argv_hash(
                    [
                        tool_intent.tool,
                        json.dumps(tool_intent.args, sort_keys=True, default=str),
                    ]
                ),
                verdict=verdict_for_result(result),
                outcome_bytes=len(payload_json.encode("utf-8")),
            )
        except Exception:
            logger.warning("harness_dispatcher.ledger_record_failed", exc_info=True)

    def _gate_completion(
        self,
        summary: str,
        history: list[tuple[ToolIntent, ToolResult]],
        turn_id: str,
    ) -> str:
        """Check the claimed summary against this turn's evidence.

        Evidence = ledger records for turn_id with verdict == "verified"
        (falls back to in-memory history when no EventStream is wired, OR
        when the ledger read itself raises — a disk error on a
        task_complete turn must never escape `dispatch()`; there is no PTB
        error handler catching it. See `chat/pipeline.py::_gate_reply` for
        the sibling pattern this mirrors; Phase 11 whole-branch review C3).
        A summary claiming tools that have no verified record gets the
        existing unverified-claim annotation; it is never silently trusted.
        """
        if self._events is not None:
            try:
                records = [
                    r
                    for r in load_tool_calls(self._events)
                    if r.imp_id == turn_id and r.verdict == "verified"
                ]
                verified = {r.tool for r in records}
            except Exception:
                logger.exception("harness_dispatcher.gate_completion_read_failed")
                verified = _history_verified_tools(history)
        else:
            verified = _history_verified_tools(history)
        # A tool that failed on an early step and then succeeded on a later
        # retry (first-class in the multi-step loop — see `_run_multi_step`'s
        # docstring) is NOT a failure: its later verification supersedes the
        # earlier error. Only tools whose failure was never superseded by a
        # verified call are reported.
        failed = sorted(
            {
                call.tool
                for call, res in history
                if verdict_for_result(res) != "verified"
            }
            - verified
        )
        if failed:
            summary = (
                f"{summary}\n\n⚠️ Note: {', '.join(failed)} did not "
                f"complete successfully — the summary above may overstate what "
                f"was done."
            )
            self._append_event(
                EventType.HARNESS_COMPLETION_GATED,
                {"turn_id": turn_id, "failed_tools": failed},
            )
        verdict = annotate_unverified_claim(summary, verified_tools=verified)
        if verdict.was_flagged:
            logger.info(
                "harness_dispatcher.completion_unverified_tool_claim",
                extra={"phrases": list(verdict.flagged_phrases)},
            )
        return verdict.annotated_reply

    async def _try_confirmed_dispatch(
        self,
        *,
        chat_id: int,
        user_text: str,
        message: Any,
        reply: Callable[[str], Awaitable[None]] | None,
    ) -> DispatchOutcome | None:
        """Pop and resolve any pending destructive-guard confirmation.

        Pop-before-check: whatever happens next (accepted, declined, or
        expired), a pending intent is consumed exactly once. No LLM call in
        this path — a 2B model must not judge its own destructive action.

        Returns the outcome of running the confirmed intent, or `None` if
        there was nothing pending, or it was declined/expired — `None`
        tells `dispatch()` to fall through to normal handling of `user_text`
        (which may be an unrelated request, not a confirmation at all).
        """
        pending = self._pending.pop(chat_id, None)
        if pending is None:
            return None
        age = (self._now() - pending.created_at).total_seconds()
        if age <= _CONFIRMATION_TTL_S and user_text.strip().casefold() in _AFFIRMATIVES:
            return await self._execute_confirmed(
                pending, chat_id=chat_id, message=message, reply=reply
            )
        self._append_event(
            EventType.HARNESS_CONFIRMATION_DECLINED,
            {"tool": pending.intent.tool, "expired": age > _CONFIRMATION_TTL_S},
        )
        return None

    async def dispatch(
        self,
        *,
        chat_id: int,
        user_text: str,
        message: Any,
        reply: Callable[[str], Awaitable[None]] | None = None,
    ) -> DispatchOutcome:
        # `reply`, when supplied, lets the caller wrap send with its typing
        # indicator / placeholder teardown. Absent (tests, CLI), we fall back
        # to posting directly on the Telegram message object.
        async def _send(text: str) -> None:
            if reply is not None:
                await reply(text)
            else:
                await message.reply_text(text)

        logger.info("harness_dispatcher.dispatch_start", extra={"chat_id": chat_id})

        # Destructive-guard confirmation check — BEFORE classification. A
        # non-None result means a pending intent from a prior turn was
        # accepted and executed; None means there was nothing pending, or it
        # was declined/expired — either way, fall through to normal dispatch
        # below (the message may be an unrelated request).
        confirmed_outcome = await self._try_confirmed_dispatch(
            chat_id=chat_id, user_text=user_text, message=message, reply=reply
        )
        if confirmed_outcome is not None:
            return confirmed_outcome

        # Minted once per dispatch call — never stored as instance state, since
        # dispatches for different chats/turns can interleave. Scopes every
        # tool-invocation record from this turn to one composite-key prefix
        # so a later completion gate can query "this turn's evidence" alone.
        turn_id = f"turn-{chat_id}-{uuid.uuid4().hex[:8]}"
        try:
            classification = await self._classifier.classify(user_text)
        except Exception:
            logger.exception("harness_dispatcher.classify_failed")
            return DispatchOutcome.PASS

        intent = classification.intent
        confidence = classification.confidence
        logger.info(
            "harness_dispatcher.classified",
            extra={"intent": intent, "confidence": confidence},
        )

        descriptor = self._registry.for_intent(intent)
        if descriptor is None:
            logger.info("harness_dispatcher.no_descriptor", extra={"intent": intent})
            return DispatchOutcome.PASS

        if not self._harness.has_tool(descriptor.tool):
            logger.info(
                "harness_dispatcher.no_tool", extra={"tool": descriptor.tool}
            )
            return DispatchOutcome.PASS

        if confidence < HARNESS_CONFIDENCE_THRESHOLD:
            question = _clarify_question(descriptor)
            await _send(question)
            self._tier3.append(str(chat_id), "user", user_text)
            self._tier3.append(str(chat_id), "bot", question)
            return DispatchOutcome.CLARIFY

        logger.info(
            "harness_dispatcher.recent_turns_start", extra={"chat_id": chat_id}
        )
        recent = self._recent_turns(chat_id)
        if self._multi_step:
            chain = await self._run_multi_step(
                descriptor=descriptor,
                user_text=user_text,
                recent=recent,
                turn_id=turn_id,
            )
            if chain.guarded_intent is not None:
                # Destructive guard tripped beyond step 1. Skip synthesis and
                # ship a deterministic confirmation prompt — the operator must
                # see the exact tool id and args, not an LLM paraphrase. Hold
                # the intent so an explicit "yes" follow-up (within TTL) can
                # run it without re-planning (Phase 11 Track D, D1).
                #
                # Arm the pending confirmation ONLY after `_send` succeeds. If
                # the send raises (e.g. a Telegram API error), the operator
                # never saw the prompt — storing the pending intent anyway
                # would let a later UNRELATED message that happens to match
                # an affirmative ("yes", "go ahead") silently execute a
                # destructive tool the operator was never shown.
                reply_text = _format_destructive_confirmation(chain.guarded_intent)
                await _send(reply_text)
                self._pending[chat_id] = _PendingConfirmation(
                    intent=chain.guarded_intent,
                    skill_id=descriptor.id,
                    user_text=user_text,
                    created_at=self._now(),
                )
                self._append_event(
                    EventType.HARNESS_CONFIRMATION_REQUESTED,
                    {"tool": chain.guarded_intent.tool, "skill_id": descriptor.id},
                )
                self._tier3.append(str(chat_id), "user", user_text)
                self._tier3.append(str(chat_id), "bot", reply_text)
                logger.info("harness_dispatcher.dispatch_complete_guarded")
                return DispatchOutcome.FIRED
            if chain.completion_summary is not None:
                # Planner claimed task_complete — the gate already checked the
                # summary against this turn's evidence (ledger or history) and
                # annotated it if needed. Use it directly as the reply; do NOT
                # run chain synthesis, which would be a second model call that
                # could re-hallucinate over the same (or gated) evidence.
                reply_text = chain.completion_summary
                logger.info("harness_dispatcher.completion_gate_reply")
            else:
                history = chain.history
                if not history:
                    # Planner immediately said "respond" — nothing ran, no synthesis,
                    # no tier3 write. Mirrors the legacy `tool == 'respond'` PASS.
                    return DispatchOutcome.PASS
                logger.info("harness_dispatcher.chain_synthesize_start")
                reply_text = await self._synthesize_chain(
                    user_text, history, chat_id=chat_id
                )
                logger.info(
                    "harness_dispatcher.chain_synthesize_done",
                    extra={"reply_chars": len(reply_text), "chain_steps": len(history)},
                )
                verdict = annotate_unverified_claim(
                    reply_text,
                    verified_tools=_history_verified_tools(history),
                )
                if verdict.was_flagged:
                    logger.info(
                        "harness_dispatcher.chain_unverified_tool_claim",
                        extra={"phrases": list(verdict.flagged_phrases)},
                    )
                reply_text = verdict.annotated_reply
        else:
            logger.info(
                "harness_dispatcher.runner_build_start",
                extra={"skill_id": descriptor.id, "recent_turns": len(recent)},
            )
            tool_intent = await self._runner.build(descriptor, user_text, recent=recent)
            logger.info(
                "harness_dispatcher.runner_build_done tool=%s args=%r",
                tool_intent.tool,
                tool_intent.args,
            )
            if tool_intent.tool == "respond":
                return DispatchOutcome.PASS

            logger.info(
                "harness_dispatcher.harness_execute_start tool=%s args=%r",
                tool_intent.tool,
                tool_intent.args,
            )
            result = self._harness.execute(tool_intent)
            logger.info(
                "harness_dispatcher.harness_execute_done", extra={"status": result.status}
            )
            self._record_tool_call(
                turn_id=turn_id,
                skill_id=descriptor.id,
                tool_intent=tool_intent,
                result=result,
            )
            logger.info("harness_dispatcher.synthesize_start")
            reply_text = await self._synthesize(user_text, tool_intent, result, chat_id=chat_id)
            logger.info(
                "harness_dispatcher.synthesize_done", extra={"reply_chars": len(reply_text)}
            )

        logger.info("harness_dispatcher.send_start")
        await _send(reply_text)
        logger.info("harness_dispatcher.send_done")
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
        logger.info("harness_dispatcher.dispatch_complete")
        return DispatchOutcome.FIRED

    async def _execute_confirmed(
        self,
        pending: _PendingConfirmation,
        *,
        chat_id: int,
        message: Any,
        reply: Callable[[str], Awaitable[None]] | None = None,
    ) -> DispatchOutcome:
        """Run a destructive intent the operator just confirmed with "yes".

        Mirrors the single-shot execute→record→synthesize→send→tier3 path
        in `dispatch()`, but the intent and its originating `user_text` come
        from `pending` instead of a fresh classify/build round — the whole
        point of holding the confirmation is to skip re-planning a 2B model
        already committed to via the guarded intent.
        """

        async def _send(text: str) -> None:
            if reply is not None:
                await reply(text)
            else:
                await message.reply_text(text)

        turn_id = f"turn-{chat_id}-{uuid.uuid4().hex[:8]}"
        logger.info(
            "harness_dispatcher.confirmed_execute_start",
            extra={"tool": pending.intent.tool, "skill_id": pending.skill_id},
        )
        result = self._harness.execute(pending.intent)
        logger.info(
            "harness_dispatcher.confirmed_execute_done",
            extra={"status": result.status},
        )
        self._record_tool_call(
            turn_id=turn_id,
            skill_id=pending.skill_id,
            tool_intent=pending.intent,
            result=result,
        )
        self._append_event(
            EventType.HARNESS_CONFIRMATION_ACCEPTED,
            {"tool": pending.intent.tool, "skill_id": pending.skill_id},
        )
        reply_text = await self._synthesize(
            pending.user_text, pending.intent, result, chat_id=chat_id
        )
        await _send(reply_text)
        self._tier3.append(str(chat_id), "user", pending.user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
        logger.info("harness_dispatcher.confirmed_dispatch_complete")
        return DispatchOutcome.FIRED

    async def _run_multi_step(
        self,
        *,
        descriptor: SkillDescriptor,
        user_text: str,
        recent: tuple[tuple[str, str], ...],
        turn_id: str,
    ) -> _ChainResult:
        """Bounded multi-step planner loop (Steps 2 + 4).

        Calls `runner.plan_next` up to `self._max_steps` times, executing
        each `tool_call` and threading the resulting history into the next
        plan call. Terminations: `kind == "respond"`, `kind == "task_complete"`,
        step cap reached, an empty available-skills set, or the destructive
        guard tripping. Tool errors are appended to history like any other
        result — the planner sees them on the next call and can decide to
        recover or respond.

        Destructive guard: a `tool_call` for a tool in `DESTRUCTIVE_TOOLS`
        is allowed at step 1 (the operator's explicit opening request) but
        intercepted at step 2+. The intercepted intent is returned via
        `_ChainResult.guarded_intent`; the caller is responsible for
        shipping a deterministic confirmation prompt.

        Completion gate (Track C, C2): a `task_complete` step with non-empty
        history is checked against this turn's evidence via
        `_gate_completion` and returned as `_ChainResult.completion_summary`
        — the caller replies with it directly, skipping chain synthesis. A
        `task_complete` with EMPTY history (nothing ran yet) breaks the loop
        with no summary, same as `respond` with no history, so the caller's
        PASS-on-empty-history path handles it unchanged.

        See docs/PLAN_MULTI_STEP_AGENT_LOOP.md and
        docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md Track C.
        """
        available = list(self._registry.all())
        history: list[tuple[ToolIntent, ToolResult]] = []
        logger.info(
            "harness_dispatcher.multi_step_start",
            extra={
                "skill_id": descriptor.id,
                "available_skills": len(available),
                "max_steps": self._max_steps,
            },
        )

        for step_no in range(1, self._max_steps + 1):
            logger.info(
                "harness_dispatcher.plan_next_start",
                extra={"step": step_no, "history_len": len(history)},
            )
            plan = await self._runner.plan_next(
                user_text=user_text,
                available_skills=available,
                history=tuple(history),
                recent=recent,
            )
            logger.info(
                "harness_dispatcher.plan_next_done step=%d kind=%s tool=%s",
                step_no,
                plan.kind,
                plan.tool,
            )
            if plan.kind == "task_complete":
                # A 2B planner can emit task_complete with no summary (or an
                # all-whitespace one) after a successful tool call — an empty
                # reply text would crash the send path (python-telegram-bot
                # rejects empty message text). Default to a minimal claim so
                # the gate always has non-empty text to check/annotate.
                summary = (plan.summary or "").strip() or "Done."
                if not history:
                    # Nothing was done this turn — same as respond-with-no-
                    # history: the caller falls through to PASS.
                    break
                gated = self._gate_completion(summary, history, turn_id)
                return _ChainResult(history=history, completion_summary=gated)

            if plan.kind != "tool_call" or plan.tool is None:
                break

            tool_intent = ToolIntent(
                tool=plan.tool,
                args=dict(plan.args or {}),
                skill_id=descriptor.id,
                rationale=f"multi-step planner: step {step_no}",
            )

            if plan.tool in DESTRUCTIVE_TOOLS and step_no >= 2:
                logger.info(
                    "harness_dispatcher.destructive_guard_triggered",
                    extra={"tool": plan.tool, "step": step_no},
                )
                return _ChainResult(history=history, guarded_intent=tool_intent)

            logger.info(
                "harness_dispatcher.harness_execute_start step=%d tool=%s args=%r",
                step_no,
                tool_intent.tool,
                tool_intent.args,
            )
            result = self._harness.execute(tool_intent)
            logger.info(
                "harness_dispatcher.harness_execute_done",
                extra={"step": step_no, "status": result.status},
            )
            self._record_tool_call(
                turn_id=turn_id,
                skill_id=descriptor.id,
                tool_intent=tool_intent,
                result=result,
            )
            history.append((tool_intent, result))

        logger.info(
            "harness_dispatcher.multi_step_done",
            extra={"chain_steps": len(history)},
        )
        return _ChainResult(history=history)

    def _recent_turns(self, chat_id: int) -> tuple[tuple[str, str], ...]:
        """Pull the rolling window of (role, text) pairs for anaphora resolution."""
        recent_fn = getattr(self._tier3, "recent", None)
        if recent_fn is None:
            return ()
        try:
            turns = recent_fn(str(chat_id))
        except Exception:
            logger.exception("harness_dispatcher.recent_turns_failed")
            return ()
        out: list[tuple[str, str]] = []
        for t in turns:
            role = getattr(t, "role", None)
            text = getattr(t, "text", None)
            if isinstance(role, str) and isinstance(text, str):
                out.append((role, text))
        return tuple(out)

    async def _synthesize_chain(
        self,
        user_text: str,
        history: list[tuple[ToolIntent, ToolResult]],
        *,
        chat_id: int,
    ) -> str:
        """Render the chain history into a single operator-facing reply.

        Mirrors `_synthesize` (single-shot) but feeds the chain prompt the
        full ordered list of (call, result) pairs and the verified-tools
        set. Fallback on synthesizer failure is the LAST tool's payload —
        that's the most-recent state the operator was waiting on.
        """
        logger.info("harness_dispatcher.chain_synthesis.tier1_load_start")
        try:
            snap = self._tier1_loader.load(str(chat_id))
            identity = snap.identity or "AEGIS, an operator-facing assistant"
        except Exception:
            logger.exception("harness_dispatcher.chain_synthesis.tier1_load_failed")
            identity = "AEGIS, an operator-facing assistant"
        logger.info("harness_dispatcher.chain_synthesis.tier1_load_done")

        chain_text = _render_chain(history)
        verified_tools = ", ".join(sorted({call.tool for call, _ in history})) or "(none)"
        last_payload_text = _last_payload_text(history)

        logger.info("harness_dispatcher.chain_synthesis.prompt_read_start")
        try:
            prompt_template = _SYNTHESIS_CHAIN_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.exception("harness_dispatcher.chain_synthesis.prompt_read_failed")
            return _clip(last_payload_text)
        logger.info(
            "harness_dispatcher.chain_synthesis.prompt_read_done",
            extra={"chars": len(prompt_template)},
        )

        system = prompt_template.format(
            identity=identity,
            user_text=user_text,
            tool_chain=chain_text,
            verified_tools=verified_tools,
        )
        request = ChatRequest(
            model=self._synthesis_model,
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user_text),
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        logger.info(
            "harness_dispatcher.chain_synthesis.chat_start",
            extra={
                "model": self._synthesis_model,
                "chain_steps": len(history),
                "chain_chars": len(chain_text),
            },
        )
        try:
            response = await self._synthesizer.chat(request)
            logger.info(
                "harness_dispatcher.chain_synthesis.chat_done",
                extra={"reply_chars": len(response.content)},
            )
            return response.content
        except Exception:
            logger.exception("harness_dispatcher.chain_synthesis_failed")
            return _clip(last_payload_text)

    async def _synthesize(
        self,
        user_text: str,
        tool_intent: ToolIntent,
        result: ToolResult,
        *,
        chat_id: int,
    ) -> str:
        logger.info("harness_dispatcher.synthesis.tier1_load_start")
        try:
            snap = self._tier1_loader.load(str(chat_id))
            identity = snap.identity or "AEGIS, an operator-facing assistant"
        except Exception:
            logger.exception("harness_dispatcher.synthesis.tier1_load_failed")
            identity = "AEGIS, an operator-facing assistant"
        logger.info("harness_dispatcher.synthesis.tier1_load_done")

        if result.status == "error" and result.error:
            tool_result_text = result.error
        elif result.payload is not None:
            tool_result_text = str(result.payload)
        else:
            tool_result_text = "(empty)"

        logger.info("harness_dispatcher.synthesis.prompt_read_start")
        try:
            prompt_template = _SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            logger.exception("harness_dispatcher.synthesis.prompt_read_failed")
            return _clip(tool_result_text)
        logger.info(
            "harness_dispatcher.synthesis.prompt_read_done",
            extra={"chars": len(prompt_template)},
        )

        system = prompt_template.format(
            identity=identity,
            user_text=user_text,
            tool=tool_intent.tool,
            tool_result=tool_result_text,
        )
        request = ChatRequest(
            model=self._synthesis_model,
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user_text),
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        logger.info(
            "harness_dispatcher.synthesis.chat_start",
            extra={"model": self._synthesis_model, "tool_result_chars": len(tool_result_text)},
        )
        try:
            response = await self._synthesizer.chat(request)
            logger.info(
                "harness_dispatcher.synthesis.chat_done",
                extra={"reply_chars": len(response.content)},
            )
            return response.content
        except Exception:
            logger.exception("harness_dispatcher.synthesis_failed")
            return _clip(tool_result_text)
