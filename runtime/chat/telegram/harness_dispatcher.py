"""HarnessDispatcher — pre-pipeline tool-use layer for Telegram free-form chat."""
from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.llm.clients.base import ChatMessage, ChatRequest, ModelClient
from runtime.reasoning.skill_runner import SkillRunner
from runtime.skills.registry import SkillDescriptor, SkillRegistry

logger = logging.getLogger(__name__)

HARNESS_CONFIDENCE_THRESHOLD = 0.7
_MAX_REPLY_CHARS = 3500
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

__all__ = ["DispatchOutcome", "HarnessDispatcher"]


class DispatchOutcome(enum.Enum):
    FIRED = "fired"
    CLARIFY = "clarify"
    PASS = "pass"


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
            tool_intent = await self._plan_first_step(
                descriptor=descriptor,
                user_text=user_text,
                recent=recent,
            )
            if tool_intent is None:
                return DispatchOutcome.PASS
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

    async def _plan_first_step(
        self,
        *,
        descriptor: SkillDescriptor,
        user_text: str,
        recent: tuple[tuple[str, str], ...],
    ) -> ToolIntent | None:
        """Single-step scaffolding for the multi-step loop (Step 1).

        Calls `runner.plan_next` once with an empty history and translates the
        result into a `ToolIntent`. Returning None means "respond" — caller
        treats it as PASS, mirroring the existing `tool_intent.tool == 'respond'`
        branch on the legacy path. The full bounded loop (history threading,
        step cap, mid-chain abort) lands in Step 2 of
        docs/PLAN_MULTI_STEP_AGENT_LOOP.md.
        """
        available = list(self._registry.all())
        logger.info(
            "harness_dispatcher.plan_next_start",
            extra={"skill_id": descriptor.id, "available_skills": len(available)},
        )
        step = await self._runner.plan_next(
            user_text=user_text,
            available_skills=available,
            history=(),
            recent=recent,
        )
        logger.info(
            "harness_dispatcher.plan_next_done kind=%s tool=%s",
            step.kind,
            step.tool,
        )
        if step.kind != "tool_call" or step.tool is None:
            return None
        return ToolIntent(
            tool=step.tool,
            args=dict(step.args or {}),
            skill_id=descriptor.id,
            rationale="multi-step planner: first step",
        )

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
