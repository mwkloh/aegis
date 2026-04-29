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


def _clip(text: str) -> str:
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
    ) -> None:
        self._classifier = classifier
        self._registry = registry
        self._runner = runner
        self._harness = harness
        self._synthesizer = synthesizer
        self._tier3 = tier3
        self._tier1_loader = tier1_loader
        self._synthesis_model = synthesis_model

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

        try:
            classification = await self._classifier.classify(user_text)
        except Exception:
            logger.exception("harness_dispatcher.classify_failed")
            return DispatchOutcome.PASS

        intent = classification.intent
        confidence = classification.confidence

        descriptor = self._registry.for_intent(intent)
        if descriptor is None:
            return DispatchOutcome.PASS

        if not self._harness.has_tool(descriptor.tool):
            return DispatchOutcome.PASS

        if confidence < HARNESS_CONFIDENCE_THRESHOLD:
            question = _clarify_question(descriptor)
            await _send(question)
            self._tier3.append(str(chat_id), "user", user_text)
            self._tier3.append(str(chat_id), "bot", question)
            return DispatchOutcome.CLARIFY

        recent = self._recent_turns(chat_id)
        tool_intent = await self._runner.build(descriptor, user_text, recent=recent)
        if tool_intent.tool == "respond":
            return DispatchOutcome.PASS

        result = self._harness.execute(tool_intent)
        reply_text = await self._synthesize(user_text, tool_intent, result, chat_id=chat_id)
        await _send(_clip(reply_text))
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
        return DispatchOutcome.FIRED

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

    async def _synthesize(
        self,
        user_text: str,
        tool_intent: ToolIntent,
        result: ToolResult,
        *,
        chat_id: int,
    ) -> str:
        try:
            snap = self._tier1_loader.load(str(chat_id))
            identity = snap.identity or "AEGIS, an operator-facing assistant"
        except Exception:
            identity = "AEGIS, an operator-facing assistant"

        if result.status == "error" and result.error:
            tool_result_text = result.error
        elif result.payload is not None:
            tool_result_text = str(result.payload)
        else:
            tool_result_text = "(empty)"

        try:
            prompt_template = _SYNTHESIS_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            return _clip(tool_result_text)

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
            max_tokens=512,
        )
        try:
            response = await self._synthesizer.chat(request)
            return response.content
        except Exception:
            logger.exception("harness_dispatcher.synthesis_failed")
            return _clip(tool_result_text)
