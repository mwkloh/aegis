"""One native-mode turn: Ollama-native tool calls plus the loop-level guardrails.

Native harness mode (spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md)
lets the model drive its trained tool-calling format instead of the AEGIS JSON planner.
This module owns the loop: path autofill before each call, the always-on
destructive-tool guard, unknown-tool errors fed back to the model, and per-model
clipping of what the model sees. Reply construction (synthesis, claim gate, evidence)
belongs to the dispatcher, which runs it the same way for both modes.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.harness.guardrails import DESTRUCTIVE_TOOLS, autofill_path, strip_evidence
from runtime.harness.limits import clip_text, result_text
from runtime.harness.native_loop import NATIVE_SYSTEM_PROMPT, build_tool_specs, parse_tool_calls
from runtime.skills.registry import SkillDescriptor

NativeChat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class NativeTurnResult:
    history: list[tuple[ToolIntent, ToolResult]]
    final_text: str | None
    hit_cap: bool
    guarded_intent: ToolIntent | None = None


def native_system_prompt(identity: str) -> str:
    """The production native-mode system prompt, speaking as the tier1 identity.

    Kept apart from NATIVE_SYSTEM_PROMPT, which the eval's bare arm shares: editing
    this must not move the benchmark baseline. `identity` is the whole IDENTITY.md as
    `Tier1Loader` returns it (editor notes already stripped), a markdown file rather
    than a noun phrase, so it is framed as a block, as the synthesis prompts do.
    """
    return (
        f"You are the assistant described here:\n\n{identity}\n\n"
        "Use the provided tools when they help you complete the operator's request, "
        "and answer in plain language when you are done."
    )


def _recent_message(role: str, text: str, limit: int) -> dict[str, str]:
    """One prior chat turn as the model sees it.

    A bot reply loses its appended tool output: replayed as "assistant" the model
    would take untrusted file contents for its own words. Each turn is clipped like
    a tool result so history cannot crowd the system prompt out of a small context.
    """
    if role == "bot":
        text = strip_evidence(text) or "(replied with tool results)"
        return {"role": "assistant", "content": clip_text(text, limit)}
    return {"role": "user", "content": clip_text(text, limit)}


async def run_native_turn(
    *,
    chat: NativeChat,
    harness: Any,
    descriptors: Sequence[SkillDescriptor],
    user_text: str,
    skill_id: str,
    max_steps: int,
    autofill: bool,
    tool_result_chars: int,
    guard_min_call: int = 2,
    on_executed: Callable[[ToolIntent, ToolResult], None] | None = None,
    recent: Sequence[tuple[str, str]] = (),
    system_prompt: str = NATIVE_SYSTEM_PROMPT,
) -> NativeTurnResult:
    """Call the model, run the tools it asks for, repeat until it answers.

    `max_steps` caps rounds of tool calls, as `run_native_loop` does. A destructive
    tool at call number `guard_min_call` or later is returned unexecuted in
    `guarded_intent` so the dispatcher can ask the operator to confirm it.
    `recent` is the chat's prior (role, text) turns, oldest first; tier3 stores the
    bot's side as "bot", which the model sees as "assistant". `system_prompt` defaults
    to the eval's bare prompt; production passes `native_system_prompt(identity)`.
    """
    specs = build_tool_specs(descriptors, harness.has_tool)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *(_recent_message(role, text, tool_result_chars) for role, text in recent),
        {"role": "user", "content": user_text},
    ]
    history: list[tuple[ToolIntent, ToolResult]] = []
    rounds = 0
    call_no = 0
    while True:
        message = await chat(messages, specs)
        calls = parse_tool_calls(message)
        if not calls:
            text = str(message.get("content") or "")
            return NativeTurnResult(history=history, final_text=text or None, hit_cap=False)
        if rounds >= max_steps:
            return NativeTurnResult(history=history, final_text=None, hit_cap=True)
        rounds += 1
        messages.append({**message, "role": "assistant"})
        for name, args in calls:
            call_no += 1
            try:
                intent = ToolIntent(
                    tool=name, args=args, skill_id=skill_id, rationale=f"native: call {call_no}"
                )
            except ValidationError as exc:
                error = json.dumps({"error": f"invalid tool call: {exc.errors()[0]['msg']}"})
                messages.append({"role": "tool", "tool_name": name, "content": error})
                continue
            if autofill:
                intent = autofill_path(intent, history)
            if intent.tool in DESTRUCTIVE_TOOLS and call_no >= guard_min_call:
                return NativeTurnResult(
                    history=history, final_text=None, hit_cap=False, guarded_intent=intent
                )
            if harness.has_tool(intent.tool):
                result = harness.execute(intent)
            else:
                result = ToolResult(status="error", error=f"unknown tool: {intent.tool}")
            if on_executed is not None:
                on_executed(intent, result)
            history.append((intent, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": intent.tool,
                    "content": clip_text(result_text(result), tool_result_chars),
                }
            )


__all__ = ["NativeChat", "NativeTurnResult", "native_system_prompt", "run_native_turn"]
