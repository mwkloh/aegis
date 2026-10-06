"""Ollama-native tool-calling loop: hand the model its trained `tools`, run what it
asks for, feed results back, stop when it stops asking.

Production code since 2026-10-06 (spec: docs/superpowers/specs/
2026-10-06-harness-ablation-design.md): the native harness mode uses it, and the
`bare` eval arm imports it unchanged as its control.
"""
from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from runtime.config import AegisConfig
from runtime.llm.clients import OllamaClient
from runtime.llm.clients.ollama_client import (
    _coerce_done_reason,
    _coerce_int,
    _ns_to_ms,
    _thinking_tokens,
)
from runtime.llm.telemetry import CallTelemetry, record_call
from runtime.skills.registry import SkillDescriptor

NATIVE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools when they help you "
    "complete the user's request, and answer in plain language when you are done."
)

# A typical harness does not squeeze a capable model into the planner's 512-token
# decision budget. Bounded rather than unlimited so a runaway thinker still ends.
NATIVE_MAX_TOKENS = 2048

_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def build_tool_specs(
    descriptors: Sequence[SkillDescriptor], has_tool: Callable[[str], bool]
) -> list[dict[str, Any]]:
    """Ollama `tools` entries for every skill whose tool the harness can run.

    `respond` is the planner's way of answering in prose; a native-tools model
    just answers, so offering it would be a tool that means "stop using tools".
    """
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in descriptors:
        if d.tool == "respond" or d.tool in seen or not has_tool(d.tool):
            continue
        seen.add(d.tool)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": d.tool,
                    "description": d.description,
                    "parameters": d.args_schema or _EMPTY_OBJECT_SCHEMA,
                },
            }
        )
    return specs


def parse_tool_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """`(tool, args)` for each call in an assistant message, tolerating bad shapes.

    Some chat templates return `arguments` as a JSON string, and a small model
    can emit a garbled one. A garbled blob becomes `{}` so it is graded as the
    model's mistake rather than crashing the variant.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = function.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


@dataclass(frozen=True)
class NativeLoopResult:
    reply: str | None
    steps: int
    hit_cap: bool


async def run_native_loop(
    *,
    chat: Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]],
    execute: Callable[[str, dict[str, Any]], str],
    messages: list[dict[str, Any]],
    max_steps: int,
) -> NativeLoopResult:
    """Call the model, run the tools it asks for, repeat until it stops asking.

    `steps` counts rounds of tool calls actually executed. Hitting `max_steps`
    with the model still asking for tools ends the loop with no reply -- the
    same hard cap the full harness applies (`harness.max_steps`).
    """
    history = list(messages)
    steps = 0
    while True:
        message = await chat(history)
        calls = parse_tool_calls(message)
        if not calls:
            return NativeLoopResult(
                reply=str(message.get("content") or "") or None, steps=steps, hit_cap=False
            )
        if steps >= max_steps:
            return NativeLoopResult(reply=None, steps=steps, hit_cap=True)
        history.append({**message, "role": "assistant"})
        for name, args in calls:
            history.append({"role": "tool", "tool_name": name, "content": execute(name, args)})
        steps += 1


async def native_chat(
    client: OllamaClient,
    cfg: AegisConfig,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    """One native-tools `/api/chat` call. Returns the assistant message dict.

    Reuses the production client's transport (loopback guard, retry policy,
    timeouts) and records the same `CallTelemetry`, so timeouts and token use
    are comparable with the full arm. `num_ctx` and `think` come from the same
    profiles the full arm uses.
    """
    options: dict[str, object] = {"temperature": 0.0, "num_predict": max_tokens}
    num_ctx = cfg.num_ctx_for(model)
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        payload["tools"] = tools
    think = cfg.think_for(model)
    if think is not None:
        payload["think"] = think

    started = time.perf_counter()
    attempts: list[int] = [1]
    try:
        data = await client._post_with_retry("/api/chat", payload, attempts)
    except httpx.ReadTimeout:
        record_call(
            CallTelemetry(
                provider="ollama",
                model=model,
                wall_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempts[0],
                timed_out=True,
            )
        )
        raise
    wall_ms = int((time.perf_counter() - started) * 1000)

    message = data.get("message", {})
    if not isinstance(message, dict):
        message = {}
    tokens_out = _coerce_int(data.get("eval_count", 0))
    record_call(
        CallTelemetry(
            provider="ollama",
            model=model,
            wall_ms=wall_ms,
            load_ms=_ns_to_ms(data.get("load_duration")),
            prompt_eval_ms=_ns_to_ms(data.get("prompt_eval_duration")),
            eval_ms=_ns_to_ms(data.get("eval_duration")),
            tokens_in=_coerce_int(data.get("prompt_eval_count", 0)),
            tokens_out=tokens_out,
            thinking_tokens=_thinking_tokens(message, tokens_out),
            done_reason=_coerce_done_reason(data.get("done_reason")),
            attempts=attempts[0],
        )
    )
    return message


__all__ = [
    "NATIVE_MAX_TOKENS",
    "NATIVE_SYSTEM_PROMPT",
    "NativeLoopResult",
    "build_tool_specs",
    "native_chat",
    "parse_tool_calls",
    "run_native_loop",
]
