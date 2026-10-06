"""The "bare" eval arm: a typical native-tool-calling loop, no AEGIS guardrails.

Every full-arm result came from the AEGIS planner protocol -- a forced JSON
schema (`remaining` / `kind` / `tool` / `args`) -- plus plan-lock, the
completion gate and unverified-claim annotation. That confounds the protocol
with the guardrails. This arm is what a harness looks like when it simply
trusts the model to tool-call: hand it the Ollama-native `tools` it was trained
on, run what it asks for, feed the results back, stop when it stops asking.

Held identical to the full arm so only the harness differs: the same models,
`num_ctx` / `think` profiles, sandboxed file tools, tasks, grader, telemetry
and reply capture. Deliberately absent: the intent classifier, the planner
prompt, plan-lock, the completion gate and claim annotation. See
docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md.

Ollama-served models only; the hosted OpenRouter configs have no bare arm yet.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.config import AegisConfig
from runtime.eval.grading import classify_failure, grade_calls
from runtime.eval.report import ObservedCall, VariantResult, VariantTelemetry
from runtime.eval.runner import _ObservingHarness, _seed_fixture
from runtime.eval.tasks import EvalTask, resolve_expected_calls, substitute_sandbox
from runtime.files.client import FilesClient
from runtime.harness import DEFAULT_TOOLS
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent
from runtime.harness.tools.files_tool import make_files_tools
from runtime.llm.clients import OllamaClient
from runtime.llm.clients.ollama_client import (
    _coerce_done_reason,
    _coerce_int,
    _ns_to_ms,
    _thinking_tokens,
)
from runtime.llm.telemetry import CallTelemetry, collect_calls, record_call
from runtime.skills.registry import SkillDescriptor, SkillRegistry

BARE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided tools when they help you "
    "complete the user's request, and answer in plain language when you are done."
)

# A typical harness does not squeeze a capable model into the planner's 512-token
# decision budget. Bounded rather than unlimited so a runaway thinker still ends.
_BARE_MAX_TOKENS = 2048
_MAX_TOOL_RESULT_CHARS = 8000

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


async def run_variant_bare(
    cfg: AegisConfig,
    registry: SkillRegistry,
    tier1_loader: Tier1Loader | None,
    task: EvalTask,
    variant_text: str,
) -> VariantResult:
    """Bare-arm counterpart of `runner.run_variant`: same signature, same result."""
    del tier1_loader  # the bare arm has no memory tier; kept for signature parity
    t0 = time.monotonic()
    sandbox = Path(tempfile.mkdtemp(prefix="aegis-eval-"))
    try:
        with collect_calls() as model_calls:
            return await _run_bare_inner(
                cfg, registry, task, variant_text, sandbox=sandbox, t0=t0, model_calls=model_calls
            )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


async def _run_bare_inner(
    cfg: AegisConfig,
    registry: SkillRegistry,
    task: EvalTask,
    variant_text: str,
    *,
    sandbox: Path,
    t0: float,
    model_calls: list[CallTelemetry],
) -> VariantResult:
    resolved_expected = resolve_expected_calls(task, sandbox)
    observing: _ObservingHarness | None = None
    reply: str | None = None

    def _result(*, passed: bool, reason: str) -> VariantResult:
        raw_calls = observing.calls if observing is not None else []
        telemetry = VariantTelemetry.from_calls(model_calls)
        return VariantResult(
            task_id=task.id,
            variant_text=variant_text,
            passed=passed,
            reason=reason,
            duration_s=time.monotonic() - t0,
            actual_calls=tuple(
                ObservedCall(tool=tool, args=args, status=status)
                for tool, args, status in raw_calls
            ),
            telemetry=telemetry,
            reply=reply,
            failure_kind=(
                None
                if passed
                else classify_failure(resolved_expected, raw_calls, telemetry)
            ),
        )

    if cfg.models.smart_provider != "ollama":
        return _result(passed=False, reason="the bare arm supports Ollama-served models only")

    try:
        _seed_fixture(task, sandbox)
        files_client = FilesClient(allowed_roots=[sandbox])
        observing = _ObservingHarness(
            HarnessAdapter(tools={**DEFAULT_TOOLS, **make_files_tools(files_client)})
        )
        harness = observing
        specs = build_tool_specs(registry.all(), harness.has_tool)
        client = OllamaClient(cfg)
        model = cfg.models.smart_local

        def _execute(tool: str, args: dict[str, Any]) -> str:
            try:
                intent = ToolIntent(tool=tool, args=args, skill_id=tool)
            except ValidationError as exc:
                return json.dumps({"error": f"invalid tool call: {exc.errors()[0]['msg']}"})
            result = harness.execute(intent)
            body = result.payload if result.status == "ok" else {"error": result.error}
            return json.dumps(body, default=str)[:_MAX_TOOL_RESULT_CHARS]

        async def _chat(history: list[dict[str, Any]]) -> dict[str, Any]:
            return await native_chat(
                client, cfg, model=model, messages=history, tools=specs, max_tokens=_BARE_MAX_TOKENS
            )

        loop = await run_native_loop(
            chat=_chat,
            execute=_execute,
            messages=[
                {"role": "system", "content": BARE_SYSTEM_PROMPT},
                {"role": "user", "content": substitute_sandbox(variant_text, sandbox)},
            ],
            max_steps=cfg.harness.max_steps,
        )
        reply = loop.reply
        grade = grade_calls(resolved_expected, harness.calls)
        return _result(passed=grade.passed, reason=grade.reason)
    except Exception as exc:  # eval harness must never crash the batch
        return _result(passed=False, reason=f"run_variant_bare raised: {exc!r}")


__all__ = [
    "BARE_SYSTEM_PROMPT",
    "NativeLoopResult",
    "build_tool_specs",
    "native_chat",
    "parse_tool_calls",
    "run_native_loop",
    "run_variant_bare",
]
