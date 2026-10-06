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
from pathlib import Path
from typing import Any

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
from runtime.harness.native_loop import (
    NATIVE_MAX_TOKENS as _BARE_MAX_TOKENS,
)
from runtime.harness.native_loop import (
    NATIVE_SYSTEM_PROMPT as BARE_SYSTEM_PROMPT,
)
from runtime.harness.native_loop import (
    NativeLoopResult,
    build_tool_specs,
    native_chat,
    parse_tool_calls,
    run_native_loop,
)
from runtime.harness.tools.files_tool import make_files_tools
from runtime.llm.clients import OllamaClient
from runtime.llm.telemetry import CallTelemetry, collect_calls
from runtime.skills.registry import SkillRegistry

# The control's flat clip on tool output.
_MAX_TOOL_RESULT_CHARS = 8000


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
