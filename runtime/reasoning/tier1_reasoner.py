"""Tier 1 reasoning — frontier model produces a `ToolIntent` for a skill.

The reasoner is **optional**: skills with `requires_tier1=True` only get to
call it when an OpenRouter key is configured. Without a key, the skill_runner
returns a graceful `tool=respond, args={message: "tier1 unavailable: ..."}`
and emits a `pattern.tier1_missing` event for the reflection plane.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events.stream import EventStream
from runtime.harness import ToolIntent
from runtime.harness.contract import ToolResult
from runtime.llm.clients import ChatMessage
from runtime.llm.clients.base import ModelClient
from runtime.llm.structured_output import request_structured
from runtime.skills import SkillDescriptor

_PROMPT_PATH: Path = Path(__file__).parent / "prompts" / "tier1_skill.txt"
_PLANNER_PROMPT_PATH: Path = Path(__file__).parent / "prompts" / "tier1_planner.txt"
_MAX_USER_CHARS: int = 8192
_MAX_TOOL_RESULT_CHARS: int = 1024

_TAG_RE = re.compile(r"</?\s*[A-Za-z][A-Za-z0-9_-]*\s*[^>]*>")


def _sanitize_user_text(text: str) -> str:
    """Make user_text safe to interpolate into the system prompt.

    - Doubles `{` and `}` so .format() leaves user braces intact.
    - Replaces XML-ish opening/closing tags (`<foo>`, `</foo>`) with
      `&lt;foo&gt;` so the user cannot inject control markers like
      `</instructions>` that the LLM might honour.
    """
    escaped = text.replace("{", "{{").replace("}", "}}")
    return _TAG_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        escaped,
    )
_MAX_RECENT_TURNS: int = 6
_MAX_TURN_CHARS: int = 400
_MAX_HISTORY_CHARS: int = 2048


class Tier1Reply(BaseModel):
    """Strict shape of the model's reply. Anything else → reasoner declines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=1024)


class PlanStep(BaseModel):
    """Discriminated union over the three planner outcomes.

    `kind="tool_call"` → execute (`tool`, `args`) and call planner again.
    `kind="respond"` → terminate the loop; synthesise the final reply.
    `kind="task_complete"` → terminate the loop; `summary` states what was
    done, grounded in the tool results already in the chain history. The
    dispatcher gates task_complete summaries against the turn's evidence
    ledger (C2) — see docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern="^(tool_call|respond|task_complete)$")
    tool: str | None = Field(default=None, max_length=64)
    args: dict[str, Any] | None = Field(default=None)
    summary: str | None = Field(default=None, max_length=2048)


class Tier1ReasonerError(RuntimeError):
    """Raised when Tier 1 is structurally unable to produce a contract."""


class Tier1Reasoner:
    """Calls a frontier model to fill a skill's args. Stateless."""

    def __init__(
        self,
        client: object,
        model: str,
        prompt_path: Path | None = None,
        events: EventStream | None = None,
        max_retries: int = 1,
        planner_prompt_path: Path | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self._planner_prompt_template = (
            planner_prompt_path or _PLANNER_PROMPT_PATH
        ).read_text(encoding="utf-8")
        self._events = events
        self._max_retries = max_retries

    async def reason(
        self,
        descriptor: SkillDescriptor,
        user_text: str,
        *,
        recent: Sequence[tuple[str, str]] = (),
    ) -> ToolIntent:
        bounded = _sanitize_user_text(user_text[:_MAX_USER_CHARS])
        system = self._prompt_template.format(
            skill_id=descriptor.id,
            tool=descriptor.tool,
            args_schema=json.dumps(descriptor.args_schema, indent=2, sort_keys=True),
            conversation_history=_format_history(recent),
            user_text=bounded,
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=bounded),
        ]
        schema = _build_schema(_allowed_keys(descriptor))

        try:
            data, outcome = await request_structured(
                cast(ModelClient, self._client),
                messages,
                schema,
                model=self._model,
                temperature=0.0,
                max_tokens=512,
                max_retries=self._max_retries,
                events=self._events,
                call_site="reasoning.tier1",
            )
        except httpx.HTTPError as exc:
            raise Tier1ReasonerError(f"transport: {type(exc).__name__}") from exc

        if outcome.error_kind != "ok":
            raise Tier1ReasonerError(f"structured output failed: {outcome.error_kind}")

        try:
            reply = Tier1Reply.model_validate(data)
        except ValidationError as exc:
            raise Tier1ReasonerError(f"reply schema mismatch: {exc.errors()}") from exc

        return ToolIntent(
            tool=descriptor.tool,
            args=reply.args,
            skill_id=descriptor.id,
            rationale=reply.rationale or "tier1 reasoning",
        )


    async def plan_next(
        self,
        *,
        user_text: str,
        available_skills: Sequence[SkillDescriptor],
        history: Sequence[tuple[ToolIntent, ToolResult]] = (),
        recent: Sequence[tuple[str, str]] = (),
    ) -> PlanStep:
        """Ask the model for the next step in a multi-step chain.

        Returns either `kind="tool_call"` (with a validated tool from
        `available_skills`) or `kind="respond"`. Raises `Tier1ReasonerError`
        on transport failure or unrecoverable schema breakage.
        """
        if not available_skills:
            raise Tier1ReasonerError("plan_next requires at least one available skill")
        bounded = _sanitize_user_text(user_text[:_MAX_USER_CHARS])
        system = self._planner_prompt_template.format(
            available_tools=_format_available_tools(available_skills),
            conversation_history=_format_history(recent),
            call_history=_format_call_history(history),
            user_text=bounded,
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=bounded),
        ]
        schema = _build_planner_schema(available_skills)

        try:
            data, outcome = await request_structured(
                cast(ModelClient, self._client),
                messages,
                schema,
                model=self._model,
                temperature=0.0,
                max_tokens=512,
                max_retries=self._max_retries,
                events=self._events,
                call_site="reasoning.tier1.plan_next",
            )
        except httpx.HTTPError as exc:
            raise Tier1ReasonerError(f"transport: {type(exc).__name__}") from exc

        if outcome.error_kind != "ok":
            raise Tier1ReasonerError(f"structured output failed: {outcome.error_kind}")

        try:
            step = PlanStep.model_validate(data)
        except ValidationError as exc:
            raise Tier1ReasonerError(f"plan_next schema mismatch: {exc.errors()}") from exc

        if step.kind == "tool_call":
            allowed_tools = {d.tool for d in available_skills}
            if step.tool is None or step.tool not in allowed_tools:
                raise Tier1ReasonerError(
                    f"plan_next picked unknown tool: {step.tool!r}"
                )
        return step


def _format_history(turns: Sequence[tuple[str, str]]) -> str:
    if not turns:
        return "(no prior turns)"
    lines: list[str] = []
    for role, text in list(turns)[-_MAX_RECENT_TURNS:]:
        label = "USER" if role == "user" else "BOT"
        clipped = text if len(text) <= _MAX_TURN_CHARS else text[:_MAX_TURN_CHARS] + "…"
        lines.append(f"{label}: {clipped}")
    joined = "\n".join(lines)
    if len(joined) > _MAX_HISTORY_CHARS:
        joined = joined[-_MAX_HISTORY_CHARS:]
    return joined


def _format_available_tools(skills: Sequence[SkillDescriptor]) -> str:
    """Render the tool catalogue for the planner prompt."""
    lines: list[str] = []
    for d in skills:
        schema = json.dumps(d.args_schema, sort_keys=True)
        lines.append(f"- {d.tool}: {d.description} args={schema}")
    return "\n".join(lines) if lines else "(none)"


def _format_call_history(history: Sequence[tuple[ToolIntent, ToolResult]]) -> str:
    """Render prior (call, result) pairs from the current turn."""
    if not history:
        return "(no calls yet on this turn)"
    lines: list[str] = []
    for idx, (call, result) in enumerate(history, start=1):
        args_blob = json.dumps(call.args, sort_keys=True)
        if result.status == "error":
            payload = f"ERROR: {result.error or '(no detail)'}"
        else:
            payload = json.dumps(result.payload, sort_keys=True)
        if len(payload) > _MAX_TOOL_RESULT_CHARS:
            payload = payload[: _MAX_TOOL_RESULT_CHARS] + "…(truncated)"
        lines.append(f"{idx}. {call.tool}({args_blob}) → {payload}")
    return "\n".join(lines)


def _build_planner_schema(skills: Sequence[SkillDescriptor]) -> dict[str, Any]:
    """JSON schema accepting a `respond`, `task_complete`, or `tool_call` step.

    `args` is left as a free-form object — per-tool arg validation happens
    downstream in the harness adapter, mirroring the existing
    `Tier1Reasoner.reason` flow where the schema rejects extra args at the
    top level but trusts the harness to enforce nested constraints. It
    declares `"properties": {}` (rather than a bare `{"type": "object"}`) to
    stay GBNF-friendly, and pairs that with an explicit
    `"additionalProperties": True` — an object node with empty `properties`
    and no `additionalProperties` compiles (in llama.cpp's
    json-schema-to-grammar converter) to a grammar that matches ONLY `{}`,
    silently closing `args` at the decoder the moment this schema is sent
    to Ollama as a `format` constraint (Track A, A2). The explicit `True`
    keeps the decoder grammar open, matching the client-side
    Draft202012Validator's (already-open) interpretation.
    """
    tool_ids = sorted({d.tool for d in skills})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": ["tool_call", "respond", "task_complete"]},
            "tool": {"type": "string", "enum": tool_ids} if tool_ids else {"type": "string"},
            "args": {"type": "object", "properties": {}, "additionalProperties": True},
            "summary": {"type": "string", "maxLength": 2048},
        },
    }


def _allowed_keys(descriptor: SkillDescriptor) -> set[str]:
    schema = descriptor.args_schema
    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict):
        return {str(k) for k in props}
    return set()


def _build_schema(allowed: set[str]) -> dict[str, Any]:
    """Top-level reply schema — enforces extra-key rejection on args."""
    args_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {k: {} for k in sorted(allowed)},
    } if allowed else {"type": "object"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["args"],
        "properties": {
            "args": args_schema,
            "rationale": {"type": "string", "maxLength": 1024},
        },
    }
