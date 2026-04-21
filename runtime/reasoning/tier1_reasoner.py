"""Tier 1 reasoning — frontier model produces a `ToolIntent` for a skill.

The reasoner is **optional**: skills with `requires_tier1=True` only get to
call it when an OpenRouter key is configured. Without a key, the skill_runner
returns a graceful `tool=respond, args={message: "tier1 unavailable: ..."}`
and emits a `pattern.tier1_missing` event for the reflection plane.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events.stream import EventStream
from runtime.harness import ToolIntent
from runtime.model_router.clients import ChatMessage
from runtime.model_router.clients.base import ModelClient
from runtime.model_router.structured_output import request_structured
from runtime.skills import SkillDescriptor

_PROMPT_PATH: Path = Path(__file__).parent / "prompts" / "tier1_skill.txt"
_MAX_USER_CHARS: int = 8192


class Tier1Reply(BaseModel):
    """Strict shape of the model's reply. Anything else → reasoner declines."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=1024)


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
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self._events = events
        self._max_retries = max_retries

    async def reason(self, descriptor: SkillDescriptor, user_text: str) -> ToolIntent:
        bounded = user_text[:_MAX_USER_CHARS]
        system = self._prompt_template.format(
            skill_id=descriptor.id,
            tool=descriptor.tool,
            args_schema=json.dumps(descriptor.args_schema, indent=2, sort_keys=True),
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
