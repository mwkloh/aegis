"""Draft `Proposal` records from `PatternRecord`s using the Reflection LLM.

One LLM call per pattern. Output is a strict JSON object validated by
`request_structured` (JSON-schema) and then Pydantic. Failure modes
(LLM unreachable, malformed output, schema rejection) yield a structural
stub instead of raising — the reflection plane never breaks the run.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events import EventStream, EventType
from runtime.llm.clients import ChatMessage, ModelClient
from runtime.llm.structured_output import request_structured

from .patterns import PatternRecord

Risk = Literal["low", "medium", "high"]

_DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "proposal_drafter.txt"
_MAX_RATIONALE = 2048
_MAX_AFFECTED = 8
_PROPOSAL_TEMPERATURE = 0.0
_PROPOSAL_MAX_TOKENS = 512


class Proposal(BaseModel):
    """Reflection-plane suggestion. Never executed automatically."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64)
    pattern_detector: str = Field(min_length=1, max_length=64)
    affected: list[str] = Field(default_factory=list, max_length=_MAX_AFFECTED)
    change: str = Field(min_length=1, max_length=512)
    risk: Risk
    rationale: str = Field(min_length=0, max_length=_MAX_RATIONALE)


class _LLMReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    affected: list[str] = Field(default_factory=list, max_length=_MAX_AFFECTED)
    change: str = Field(min_length=1, max_length=512)
    risk: Risk
    rationale: str = Field(min_length=0, max_length=_MAX_RATIONALE)


_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["affected", "change", "risk", "rationale"],
    "properties": {
        "affected": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": _MAX_AFFECTED,
        },
        "change": {"type": "string", "minLength": 1, "maxLength": 512},
        "risk": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string", "maxLength": _MAX_RATIONALE},
    },
}


async def draft(
    patterns: Iterable[PatternRecord],
    *,
    client: ModelClient | None,
    model: str,
    available_skills: list[str] | None = None,
    events: EventStream | None = None,
    prompt_path: Path | None = None,
) -> list[Proposal]:
    """Return one `Proposal` per pattern (stub if the LLM step fails)."""
    template = (prompt_path or _DEFAULT_PROMPT).read_text(encoding="utf-8")
    skills_blob = ", ".join(available_skills or []) or "(none provided)"
    out: list[Proposal] = []
    for idx, pattern in enumerate(patterns, start=1):
        proposal_id = f"P-{idx:03d}"
        if client is None:
            out.append(_stub(proposal_id, pattern, "reflection LLM not configured"))
            continue
        try:
            reply = await _ask(client, model, template, skills_blob, pattern, events)
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            if events is not None:
                events.append(
                    EventType.PATTERN_OBSERVED,
                    {
                        "pattern": "proposal_parse_failed",
                        "detector": pattern.detector,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            out.append(_stub(proposal_id, pattern, f"parse failed: {type(exc).__name__}"))
            continue
        if reply is None:
            if events is not None:
                events.append(
                    EventType.PATTERN_OBSERVED,
                    {
                        "pattern": "proposal_parse_failed",
                        "detector": pattern.detector,
                        "error": "structured_output_failed",
                    },
                )
            out.append(_stub(proposal_id, pattern, "parse failed: structured_output"))
            continue
        out.append(
            Proposal(
                id=proposal_id,
                pattern_detector=pattern.detector,
                affected=reply.affected,
                change=reply.change,
                risk=reply.risk,
                rationale=reply.rationale,
            )
        )
    return out


async def _ask(
    client: ModelClient,
    model: str,
    template: str,
    skills_blob: str,
    pattern: PatternRecord,
    events: EventStream | None,
) -> _LLMReply | None:
    pattern_json = json.dumps(pattern.model_dump(), sort_keys=True)
    system = template.replace("{available_skills}", skills_blob).replace(
        "{pattern_json}", pattern_json
    )
    data, outcome = await request_structured(
        client,
        [ChatMessage(role="system", content=system)],
        _PROPOSAL_SCHEMA,
        model=model,
        temperature=_PROPOSAL_TEMPERATURE,
        max_tokens=_PROPOSAL_MAX_TOKENS,
        max_retries=0,
        events=events,
        call_site="reflection.proposal",
    )
    if outcome.error_kind != "ok":
        return None
    return _LLMReply.model_validate(data)


def _stub(proposal_id: str, pattern: PatternRecord, reason: str) -> Proposal:
    return Proposal(
        id=proposal_id,
        pattern_detector=pattern.detector,
        affected=[],
        change=f"(reflection LLM unavailable — manual review): {reason}",
        risk="low",
        rationale=f"Auto-generated stub for {pattern.detector!r}: {pattern.summary}",
    )
