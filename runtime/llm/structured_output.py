"""Schema-validated structured-output wrapper.

Phase 8 Track B §2.4: every LLM call that consumes a structured
answer goes through this wrapper. Local and frontier models both
benefit — the flow is:

1. Call the primary `client` with `response_format="json"`.
2. Parse the response body as JSON. On JSON parse failure, attempt
   deterministic repair (`repair_json`) before declaring `invalid_json`.
3. Validate against `schema` (draft-2020-12 JSON schema).
4. On parse/validate failure, append a corrective system message
   containing the failure reason + the schema, and retry up to
   `max_retries` times. Each retry emits `llm.structured_retry`.
5. If all retries exhausted and `escalate_to` is provided, try one
   more time on that target. Emits `llm.tier_escalated`.
6. If everything fails, emit `llm.structured_failed` and return the
   last raw body + an `error_kind="failed"` outcome — never raises.

The caller decides what to do with a failed outcome. Most sites
will degrade to a safe default (e.g. intent classifier → `general_question`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator, ValidationError

from runtime.events.stream import EventStream, EventType
from runtime.llm.clients.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ModelClient,
)

ErrorKind = Literal["ok", "invalid_json", "schema_violation", "empty", "failed"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def repair_json(content: str) -> str | None:
    """Deterministic salvage of near-miss JSON. Returns a parseable string
    or None. Never guesses at semantics — only removes wrapper noise:

    1. markdown fences (```json ... ```)
    2. prose before the first '{' / after the last '}'
    3. trailing commas before '}' or ']'
    """
    candidate = content.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = candidate[start : end + 1]
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


@dataclass(frozen=True)
class StructuredOutcome:
    """Explains how the structured call resolved.

    `error_kind="ok"` means the returned dict is valid against the
    schema (possibly after retries or escalation). Any other kind
    means `data` is the empty dict and the caller should fall back.
    """

    attempts: int
    escalated: bool
    error_kind: ErrorKind
    final_model: str
    repaired: bool = False


@dataclass(frozen=True)
class EscalationTarget:
    """Optional smarter client to try after the primary exhausts retries."""

    client: ModelClient
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024


def _corrective_messages(
    original: list[ChatMessage], schema: dict[str, Any], reason: str, last_raw: str
) -> list[ChatMessage]:
    """Append a corrective system turn describing what went wrong."""
    schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    snippet = last_raw[:500]
    corrective = ChatMessage(
        role="system",
        content=(
            "Your previous response could not be parsed as JSON that matches "
            "the required schema.\n\n"
            f"Failure: {reason}\n"
            f"Previous response (truncated): {snippet}\n\n"
            f"Required JSON schema:\n{schema_json}\n\n"
            "Respond ONLY with a single JSON object that matches the schema. "
            "Do not include any prose, markdown, or code fences."
        ),
    )
    return [*original, corrective]


def _validate(
    content: str, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, ErrorKind, str, bool]:
    """Parse + validate. Returns (data_or_None, error_kind, reason, repaired).

    `repaired` is True when the JSON only parsed after `repair_json`
    salvaged it (fences / prose / trailing commas stripped) — a signal
    for the caller, never part of the parse/validate contract itself.
    """
    stripped = content.strip()
    if not stripped:
        return None, "empty", "model returned empty content", False
    repaired = False
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        reason = f"json parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
        candidate = repair_json(stripped)
        if candidate is None:
            return None, "invalid_json", reason, False
        data = json.loads(candidate)
        repaired = True
    if not isinstance(data, dict):
        reason = f"expected JSON object, got {type(data).__name__}"
        return None, "schema_violation", reason, repaired
    try:
        Draft202012Validator(schema).validate(data)
    except ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return None, "schema_violation", f"schema violation at {path}: {exc.message}", repaired
    return data, "ok", "", repaired


async def _one_call(
    client: ModelClient,
    model: str,
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int,
    schema: dict[str, Any],
) -> ChatResponse:
    request = ChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format="json",
        response_schema=schema,
    )
    return await client.chat(request)


async def request_structured(
    client: ModelClient,
    messages: list[ChatMessage],
    schema: dict[str, Any],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 2,
    escalate_to: EscalationTarget | None = None,
    events: EventStream | None = None,
    call_site: str = "unknown",
) -> tuple[dict[str, Any], StructuredOutcome]:
    """Run a structured LLM call with retry + optional tier escalation.

    Never raises on bad model output — always returns an outcome.
    Transport errors from the client (e.g. connection refused) still
    propagate; those are a different failure class handled by the
    caller's own `try/except`.
    """
    attempts = 0
    current_messages = list(messages)
    last_reason = ""
    last_kind: ErrorKind = "failed"
    last_raw = ""

    # Primary client: up to 1 + max_retries attempts.
    for attempt_idx in range(max_retries + 1):
        attempts += 1
        resp = await _one_call(
            client, model, current_messages, temperature, max_tokens, schema
        )
        last_raw = resp.content
        data, kind, reason, repaired = _validate(resp.content, schema)
        if kind == "ok" and data is not None:
            if repaired and events is not None:
                events.append(
                    EventType.LLM_JSON_REPAIRED,
                    {"call_site": call_site, "model": model},
                )
            return data, StructuredOutcome(
                attempts=attempts,
                escalated=False,
                error_kind="ok",
                final_model=model,
                repaired=repaired,
            )
        last_kind, last_reason = kind, reason
        if events is not None and attempt_idx < max_retries:
            events.append(
                EventType.LLM_STRUCTURED_RETRY,
                {
                    "call_site": call_site,
                    "model": model,
                    "attempt": attempts,
                    "error_kind": kind,
                    "reason": reason[:200],
                },
            )
        # Append corrective turn before the next retry.
        if attempt_idx < max_retries:
            current_messages = _corrective_messages(
                messages, schema, reason, resp.content
            )

    # All primary retries exhausted — try escalation if available.
    if escalate_to is not None:
        if events is not None:
            events.append(
                EventType.LLM_TIER_ESCALATED,
                {
                    "call_site": call_site,
                    "from_model": model,
                    "to_model": escalate_to.model,
                    "primary_attempts": attempts,
                    "last_error_kind": last_kind,
                },
            )
        attempts += 1
        escalated_messages = _corrective_messages(
            messages, schema, last_reason, last_raw
        )
        resp = await _one_call(
            escalate_to.client,
            escalate_to.model,
            escalated_messages,
            escalate_to.temperature,
            escalate_to.max_tokens,
            schema,
        )
        data, kind, reason, repaired = _validate(resp.content, schema)
        if kind == "ok" and data is not None:
            if repaired and events is not None:
                # Deliberate: logs the model that actually produced the
                # repaired content (escalated), unlike LLM_STRUCTURED_FAILED
                # which always logs the primary model.
                events.append(
                    EventType.LLM_JSON_REPAIRED,
                    {"call_site": call_site, "model": escalate_to.model},
                )
            return data, StructuredOutcome(
                attempts=attempts,
                escalated=True,
                error_kind="ok",
                final_model=escalate_to.model,
                repaired=repaired,
            )
        last_kind, last_reason = kind, reason

    if events is not None:
        events.append(
            EventType.LLM_STRUCTURED_FAILED,
            {
                "call_site": call_site,
                "model": model,
                "attempts": attempts,
                "escalated": escalate_to is not None,
                "final_error_kind": last_kind,
                "reason": last_reason[:200],
            },
        )
    return {}, StructuredOutcome(
        attempts=attempts,
        escalated=escalate_to is not None,
        error_kind=last_kind if last_kind != "ok" else "failed",
        final_model=escalate_to.model if escalate_to is not None else model,
    )
