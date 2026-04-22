"""Ask the coding model for a single draft, with canon-safety guards.

The coder enforces three discipline gates:

1. **Pre-call scope check** — refuses without an LLM call if the
   `CodingTask.scope` lists a canonical file or anything under
   `coding_harness/`.
2. **Stub-on-failure** — `httpx.HTTPError`, malformed JSON, or
   schema-rejected output produces a `Draft(status="stub")` instead
   of raising. The workflow always exits cleanly.
3. **Post-call diff check** — if the returned `unified_diff` proposes
   a write to a canonical file, the draft is converted to
   `status="refused"` (the diff text is dropped from the patch.md).
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events import EventStream, EventType
from runtime.improvement.coding_tasks import CodingTask
from runtime.llm.clients import ChatMessage, ModelClient
from runtime.llm.structured_output import request_structured

from .context import ContextBundle
from .draft import Draft

_DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "coder.txt"
_TEMPERATURE = 0.0
_MAX_TOKENS = 2048

_CANON_BASENAMES = (
    "AGENTS.md",
    "USER.md",
    "IDENTITY.md",
    "SOUL.md",
    "HEARTBEAT.md",
    "MEMORY.md",
    "SKILLS_INDEX.md",
)
_CODING_HARNESS_PREFIX = "coding_harness/"

_DIFF_TARGET_RE = re.compile(r"^\+\+\+ b/(\S+)$", re.MULTILINE)


class _LLMReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=0, max_length=512)
    unified_diff: str = Field(min_length=0, max_length=32_000)
    test_notes: str = Field(min_length=0, max_length=2_048)
    rollback: str = Field(min_length=0, max_length=2_048)


_CODER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "unified_diff", "test_notes", "rollback"],
    "properties": {
        "summary": {"type": "string", "maxLength": 512},
        "unified_diff": {"type": "string", "maxLength": 32_000},
        "test_notes": {"type": "string", "maxLength": 2_048},
        "rollback": {"type": "string", "maxLength": 2_048},
    },
}


def scope_touches_canon(paths: list[str]) -> str | None:
    """Return the first scope entry that names canon, else None."""
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path.startswith(_CODING_HARNESS_PREFIX) or path == "coding_harness":
            return path
        basename = path.rsplit("/", 1)[-1]
        if basename in _CANON_BASENAMES:
            return path
    return None


def diff_touches_canon(unified_diff: str) -> str | None:
    """Scan a unified diff for `+++ b/<canon>` writes. Returns the offending path."""
    for match in _DIFF_TARGET_RE.finditer(unified_diff or ""):
        target = match.group(1).strip()
        if target.startswith(_CODING_HARNESS_PREFIX):
            return target
        basename = target.rsplit("/", 1)[-1]
        if basename in _CANON_BASENAMES:
            return target
    return None


async def draft_for(
    task: CodingTask,
    *,
    client: ModelClient | None,
    model: str,
    available_skills: list[str] | None = None,
    events: EventStream | None = None,
    prompt_path: Path | None = None,
    context: ContextBundle | None = None,
    when: datetime | None = None,
) -> Draft:
    """Produce one `Draft` for `task`. Always returns; never raises."""
    drafted_at = when or datetime.now(tz=UTC)

    def _build(
        *,
        summary: str,
        unified_diff: str,
        test_notes: str,
        rollback: str,
        status: str,
        reason: str,
    ) -> Draft:
        return Draft(
            ct_id=task.ct_id,
            imp_id=task.imp_id,
            model=model,
            drafted_at=drafted_at,
            summary=summary,
            unified_diff=unified_diff,
            test_notes=test_notes,
            rollback=rollback,
            status=status,  # type: ignore[arg-type]
            reason=reason,
        )

    canon_hit = scope_touches_canon(list(task.scope))
    if canon_hit:
        _emit(events, EventType.HARNESS_REFUSED, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "reason": "scope_touches_canon",
            "path": canon_hit,
        })
        return _build(
            summary=f"Refused — task scope names canonical path {canon_hit!r}.",
            unified_diff="",
            test_notes="",
            rollback="",
            status="refused",
            reason=f"scope touches canon: {canon_hit}",
        )

    if client is None:
        return _build(
            summary="Coding model not configured — manual implementation required.",
            unified_diff="",
            test_notes="",
            rollback="",
            status="stub",
            reason="coding model not configured",
        )

    _emit(events, EventType.HARNESS_DRAFT_START, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
    })

    template = (prompt_path or _DEFAULT_PROMPT).read_text(encoding="utf-8")
    skills_blob = ", ".join(available_skills or []) or "(none provided)"
    try:
        reply = await _ask(
            client, model, template, skills_blob, task, context, events,
        )
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        reason = f"{type(exc).__name__}: {exc}"[:256]
        _emit(events, EventType.HARNESS_DRAFT_END, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "model": model,
            "status": "stub",
            "error": reason,
        })
        return _build(
            summary="Draft failed — coding model error.",
            unified_diff="",
            test_notes="",
            rollback="",
            status="stub",
            reason=reason,
        )
    if reply is None:
        reason = "structured_output_failed"
        _emit(events, EventType.HARNESS_DRAFT_END, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "model": model,
            "status": "stub",
            "error": reason,
        })
        return _build(
            summary="Draft failed — coding model error.",
            unified_diff="",
            test_notes="",
            rollback="",
            status="stub",
            reason=reason,
        )

    diff_hit = diff_touches_canon(reply.unified_diff)
    if diff_hit:
        _emit(events, EventType.HARNESS_REFUSED, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "reason": "diff_touches_canon",
            "path": diff_hit,
        })
        return _build(
            summary=reply.summary,
            unified_diff="",
            test_notes=reply.test_notes,
            rollback=reply.rollback,
            status="refused",
            reason=f"diff touches canon: {diff_hit}",
        )

    _emit(events, EventType.HARNESS_DRAFT_END, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
        "status": "ok",
    })
    return _build(
        summary=reply.summary,
        unified_diff=reply.unified_diff,
        test_notes=reply.test_notes,
        rollback=reply.rollback,
        status="ok",
        reason="",
    )


async def _ask(
    client: ModelClient,
    model: str,
    template: str,
    skills_blob: str,
    task: CodingTask,
    context: ContextBundle | None,
    events: EventStream | None,
) -> _LLMReply | None:
    scope_blob = ", ".join(task.scope) if task.scope else "(empty)"
    rendered = (
        template.replace("{ct_id}", task.ct_id)
        .replace("{imp_id}", task.imp_id)
        .replace("{scope}", scope_blob)
        .replace("{constraints}", task.constraints)
        .replace("{expected_output}", task.expected_output)
        .replace("{available_skills}", skills_blob)
    )
    if context is not None:
        blob = _format_context_blob(context)
        if blob:
            marker = "## Approved task"
            if marker in rendered:
                rendered = rendered.replace(marker, blob + "\n\n" + marker, 1)
            else:
                rendered = rendered.rstrip() + "\n\n" + blob + "\n"
    data, outcome = await request_structured(
        client,
        [ChatMessage(role="system", content=rendered)],
        _CODER_SCHEMA,
        model=model,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        max_retries=0,
        events=events,
        call_site="harness.coder",
    )
    if outcome.error_kind != "ok":
        return None
    return _LLMReply.model_validate(data)


def _format_context_blob(context: ContextBundle) -> str:
    """Render a ``ContextBundle`` as a readable section for the coder prompt.

    Empty bundles render as the empty string so the caller can decide
    not to inject any section at all (preserving Phase 4 bit-identity
    when ``context`` carries no useful payload).
    """
    if not context.files and not context.skills:
        return ""
    parts: list[str] = ["## In-scope context (read-only)"]
    if context.truncated:
        parts.append(
            "_Some context was withheld or truncated to fit the budget._",
        )
    for fslice in context.files:
        parts.append("")
        parts.append(f"### {fslice.path}")
        parts.append("```")
        parts.append(fslice.content.rstrip("\n"))
        parts.append("```")
    for sslice in context.skills:
        parts.append("")
        parts.append(f"### {sslice.path} (skill)")
        parts.append("```yaml")
        parts.append(sslice.content.rstrip("\n"))
        parts.append("```")
    return "\n".join(parts)


def _emit(
    events: EventStream | None, kind: EventType, payload: dict[str, object]
) -> None:
    if events is not None:
        events.append(kind, payload)
