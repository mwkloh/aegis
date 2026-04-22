"""One-shot critique-then-revise pass for a coding-harness draft.

Phase 5 Track B addition. Bounded by spec — exactly one critique
call followed by at most one revise call:

* If the critique returns ``{"issues": []}`` we skip revise and
  return the original draft (no extra model call).
* If the critique fails for any reason (HTTP error, malformed JSON,
  schema rejection) we degrade silently to the original draft. The
  failure is recorded as an event but never raised.
* If the revise step fails we degrade silently to the original draft
  for the same reason.
* If the revise output proposes a write to a canonical file we
  return a ``status="refused"`` draft (mirrors the post-call check
  in ``coder.draft_for``).

This module never raises and never re-enters itself — there is no
recursion, no convergence loop. Total Phase-5 budget when
``--with-context`` is on: 1 (draft) + 1 (critique) + 0-or-1 (revise).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events import EventStream, EventType
from runtime.improvement.coding_tasks import CodingTask
from runtime.llm.clients import ChatMessage, ModelClient
from runtime.llm.structured_output import request_structured

from .coder import _CODER_SCHEMA, _LLMReply, diff_touches_canon
from .context import ContextBundle
from .draft import Draft

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_CRITIC_PROMPT = _PROMPTS_DIR / "critic.txt"
_DEFAULT_REVISE_PROMPT = _PROMPTS_DIR / "revise.txt"

_TEMPERATURE = 0.0
_MAX_CRITIQUE_TOKENS = 1024
_MAX_REVISE_TOKENS = 2048
_MAX_ISSUES = 16
_MAX_ISSUE_LEN = 512


class _Critique(BaseModel):
    """Critic reply schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issues: list[str] = Field(default_factory=list, max_length=_MAX_ISSUES)


_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {"type": "string", "maxLength": _MAX_ISSUE_LEN},
            "maxItems": _MAX_ISSUES,
        },
    },
}


async def critique_then_revise(
    draft: Draft,
    task: CodingTask,
    context: ContextBundle | None,
    *,
    client: ModelClient | None,
    model: str,
    events: EventStream | None = None,
    critic_prompt_path: Path | None = None,
    revise_prompt_path: Path | None = None,
    when: datetime | None = None,
) -> Draft:
    """Run one critique → at-most-one revise. Returns a Draft, never raises.

    Inputs that make the pass a no-op (returning ``draft`` unchanged):
      * ``draft.status != "ok"`` — only "ok" drafts can be revised
      * ``draft.unified_diff == ""`` — nothing concrete to critique
      * ``client is None`` — no model configured
    """
    if client is None or draft.status != "ok" or not draft.unified_diff:
        return draft

    issues = await _run_critique(
        draft, task, client=client, model=model,
        events=events, prompt_path=critic_prompt_path,
    )
    if issues is None or not issues:
        return draft

    revised = await _run_revise(
        draft, task, issues, client=client, model=model,
        events=events, prompt_path=revise_prompt_path, when=when,
    )
    return revised if revised is not None else draft


# --- internals ---------------------------------------------------------------


async def _run_critique(
    draft: Draft,
    task: CodingTask,
    *,
    client: ModelClient,
    model: str,
    events: EventStream | None,
    prompt_path: Path | None,
) -> list[str] | None:
    """Returns the list of issues, or ``None`` on failure (graceful)."""
    _emit(events, EventType.HARNESS_CRITIQUE_START, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
    })
    template = (prompt_path or _DEFAULT_CRITIC_PROMPT).read_text(encoding="utf-8")
    rendered = _render_review_prompt(template, task, draft)

    try:
        data, outcome = await request_structured(
            client,
            [ChatMessage(role="system", content=rendered)],
            _CRITIQUE_SCHEMA,
            model=model,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_CRITIQUE_TOKENS,
            max_retries=0,
            events=events,
            call_site="harness.critique",
        )
        if outcome.error_kind != "ok":
            _emit(events, EventType.HARNESS_CRITIQUE_END, {
                "ct_id": task.ct_id,
                "imp_id": task.imp_id,
                "model": model,
                "status": "failed",
                "error": f"structured:{outcome.error_kind}"[:256],
            })
            return None
        critique = _Critique.model_validate(data)
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        _emit(events, EventType.HARNESS_CRITIQUE_END, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "model": model,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:256],
        })
        return None

    issues = [
        s.strip()[:_MAX_ISSUE_LEN]
        for s in critique.issues
        if isinstance(s, str) and s.strip()
    ]
    status = "skipped" if not issues else "ok"
    _emit(events, EventType.HARNESS_CRITIQUE_END, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
        "status": status,
        "issue_count": len(issues),
    })
    return issues


async def _run_revise(
    draft: Draft,
    task: CodingTask,
    issues: list[str],
    *,
    client: ModelClient,
    model: str,
    events: EventStream | None,
    prompt_path: Path | None,
    when: datetime | None,
) -> Draft | None:
    """Returns the revised draft, or ``None`` on failure (graceful)."""
    _emit(events, EventType.HARNESS_REVISE_START, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
        "issue_count": len(issues),
    })
    template = (prompt_path or _DEFAULT_REVISE_PROMPT).read_text(encoding="utf-8")
    rendered = _render_revise_prompt(template, task, draft, issues)

    try:
        data, outcome = await request_structured(
            client,
            [ChatMessage(role="system", content=rendered)],
            _CODER_SCHEMA,
            model=model,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_REVISE_TOKENS,
            max_retries=0,
            events=events,
            call_site="harness.revise",
        )
        if outcome.error_kind != "ok":
            _emit(events, EventType.HARNESS_REVISE_END, {
                "ct_id": task.ct_id,
                "imp_id": task.imp_id,
                "model": model,
                "status": "failed",
                "error": f"structured:{outcome.error_kind}"[:256],
            })
            return None
        reply = _LLMReply.model_validate(data)
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        _emit(events, EventType.HARNESS_REVISE_END, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "model": model,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:256],
        })
        return None

    drafted_at = when or datetime.now(tz=UTC)
    diff_hit = diff_touches_canon(reply.unified_diff)
    if diff_hit:
        _emit(events, EventType.HARNESS_REFUSED, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "reason": "revise_diff_touches_canon",
            "path": diff_hit,
        })
        _emit(events, EventType.HARNESS_REVISE_END, {
            "ct_id": task.ct_id,
            "imp_id": task.imp_id,
            "model": model,
            "status": "refused",
            "path": diff_hit,
        })
        return Draft(
            ct_id=task.ct_id,
            imp_id=task.imp_id,
            model=model,
            drafted_at=drafted_at,
            summary=reply.summary,
            unified_diff="",
            test_notes=reply.test_notes,
            rollback=reply.rollback,
            status="refused",
            reason=f"revise diff touches canon: {diff_hit}",
        )

    _emit(events, EventType.HARNESS_REVISE_END, {
        "ct_id": task.ct_id,
        "imp_id": task.imp_id,
        "model": model,
        "status": "ok",
    })
    return Draft(
        ct_id=task.ct_id,
        imp_id=task.imp_id,
        model=model,
        drafted_at=drafted_at,
        summary=reply.summary,
        unified_diff=reply.unified_diff,
        test_notes=reply.test_notes,
        rollback=reply.rollback,
        status="ok",
        reason="",
    )


def _render_review_prompt(template: str, task: CodingTask, draft: Draft) -> str:
    scope_blob = ", ".join(task.scope) if task.scope else "(empty)"
    return (
        template.replace("{ct_id}", task.ct_id)
        .replace("{imp_id}", task.imp_id)
        .replace("{scope}", scope_blob)
        .replace("{constraints}", task.constraints)
        .replace("{expected_output}", task.expected_output)
        .replace("{draft_summary}", draft.summary or "(empty)")
        .replace("{draft_unified_diff}", draft.unified_diff or "(empty)")
        .replace("{draft_test_notes}", draft.test_notes or "(empty)")
        .replace("{draft_rollback}", draft.rollback or "(empty)")
    )


def _render_revise_prompt(
    template: str, task: CodingTask, draft: Draft, issues: list[str],
) -> str:
    issues_blob = "\n".join(f"- {issue}" for issue in issues) or "- (none)"
    base = _render_review_prompt(template, task, draft)
    return base.replace("{critique_issues}", issues_blob)


def _emit(
    events: EventStream | None, kind: EventType, payload: dict[str, object]
) -> None:
    if events is not None:
        events.append(kind, payload)
