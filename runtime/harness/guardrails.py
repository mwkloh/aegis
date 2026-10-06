"""Deterministic guardrails shared by the planner and native harness modes.

Pure functions over a turn's (call, result) history: no model calls, so none of them
costs latency. Extracted from `HarnessDispatcher` so the native tool-calling mode
(spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md) runs the same code
as the planner, and `Guardrails` lets an ablation switch each one off.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.harness.limits import (
    DEFAULT_TOOL_RESULT_CHARS,
    DEFAULT_TURN_RESULT_BUDGET_CHARS,
    allocate_result_limits,
    clip_text,
    result_text,
)
from runtime.tools.record import verdict_for_result

# Keep the dispatcher's logger name so existing log searches still match.
logger = logging.getLogger("runtime.chat.telegram.harness_dispatcher")

History = Sequence[tuple[ToolIntent, ToolResult]]

# Tools that mutate filesystem state. Both harness modes intercept one proposed
# beyond the first tool call and route the operator through a deterministic
# confirmation (docs/PLAN_MULTI_STEP_AGENT_LOOP.md §5 decision 5).
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"files_delete", "files_move", "files_write"})

# Path auto-fill (2026-08-21 handoff, open thread #2): models retype a plausible but
# wrong path for a `files_read` right after `files_search`; trust the search's own
# first match whenever the guess is not one of the actual matches.
_PATH_SOURCE_TOOL = "files_search"
_PATH_CONSUMER_TOOLS: frozenset[str] = frozenset({"files_read"})


@dataclass(frozen=True)
class Guardrails:
    """Which deterministic mechanisms a turn applies. Defaults = today's planner.

    `synthesis` is honoured only in native mode; the planner always synthesises on
    `respond` and never on `task_complete`. `plan_lock` is honoured only by the
    planner (it needs the planner's `remaining` list).
    """

    autofill: bool = True
    claim_gate: bool = True
    evidence: bool = True
    synthesis: bool = False
    plan_lock: bool = True


def format_args(args: dict[str, Any]) -> str:
    """Compact args repr — sorted keys for stable prompts."""
    if not args:
        return ""
    parts = [f"{k}={args[k]!r}" for k in sorted(args)]
    return ", ".join(parts)


def render_chain(
    history: History,
    *,
    per_result_chars: int = DEFAULT_TOOL_RESULT_CHARS,
    turn_budget_chars: int = DEFAULT_TURN_RESULT_BUDGET_CHARS,
) -> str:
    """Render the (call, result) chain into the chain-synthesis prompt slot.

    Shares the planner's rules (runtime.harness.limits): results are decoded text,
    each limited per model, the newest spending a shared per-turn budget first. It
    used to carry its own duplicate 1,024-character constant over the JSON-encoded
    payload, so the planner and the synthesiser could disagree about what a result
    said.
    """
    if not history:
        return "(no tools ran on this turn)"
    texts = [result_text(result) for _call, result in history]
    limits = allocate_result_limits(
        [len(t) for t in texts], per_result=per_result_chars, total=turn_budget_chars
    )
    lines: list[str] = []
    for idx, ((call, _result), text, limit) in enumerate(
        zip(history, texts, limits, strict=True), start=1
    ):
        args_blob = format_args(call.args)
        lines.append(f"{idx}. {call.tool}({args_blob}) → {clip_text(text, limit)}")
    return "\n".join(lines)


def with_evidence(
    summary: str, history: History, *, per_result_chars: int, turn_budget_chars: int
) -> str:
    """The summary followed by the tool output it summarises (no model call).

    An empty summary (native mode: a reasoning model that spent its budget thinking)
    yields just the results, never a reply that starts with blank lines.
    """
    if not history:
        return summary
    evidence = render_chain(
        history, per_result_chars=per_result_chars, turn_budget_chars=turn_budget_chars
    )
    if not summary.strip():
        return f"Results:\n{evidence}"
    return f"{summary}\n\nResults:\n{evidence}"


def history_verified_tools(history: History) -> set[str]:
    """Tools that actually succeeded on this turn, from in-memory history.

    Shared by `_gate_completion` (both its no-ledger and ledger-read-failed
    branches) and `dispatch()`'s respond-path verdict gate — a failed tool
    must never count as "verified" regardless of which of those call sites
    is asking (Phase 11 whole-branch review, C3/I3).

    Uses `verdict_for_result` rather than a bare `res.status == "ok"` check
    (Phase 11 review follow-up) — some tools report a soft failure inside
    an "ok" status (e.g. `run_command`'s `payload["verdict"] ==
    "exit_nonzero"` for a non-zero exit). A raw status check would still
    count that tool as verified here even though C4 already teaches the
    ledger path not to.
    """
    return {
        call.tool for call, res in history if verdict_for_result(res) == "verified"
    }


def autofill_path(tool_intent: ToolIntent, history: History) -> ToolIntent:
    """Correct `path` for a path-consuming tool immediately following a
    successful `files_search` step, when the model's own guess doesn't
    match any of that search's actual results. See `_PATH_CONSUMER_TOOLS`.

    Never overrides a guess that already matches a real result — this only
    ever turns a guaranteed-wrong path into the search's own top match, it
    never second-guesses a model that got it right.
    """
    if tool_intent.tool not in _PATH_CONSUMER_TOOLS or not history:
        return tool_intent
    prior_intent, prior_result = history[-1]
    if prior_intent.tool != _PATH_SOURCE_TOOL or prior_result.status != "ok":
        return tool_intent
    raw = str(prior_result.payload.get("result", ""))
    matches = [line for line in raw.splitlines() if line and line != "No matches."]
    if not matches or tool_intent.args.get("path") in matches:
        return tool_intent
    logger.info(
        "harness_dispatcher.path_autofilled",
        extra={
            "tool": tool_intent.tool,
            "guessed_path": tool_intent.args.get("path"),
            "corrected_path": matches[0],
        },
    )
    return tool_intent.model_copy(
        update={"args": {**tool_intent.args, "path": matches[0]}}
    )


__all__ = [
    "DESTRUCTIVE_TOOLS",
    "Guardrails",
    "autofill_path",
    "format_args",
    "history_verified_tools",
    "render_chain",
    "with_evidence",
]
