"""How much of a tool result the model gets to see.

Two prompts carry tool results back to the model: the planner's per-step prompt and
the chain-synthesis prompt. Each used to clip to 1,024 characters of the
*JSON-encoded* payload, with a separate copy of the constant. Measured 2026-10-04
(docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md, section 14):

* No rationale or measurement stood behind 1,024.
* Encoding cost ~12% of it -- a newline became two characters and every result was
  wrapped in `{"result": "..."}` -- and the cut landed mid-JSON.
* Prefill speed differs ~12x between the benchmarked models (2,500 vs ~185 tok/s),
  so the same clip costs +0.5 s on one and +6.4 s per planner call on another. Latency
  on this device, not context size, is the constraint; so the bound is per model.

This module is the single place those rules live: decode the result, give each one a
per-model limit, and share a per-turn budget so a long chain cannot compound.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from runtime.harness.contract import ToolResult

# Per-result default. Equals `files_read`'s own page size plus its notice, so the
# tool and the prompt stop disagreeing about how much is "a result". A hypothesis
# until the sweep in the doc above picks the knee; override per model with
# `ModelProfile.tool_result_chars`.
DEFAULT_TOOL_RESULT_CHARS = 3500

# Output-token budget for the planner's structured decisions (thinking included).
# Kept at the long-standing 512 until a sweep decides otherwise, so a run with no
# profile is still the clean baseline the sweep is measured against. Measured on run 2
# (full arm, easy suite): qwen3-vl:4b had 81 calls cut off by it and all 30 of its
# failures were `thinking_budget_exhausted`; qwen3.5:4b-mlx hit it in 13 of 45 variants
# and paid in retries. The dispatcher's free-text synthesis already allows 2,048.
DEFAULT_PLANNER_MAX_TOKENS = 512

# How much of a file one `files_read` call returns. Below the per-result default on
# purpose: the page plus its continuation notice (~120 chars) must fit inside the
# planner's limit, or the planner would clip the very notice that tells the model how
# to read on.
FILES_READ_PAGE_CHARS = 3000

# Total tool-result text one turn may put in a prompt, newest results first. The
# planner re-sends every earlier result at each step, so without this a 5-step
# chain carries ~5x the per-result limit.
DEFAULT_TURN_RESULT_BUDGET_CHARS = 8000

# An exhausted budget leaves each older result this much -- a stub, never nothing, so
# the model can still see that the call happened and roughly what it returned.
MIN_RESULT_STUB_CHARS = 200


def result_text(result: ToolResult) -> str:
    """The result as the model should read it: decoded, not JSON-encoded.

    A plain `{"result": "..."}` payload is just the text. Anything structured stays
    JSON, with sorted keys so the prompt is stable. An error is prefixed so it can
    never be mistaken for output.
    """
    if result.status == "error":
        return f"ERROR: {result.error or '(no detail)'}"
    payload = result.payload
    if not payload:
        return "(empty)"
    if set(payload) == {"result"} and isinstance(payload["result"], str):
        return payload["result"] or "(empty)"
    return json.dumps(payload, sort_keys=True, default=str)


def clip_text(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, saying how much was shown and how much there was.

    The total matters: a model told only "(truncated)" cannot tell a file that is a
    little too long from one that is 80 KB, and so cannot decide whether to page.
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(truncated: showing {limit:,} of {len(text):,} chars)"


def allocate_result_limits(sizes: Sequence[int], *, per_result: int, total: int) -> list[int]:
    """A character limit for each result, oldest first, spending `total` newest first.

    Each result is capped at `per_result` and by what is left of the budget, but never
    below `MIN_RESULT_STUB_CHARS`. A result shorter than its limit uses only its own
    length, so small results never eat budget they do not need.
    """
    limits: list[int] = []
    remaining = total
    for size in reversed(sizes):
        wanted = min(size, per_result)
        allowed = min(wanted, max(remaining, MIN_RESULT_STUB_CHARS))
        limits.append(allowed)
        remaining = max(0, remaining - allowed)
    limits.reverse()
    return limits


__all__ = [
    "DEFAULT_PLANNER_MAX_TOKENS",
    "DEFAULT_TOOL_RESULT_CHARS",
    "DEFAULT_TURN_RESULT_BUDGET_CHARS",
    "FILES_READ_PAGE_CHARS",
    "MIN_RESULT_STUB_CHARS",
    "allocate_result_limits",
    "clip_text",
    "result_text",
]
