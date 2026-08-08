"""Reply verdict gate — flag unverified tool claims in LLM output.

Phase 8 §D2. An LLM's reply is a *claim*; whether an action actually
happened is determined by the Plane-1 ``tool.invoked`` records (§D1).
This module detects when a reply asserts "I did X" but the audit
trail shows no verified tool call for the action implied by X, and
prepends:

    ⚠️ unverified tool claim — not executed

The gate is deliberately conservative. We would rather miss a few
weasel phrasings than nag the operator on every harmless "I can help
with that" reply. The default pattern set fires only on first-person
assertions of concrete, verifiable actions (ran, executed, invoked,
published, applied, created a file, etc.) — not on offers, promises,
or hypotheticals.

Design:

* **Pure function.** ``annotate_unverified_claim(reply, verified_tools)``
  is deterministic; it doesn't touch the event stream or any store.
  The pipeline owns the "which tools actually ran this turn?" set
  and passes it in.
* **Per-tool match.** Each claim pattern is paired with the tool id
  it implies (or ``None`` for generic action verbs). When ``verified_tools``
  is non-empty, a specific-verb claim is flagged only if its implied
  tool id is *not* in the set; generic-verb claims are trusted because
  *some* tool ran. When ``verified_tools`` is empty, every claim hit
  is flagged — same as the legacy ``count == 0`` path.
* **Structural verdict.** ``ReplyVerdict`` returns the flagged
  phrases + annotated reply so callers can log which phrase tripped
  the gate without the caller re-running the regex.
* **Easy to extend.** ``_DEFAULT_CLAIM_PATTERNS`` is exposed as a
  module constant so operators who see false negatives in practice
  can patch in new patterns without touching the gate logic.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

UNVERIFIED_BANNER = "⚠️ unverified tool claim — not executed"


# Each entry is ``(implied_tool_id, regex)``. ``implied_tool_id`` is the
# harness tool id the verb maps to, or ``None`` for generic action verbs
# that don't pin a specific tool ("ran", "executed", "applied"…).
#
# When adding patterns:
#   - Anchor on first-person subject ("I ", "I've", "I have") or
#     imperative-result form ("Ran X", "Executed Y") to avoid
#     matching hypotheticals.
#   - Avoid matching offers ("I can run", "I will run", "would you
#     like me to run") — those aren't claims.
#   - Keep each regex small and readable; we want false-negative
#     leaning behaviour.
#   - Use ``None`` for generic verbs so that *any* verified tool
#     trusts the claim. Use a specific tool id when the verb has a
#     1:1 meaning ("deleted" → ``files_delete``).
_DEFAULT_CLAIM_PATTERNS: tuple[tuple[str | None, re.Pattern[str]], ...] = tuple(
    (tool_id, re.compile(p, re.IGNORECASE))
    for tool_id, p in (
        # Generic action verbs — trusted whenever any tool ran.
        (None, r"\bI(?:'ve| have)? (?:just )?(?:ran|run|executed|invoked|called) \b"),
        (None, r"\bI(?:'ve| have)? (?:just )?applied (?:the |your |a )?(?:patch|diff|change)\b"),
        (None, r"\bI(?:'ve| have)? (?:just )?(?:published|pushed|posted|sent) \b"),
        (None, r"\bI(?:'ve| have)? (?:just )?(?:fetched|downloaded|retrieved) \b"),
        (None, r"\b(?:ran|executed|invoked) (?:the )?(?:tool|skill|command)s?\b"),
        (None, r"\b(?:tool|skill|command) (?:ran|executed|succeeded|completed)\b"),
        # Terse completion claims — "I did.", "I've done that.", "I completed
        # the cleanup." These weasel phrasings let "say you ran X" prompts
        # slip past the gate in practice.
        (None, r"\bI(?:'ve| have)? (?:just )?(?:did|done)\b"),
        (None, r"\bI(?:'ve| have)? (?:just )?(?:completed|finished|cleaned up|cleared)\b"),
        # Specific-verb claims — pinned to a tool id.
        (
            "files_write",
            r"\bI(?:'ve| have)? (?:just )?(?:created|wrote|written|generated|saved) \b",
        ),
        ("files_delete", r"\bI(?:'ve| have)? (?:just )?(?:deleted|removed|dropped) \b"),
        ("files_move", r"\bI(?:'ve| have)? (?:just )?(?:moved|renamed) \b"),
        ("files_open", r"\bI(?:'ve| have)? (?:just )?opened \b"),
        ("files_search", r"\bI(?:'ve| have)? (?:just )?searched \b"),
        ("files_read", r"\bI(?:'ve| have)? (?:just )?read \b"),
        ("files_list", r"\bI(?:'ve| have)? (?:just )?listed \b"),
    )
)

# Patterns that suppress a match even when a claim pattern hit — these
# cover the common offer/promise phrasings that should never be
# annotated. Applied after ``_DEFAULT_CLAIM_PATTERNS``.
_DEFAULT_SUPPRESS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI (?:can|could|will|would|should|might|may) \b",
        r"\bwould you like me to \b",
        r"\bshall I \b",
        r"\bif you'?d like,? I\b",
        # Negations: "I did not …" / "I didn't …" / "I haven't …" must
        # not trip the new "I did/done" claim patterns above.
        r"\bI (?:did not|didn'?t|have not|haven'?t)\b",
    )
)


@dataclass(frozen=True)
class ReplyVerdict:
    """Outcome of one pass through the gate."""

    annotated_reply: str
    flagged_phrases: tuple[str, ...]

    @property
    def was_flagged(self) -> bool:
        return bool(self.flagged_phrases)


def annotate_unverified_claim(
    reply: str,
    *,
    verified_tools: Iterable[str],
    claim_patterns: tuple[tuple[str | None, re.Pattern[str]], ...] = _DEFAULT_CLAIM_PATTERNS,
    suppress_patterns: tuple[re.Pattern[str], ...] = _DEFAULT_SUPPRESS_PATTERNS,
) -> ReplyVerdict:
    """Prepend the banner if ``reply`` asserts an unverified action.

    - ``verified_tools`` is the set of tool ids that actually executed
      on this turn (from ``tool.invoked`` records with verdict
      ``"verified"``, or the in-memory chain history). Empty set means
      "no tools ran" — every claim hit is flagged.
    - When non-empty, a generic-verb claim (e.g. "I ran the tool")
      passes because *some* tool ran; a specific-verb claim
      (e.g. "I deleted the file") is flagged only if its implied tool
      isn't in the set.
    - ``claim_patterns`` and ``suppress_patterns`` are overridable for
      tests / operator tuning.
    - ``suppress_patterns`` wins: if any suppress pattern matches the
      reply, no annotation even if a claim pattern also hit.
    """
    if not reply.strip():
        return ReplyVerdict(annotated_reply=reply, flagged_phrases=())

    verified_set = frozenset(verified_tools)
    flagged: list[str] = []
    for tool_id, pattern in claim_patterns:
        match = pattern.search(reply)
        if match is None:
            continue
        if tool_id is None:
            # Generic verb: trusted whenever any tool ran.
            if not verified_set:
                flagged.append(match.group(0).strip())
            continue
        # Specific verb: trusted iff its implied tool actually ran.
        if tool_id not in verified_set:
            flagged.append(match.group(0).strip())

    if not flagged:
        return ReplyVerdict(annotated_reply=reply, flagged_phrases=())

    for pattern in suppress_patterns:
        if pattern.search(reply):
            return ReplyVerdict(annotated_reply=reply, flagged_phrases=())

    annotated = f"{UNVERIFIED_BANNER}\n\n{reply}"
    return ReplyVerdict(annotated_reply=annotated, flagged_phrases=tuple(flagged))


__all__ = [
    "UNVERIFIED_BANNER",
    "ReplyVerdict",
    "annotate_unverified_claim",
]
