"""Reply verdict gate — flag unverified tool claims in LLM output.

Phase 8 §D2. An LLM's reply is a *claim*; whether an action actually
happened is determined by the Plane-1 ``tool.invoked`` records (§D1).
This module detects when a reply asserts "I did X" but the audit
trail shows no verified tool call for the turn, and prepends:

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
  The pipeline owns the "did any verified tool run this turn?"
  boolean and passes it in.
* **Structural verdict.** ``ReplyVerdict`` returns the flagged
  phrases + annotated reply so callers can log which phrase tripped
  the gate without the caller re-running the regex.
* **Easy to extend.** ``_DEFAULT_CLAIM_PATTERNS`` is exposed as a
  module constant so operators who see false negatives in practice
  can patch in new patterns without touching the gate logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNVERIFIED_BANNER = "⚠️ unverified tool claim — not executed"


# First-person assertion patterns for actions that *should* leave a
# tool.invoked verdict. Each pattern matches a single phrase; the gate
# aggregates hits and annotates once. Patterns are case-insensitive.
#
# When adding patterns:
#   - Anchor on first-person subject ("I ", "I've", "I have") or
#     imperative-result form ("Ran X", "Executed Y") to avoid
#     matching hypotheticals.
#   - Avoid matching offers ("I can run", "I will run", "would you
#     like me to run") — those aren't claims.
#   - Keep each regex small and readable; we want false-negative
#     leaning behaviour.
_DEFAULT_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bI(?:'ve| have)? (?:just )?(?:ran|run|executed|invoked|called) \b",
        r"\bI(?:'ve| have)? (?:just )?applied (?:the |your |a )?(?:patch|diff|change)\b",
        r"\bI(?:'ve| have)? (?:just )?(?:created|wrote|written|generated|saved) \b",
        r"\bI(?:'ve| have)? (?:just )?(?:deleted|removed|dropped) \b",
        r"\bI(?:'ve| have)? (?:just )?(?:published|pushed|posted|sent) \b",
        r"\bI(?:'ve| have)? (?:just )?(?:fetched|downloaded|retrieved) \b",
        r"\b(?:ran|executed|invoked) (?:the )?(?:tool|skill|command)s?\b",
        r"\b(?:tool|skill|command) (?:ran|executed|succeeded|completed)\b",
        # Terse completion claims — "I did.", "I've done that.", "I completed
        # the cleanup." These weasel phrasings were missing from the original
        # set and let "say you ran X" prompts slip past the gate in practice.
        r"\bI(?:'ve| have)? (?:just )?(?:did|done)\b",
        r"\bI(?:'ve| have)? (?:just )?(?:completed|finished|cleaned up|cleared)\b",
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
    verified_tools: int,
    claim_patterns: tuple[re.Pattern[str], ...] = _DEFAULT_CLAIM_PATTERNS,
    suppress_patterns: tuple[re.Pattern[str], ...] = _DEFAULT_SUPPRESS_PATTERNS,
) -> ReplyVerdict:
    """Prepend the banner if ``reply`` asserts an action without verification.

    - ``verified_tools`` is the count of ``tool.invoked`` events with
      ``verdict="verified"`` on record for this turn. When non-zero,
      we trust the claim and return the reply untouched.
    - ``claim_patterns`` is overridable for tests / operator tuning.
    - ``suppress_patterns`` wins: if any suppress pattern matches the
      reply, no annotation even if a claim pattern also hit.
    """
    if verified_tools > 0 or not reply.strip():
        return ReplyVerdict(annotated_reply=reply, flagged_phrases=())

    flagged: list[str] = []
    for pattern in claim_patterns:
        match = pattern.search(reply)
        if match is not None:
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
