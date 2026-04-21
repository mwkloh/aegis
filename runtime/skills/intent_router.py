"""Intent → skill router.

Phase 7/8 bridge. `runtime/chat/pipeline.py` §3.3 step 2 called out an
intent classifier as a future enhancement; this module ships the
deterministic half of that story. No LLM, no fuzzy matching — just a
normalized-phrase substring check against the intents every skill
declares in its YAML descriptor.

Why deterministic first:

* **Auditable.** Every short-circuit decision is reproducible from the
  input text alone. A future LLM classifier can layer on top without
  changing the contract (``match(text) -> SkillDescriptor | None``).
* **Safe by default.** A miss falls through to the regular chat path
  — nothing is degraded, the operator just gets an LLM reply as before.
* **First-writer-wins.** Order of descriptors (registry insertion
  order, which follows sorted catalog filenames) determines priority
  when two skills declare overlapping intents. Longer phrases win
  within that ordering so ``morning_brief`` beats ``brief``.
"""
from __future__ import annotations

import re

from runtime.skills.registry import SkillDescriptor, SkillRegistry

# Any run of non-alphanumeric characters collapses to a single space.
# Handles underscores, punctuation, stray emoji — normalizing both the
# intent label (``morning_brief`` → ``morning brief``) and the user's
# text (``send me a morning-brief, please!`` → ``send me a morning brief please``).
_WORD_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    lowered = text.lower()
    collapsed = _WORD_SPLIT_RE.sub(" ", lowered)
    return " ".join(collapsed.split())


def _intent_phrase(intent: str) -> str:
    return _normalize(intent)


class IntentRouter:
    """Maps user text to a declared skill intent.

    Match rule: the normalized intent phrase must appear as a
    whitespace-bounded token sequence inside the normalized user text.
    "morning brief" matches "send me a morning brief" and "show the
    morning brief now" but not "briefing me on the morning meeting".
    """

    def __init__(self, registry: SkillRegistry) -> None:
        pairs: list[tuple[str, SkillDescriptor]] = []
        for descriptor in registry.all():
            for intent in descriptor.intents:
                phrase = _intent_phrase(intent)
                if phrase:
                    pairs.append((phrase, descriptor))
        # Longer phrases first so a more specific intent wins when
        # shorter ones would also match (e.g. a future "brief" intent
        # must not steal a request that also mentions "morning brief").
        pairs.sort(key=lambda p: -len(p[0]))
        self._pairs: tuple[tuple[str, SkillDescriptor], ...] = tuple(pairs)

    def match(self, text: str) -> SkillDescriptor | None:
        normalized = _normalize(text)
        if not normalized:
            return None
        padded = f" {normalized} "
        for phrase, descriptor in self._pairs:
            if f" {phrase} " in padded:
                return descriptor
        return None


__all__ = ["IntentRouter"]
