"""Deterministic skill-intent router.

Pins:

* Matches are whitespace-bounded (substring of words, not substring
  of characters) so "briefing the team" doesn't match "brief".
* Normalization collapses underscores, punctuation, case. So an
  intent of ``morning_brief`` matches "send me the Morning Brief!".
* Longer intents win when phrases would both fit — prevents a future
  shorter intent from stealing a more specific match.
* An empty registry or empty/whitespace text returns None — the
  caller must fall through to the chat pipeline unchanged.
"""
from __future__ import annotations

import pytest

from runtime.skills.intent_router import IntentRouter
from runtime.skills.registry import SkillDescriptor, SkillRegistry

pytestmark = pytest.mark.unit


def _desc(
    skill_id: str,
    *,
    intents: list[str],
    tool: str | None = None,
) -> SkillDescriptor:
    return SkillDescriptor(
        id=skill_id,
        description=f"test descriptor for {skill_id}",
        intents=intents,
        tool=tool if tool is not None else skill_id,
    )


def _router(*descriptors: SkillDescriptor) -> IntentRouter:
    return IntentRouter(SkillRegistry(list(descriptors)))


def test_empty_registry_never_matches() -> None:
    router = _router()
    assert router.match("morning brief please") is None


def test_empty_text_returns_none() -> None:
    router = _router(_desc("morning_brief", intents=["morning_brief"]))
    assert router.match("") is None
    assert router.match("   ") is None


def test_match_on_normalized_intent_phrase() -> None:
    router = _router(_desc("morning_brief", intents=["morning_brief"]))
    # Underscores in the intent become spaces; both sides lowercased.
    hit = router.match("can you send me a Morning Brief")
    assert hit is not None
    assert hit.id == "morning_brief"


def test_match_respects_word_boundaries() -> None:
    # "brief" must not match "briefing" / "debriefed" — the normalizer
    # collapses punctuation but we still require whole-word containment.
    router = _router(_desc("brief", intents=["brief"]))
    assert router.match("briefing the team") is None
    assert router.match("please brief me") is not None


def test_match_handles_punctuation_and_extra_whitespace() -> None:
    router = _router(_desc("morning_brief", intents=["morning_brief"]))
    hit = router.match("  please,   send the  morning-brief!!! ")
    assert hit is not None
    assert hit.id == "morning_brief"


def test_longer_intent_wins_over_shorter() -> None:
    # Two skills: one with a short intent, one with a specific compound.
    # The compound should win when both would match.
    short = _desc("briefer", intents=["brief"])
    specific = _desc("morning_brief", intents=["morning_brief"])
    router = _router(short, specific)
    hit = router.match("send me the morning brief now")
    assert hit is not None
    assert hit.id == "morning_brief"


def test_miss_when_no_intent_phrase_present() -> None:
    router = _router(_desc("morning_brief", intents=["morning_brief"]))
    assert router.match("what's the weather like today?") is None


def test_multiple_intents_per_skill_all_match() -> None:
    router = _router(
        _desc("morning_brief", intents=["morning_brief", "daily_brief"])
    )
    assert router.match("give me the daily brief").id == "morning_brief"  # type: ignore[union-attr]
    assert router.match("give me the morning brief").id == "morning_brief"  # type: ignore[union-attr]


def test_first_registered_descriptor_wins_on_tie() -> None:
    # Same-length intents registered in two descriptors: first one wins
    # because SkillRegistry.for_intent uses setdefault (first-writer wins)
    # — the router preserves that precedence.
    first = _desc("first_skill", intents=["hello"])
    second = _desc("second_skill", intents=["hello"])
    router = _router(first, second)
    hit = router.match("hello there")
    assert hit is not None
    assert hit.id == "first_skill"
