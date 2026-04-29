"""Skill runner — turns (descriptor, user_text) into a ToolIntent.

For Tier 0 skills (`requires_tier1=False`) the runner builds the contract from
the descriptor + raw text deterministically. For Tier 1 skills it asks an
optional `Tier1Reasoner`. When no reasoner is supplied (no key configured) the
runner returns a graceful "tier1 unavailable" `respond` contract so the
pipeline never raises.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from runtime.harness import ToolIntent
from runtime.skills import SkillDescriptor

from .tier1_reasoner import Tier1Reasoner, Tier1ReasonerError

_ECHO_STRIP = re.compile(r"^\s*(?:echo|ping)\s*", re.IGNORECASE)


class SkillRunner:
    """Build a ToolIntent for the matched skill. No side effects."""

    def __init__(self, tier1: Tier1Reasoner | None = None) -> None:
        self._tier1 = tier1

    async def build(
        self,
        descriptor: SkillDescriptor,
        user_text: str,
        *,
        recent: Sequence[tuple[str, str]] = (),
    ) -> ToolIntent:
        if descriptor.requires_tier1:
            return await self._build_tier1(descriptor, user_text, recent=recent)
        return ToolIntent(
            tool=descriptor.tool,
            args=self._args_for(descriptor, user_text),
            skill_id=descriptor.id,
            rationale="tier 0 deterministic mapping from intent to tool",
        )

    async def _build_tier1(
        self,
        descriptor: SkillDescriptor,
        user_text: str,
        *,
        recent: Sequence[tuple[str, str]] = (),
    ) -> ToolIntent:
        if self._tier1 is None:
            return _tier1_unavailable(descriptor, "no OPENROUTER_API_KEY configured")
        try:
            return await self._tier1.reason(descriptor, user_text, recent=recent)
        except Tier1ReasonerError as exc:
            return _tier1_unavailable(descriptor, f"reasoner failed: {exc}")

    @staticmethod
    def _args_for(descriptor: SkillDescriptor, user_text: str) -> dict[str, object]:
        # Tier 0 only knows how to populate args for built-in skills.
        if descriptor.id == "echo":
            stripped = _ECHO_STRIP.sub("", user_text).strip()
            return {"message": stripped or "pong"}
        if descriptor.id == "ask_time":
            return {"query": user_text.strip()}
        return {}


def _tier1_unavailable(descriptor: SkillDescriptor, reason: str) -> ToolIntent:
    """Build a graceful 'cannot answer' contract that uses the respond tool."""
    return ToolIntent(
        tool="respond",
        args={"message": f"tier1 unavailable: skill {descriptor.id!r} requires it ({reason})"},
        skill_id=descriptor.id,
        rationale="tier1 degraded — respond with explanatory message",
    )
