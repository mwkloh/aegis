"""Skill registry. Read-only at runtime — descriptors live in `catalog/`."""

from .loader import SkillLoader
from .registry import SkillDescriptor, SkillRegistry, ToolSpec

__all__ = ["SkillDescriptor", "SkillLoader", "SkillRegistry", "ToolSpec"]
