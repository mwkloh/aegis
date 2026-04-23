"""Skill registry — loads YAML descriptors with `safe_load` only.

Descriptors are declarative scaffolds (not subagents, not tools). A descriptor
binds an intent label to a tool the harness can run, plus optional reasoning
hints for Tier 1.

Phase 8 C1 adds an optional ``tools`` list — CLI subprocess tools the skill
can invoke via :mod:`runtime.tools.harness`. Each entry declares an
``argv_template`` (list of tokens with ``{arg}`` placeholders), an optional
stdout ``schema`` for verdict validation, a bounded ``timeout_ms``, and an
``allow_net`` flag that the harness consumes. The descriptor is still
purely declarative — nothing here executes a subprocess; it only validates
shape and that placeholders resolve against the skill's ``args_schema``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_MIN_TIMEOUT_MS = 100
_MAX_TIMEOUT_MS = 600_000  # 10 minutes — matches `Bash` tool ceiling
_MAX_TOOLS_PER_SKILL = 16
_MAX_ARGV_TOKENS = 32
_MAX_ARGV_TOKEN_CHARS = 256


class ToolSpec(BaseModel):
    """One CLI subprocess tool a skill may invoke through the tool harness.

    ``argv_template`` is a list (never a string) — execution is always argv
    arrays passed to ``asyncio.create_subprocess_exec``. No shell, ever.
    Each token may contain ``{arg}`` placeholders which resolve against the
    parent skill's ``args_schema.properties`` at harness-call time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    argv_template: list[str] = Field(min_length=1, max_length=_MAX_ARGV_TOKENS)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    timeout_ms: int = Field(default=30_000, ge=_MIN_TIMEOUT_MS, le=_MAX_TIMEOUT_MS)
    allow_net: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_argv_tokens(self) -> ToolSpec:
        for idx, token in enumerate(self.argv_template):
            if not isinstance(token, str):
                raise ValueError(
                    f"argv_template[{idx}] must be a string, got {type(token).__name__}"
                )
            if not token:
                raise ValueError(f"argv_template[{idx}] must be non-empty")
            if len(token) > _MAX_ARGV_TOKEN_CHARS:
                raise ValueError(
                    f"argv_template[{idx}] exceeds {_MAX_ARGV_TOKEN_CHARS} chars"
                )
        return self

    def placeholders(self) -> set[str]:
        """Return the set of ``{name}`` placeholders referenced in argv_template."""
        found: set[str] = set()
        for token in self.argv_template:
            found.update(_PLACEHOLDER_RE.findall(token))
        return found


class SkillDescriptor(BaseModel):
    """Validated skill descriptor — every field is bounded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    version: str = Field(default="0.1.0")
    description: str = Field(min_length=1, max_length=512)
    intents: list[str] = Field(default_factory=list)
    tool: str = Field(min_length=1, max_length=64)
    args_schema: dict[str, Any] = Field(default_factory=dict)
    requires_tier1: bool = Field(default=False)
    tools: list[ToolSpec] = Field(
        default_factory=list,
        max_length=_MAX_TOOLS_PER_SKILL,
    )

    @model_validator(mode="after")
    def _validate_tool_placeholders(self) -> SkillDescriptor:
        if not self.tools:
            return self
        props = self.args_schema.get("properties") if isinstance(self.args_schema, dict) else None
        allowed: set[str] = {str(k) for k in props} if isinstance(props, dict) else set()

        names_seen: set[str] = set()
        for spec in self.tools:
            if spec.name in names_seen:
                raise ValueError(f"duplicate tool name: {spec.name!r}")
            names_seen.add(spec.name)
            missing = spec.placeholders() - allowed
            if missing:
                keys = ", ".join(sorted(missing))
                raise ValueError(
                    f"tool {spec.name!r}: placeholders {{{keys}}} not in args_schema.properties"
                )
        return self


class SkillRegistry:
    """In-memory index. Build at startup, never mutate at runtime."""

    def __init__(
        self,
        descriptors: list[SkillDescriptor],
        *,
        source_dirs: dict[str, Path] | None = None,
    ) -> None:
        self._by_id: dict[str, SkillDescriptor] = {d.id: d for d in descriptors}
        self._by_intent: dict[str, SkillDescriptor] = {}
        for d in descriptors:
            for intent in d.intents:
                # Deterministic: first descriptor to claim an intent wins.
                self._by_intent.setdefault(intent, d)
        self._source_dirs: dict[str, Path] = dict(source_dirs or {})

    @classmethod
    def from_directory(cls, catalog_dir: Path) -> SkillRegistry:
        """Load every skill under ``catalog_dir``.

        Recognised layouts, in priority order:

        1. **Directory-per-skill** — ``<catalog_dir>/<id>/skill.yaml``.
           This is the preferred sovereign layout. ``source_dir_of(id)``
           returns the per-skill directory so ``{skill_dir}`` placeholders
           in ``argv_template`` resolve to a co-located script.
        2. **Flat** — ``<catalog_dir>/<id>.yaml``. Legacy in-repo layout.
           Kept for back-compat during migration; ``source_dir_of`` returns
           the catalog directory itself.

        When both forms declare the same id, the directory form wins.
        """
        catalog_dir = Path(catalog_dir)
        if not catalog_dir.is_dir():
            return cls([])
        descriptors: list[SkillDescriptor] = []
        source_dirs: dict[str, Path] = {}
        seen: set[str] = set()

        for entry in sorted(catalog_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_yaml = entry / "skill.yaml"
            if not skill_yaml.is_file():
                continue
            descriptor = cls._load_one(skill_yaml)
            if descriptor.id in seen:
                continue
            descriptors.append(descriptor)
            source_dirs[descriptor.id] = entry
            seen.add(descriptor.id)

        for path in sorted(catalog_dir.glob("*.yaml")):
            descriptor = cls._load_one(path)
            if descriptor.id in seen:
                continue
            descriptors.append(descriptor)
            source_dirs[descriptor.id] = catalog_dir
            seen.add(descriptor.id)

        return cls(descriptors, source_dirs=source_dirs)

    @staticmethod
    def _load_one(path: Path) -> SkillDescriptor:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: skill descriptor must be a YAML mapping")
        return SkillDescriptor(**raw)

    def get(self, skill_id: str) -> SkillDescriptor | None:
        return self._by_id.get(skill_id)

    def for_intent(self, intent: str) -> SkillDescriptor | None:
        return self._by_intent.get(intent)

    def all(self) -> list[SkillDescriptor]:
        return list(self._by_id.values())

    def source_dir_of(self, skill_id: str) -> Path | None:
        """Directory that holds the descriptor — or ``None`` if unknown."""
        return self._source_dirs.get(skill_id)
