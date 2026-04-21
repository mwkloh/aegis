"""Progressive-disclosure skill loader.

Where :class:`runtime.skills.SkillRegistry` eagerly parses every YAML in
the catalog at startup, :class:`SkillLoader` parses a skill only when
first requested and caches the parsed descriptor per-file with mtime
invalidation. The intent→path and id→path indexes rebuild when the
catalog directory's mtime changes (file add/remove).

Why progressive disclosure: Phase 8 §2.1 requires that a chat turn's
prompt payload contain ONLY the matched skill's descriptor + its tool
list, never the full catalog. Passing full catalogs to the LLM bloats
context and leaks unrelated capabilities into the reasoning trace.

Use :meth:`SkillLoader.render_for_prompt` to turn a resolved descriptor
into the compact YAML blob the reasoner gets pasted into its system
prompt. The blob deliberately excludes fields that don't affect
reasoning (``version``, ``intents``, ``requires_tier1``) so the prompt
surface matches "what the LLM needs to produce valid args."
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .registry import SkillDescriptor


class SkillLoader:
    """Lazy, cached skill loader.

    Cache policy:
      * **Parse cache** — per-file ``(mtime, SkillDescriptor)``. A file
        edit bumps mtime and triggers a re-parse on the next access.
      * **Index cache** — directory-level ``intent -> path`` and
        ``id -> path`` maps, invalidated when the catalog directory's
        own mtime changes (file add / remove).

    Rebuilding the index does NOT re-parse every YAML; it only reads
    the header fields (``id``, ``intents``) to fill the maps. Full
    parsing still happens lazily on first access per skill.
    """

    def __init__(self, catalog_dir: Path) -> None:
        self._catalog_dir = Path(catalog_dir)
        self._descriptor_cache: dict[Path, tuple[float, SkillDescriptor]] = {}
        self._intent_index: dict[str, Path] = {}
        self._id_index: dict[str, Path] = {}
        self._index_mtime: float | None = None

    # --- public API ----------------------------------------------------------

    def get(self, skill_id: str) -> SkillDescriptor | None:
        """Return the descriptor for ``skill_id``, or ``None`` if unknown."""
        self._ensure_index()
        path = self._id_index.get(skill_id)
        if path is None:
            return None
        return self._load_path(path)

    def for_intent(self, intent: str) -> SkillDescriptor | None:
        """Return the descriptor claiming ``intent``, or ``None`` if unknown.

        When multiple descriptors claim the same intent, the first-sorted
        filename wins (deterministic across runs).
        """
        self._ensure_index()
        path = self._intent_index.get(intent)
        if path is None:
            return None
        return self._load_path(path)

    def refresh(self) -> None:
        """Force the next ``get`` / ``for_intent`` call to rebuild the index.

        Per-file parse cache is preserved; stale descriptors are caught by
        the mtime check on the next ``_load_path`` call anyway.
        """
        self._index_mtime = None

    @staticmethod
    def render_for_prompt(descriptor: SkillDescriptor) -> str:
        """Render ``descriptor`` as a compact YAML blob for LLM prompting.

        Only emits the fields a reasoning model needs to produce valid
        args — no ``version``, no ``intents``, no ``requires_tier1``.
        The YAML is ``safe_dump`` output with ``sort_keys=False`` so the
        order matches a human-authored catalog file.
        """
        payload: dict[str, Any] = {
            "id": descriptor.id,
            "description": descriptor.description,
            "tool": descriptor.tool,
            "args_schema": descriptor.args_schema,
        }
        if descriptor.tools:
            payload["tools"] = [
                _render_tool(tool_index, tool)
                for tool_index, tool in enumerate(descriptor.tools)
            ]
        return yaml.safe_dump(payload, sort_keys=False)

    # --- internals -----------------------------------------------------------

    def _dir_mtime(self) -> float | None:
        try:
            return self._catalog_dir.stat().st_mtime
        except OSError:
            return None

    def _ensure_index(self) -> None:
        current = self._dir_mtime()
        if current is None:
            # Catalog directory missing — empty indexes, don't cache a mtime
            # so a later mkdir rebuilds transparently.
            self._intent_index = {}
            self._id_index = {}
            self._index_mtime = None
            return
        if self._index_mtime == current and self._id_index:
            return
        self._rebuild_index()
        self._index_mtime = current

    def _rebuild_index(self) -> None:
        intent_index: dict[str, Path] = {}
        id_index: dict[str, Path] = {}
        for path in sorted(self._catalog_dir.glob("*.yaml")):
            header = _read_header(path)
            if header is None:
                continue
            skill_id, intents = header
            # First-sorted filename wins when ids or intents collide.
            id_index.setdefault(skill_id, path)
            for intent in intents:
                intent_index.setdefault(intent, path)
        self._intent_index = intent_index
        self._id_index = id_index

    def _load_path(self, path: Path) -> SkillDescriptor | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            # File disappeared between index and parse — drop it and retry.
            self._descriptor_cache.pop(path, None)
            self.refresh()
            return None
        cached = self._descriptor_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: skill descriptor must be a YAML mapping")
        descriptor = SkillDescriptor(**raw)
        self._descriptor_cache[path] = (mtime, descriptor)
        return descriptor


def _read_header(path: Path) -> tuple[str, list[str]] | None:
    """Extract ``(id, intents)`` for the index without full Pydantic validation.

    Returns ``None`` for anything that fails the minimal well-formedness
    check; full validation errors still surface at ``_load_path`` time
    when the descriptor is actually requested.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (yaml.YAMLError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    skill_id = raw.get("id")
    if not isinstance(skill_id, str) or not skill_id:
        return None
    intents_raw = raw.get("intents") or []
    if not isinstance(intents_raw, list):
        return None
    intents = [i for i in intents_raw if isinstance(i, str) and i]
    return skill_id, intents


def _render_tool(index: int, tool: Any) -> dict[str, Any]:
    """Render a ``ToolSpec`` as a prompt-friendly dict.

    Kept outside the class + loose-typed to avoid a cycle import on
    ``ToolSpec`` — we duck-type on attribute access.
    """
    payload: dict[str, Any] = {
        "name": getattr(tool, "name", f"tool_{index}"),
        "argv_template": list(getattr(tool, "argv_template", []) or []),
        "timeout_ms": getattr(tool, "timeout_ms", 0),
        "allow_net": bool(getattr(tool, "allow_net", False)),
    }
    schema_ = getattr(tool, "schema_", None)
    if schema_ is not None:
        payload["schema"] = schema_
    return payload
