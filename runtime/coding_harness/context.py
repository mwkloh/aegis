"""Read-only skill-aware context gatherer for the coding harness.

Used by ``--with-context`` mode (Phase 5 Track B). For a given task,
read the in-scope source files plus any skill YAML whose id appears
in a scope path, capped per-file (4 KB) and per-bundle (15 KB).

Read-only. Refuses paths that escape the repo root via ``..``. Never
raises on missing files or unreadable bytes — the gatherer's failure
mode is "smaller bundle", never "exception".
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PER_FILE_BYTES: Final[int] = 4096
DEFAULT_TOTAL_BYTES: Final[int] = 15360  # 15 KB — Decision #5 (Phase 5 sign-off)
_SKILLS_RELDIR: Final[str] = "runtime/skills/_bundle"
_PATH_MAX: Final[int] = 512
_MARKER_OVERHEAD: Final[int] = 100  # generous bound on truncation marker bytes
_MIN_USEFUL_BYTES: Final[int] = 200  # below this, skip the file entirely


class FileSlice(BaseModel):
    """One in-scope source file. ``content`` is ≤ per-file cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=_PATH_MAX)
    content: str
    bytes_total: int = Field(ge=0)
    was_truncated: bool


class SkillSlice(BaseModel):
    """One skill YAML matched to the scope. Same shape as ``FileSlice``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=_PATH_MAX)
    content: str
    bytes_total: int = Field(ge=0)
    was_truncated: bool


class ContextBundle(BaseModel):
    """All context the coder is allowed to see for one CT.

    ``truncated`` is sticky: ``True`` if any slice was head-truncated
    or if the total budget capped further entries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: list[FileSlice] = Field(default_factory=list)
    skills: list[SkillSlice] = Field(default_factory=list)
    truncated: bool
    total_bytes: int = Field(ge=0)


def _safe_resolve(repo_root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``repo_root``. ``None`` on escape or error."""
    rr = repo_root.resolve()
    try:
        candidate = (rr / rel).resolve()
    except OSError:
        return None
    try:
        candidate.relative_to(rr)
    except ValueError:
        return None
    return candidate


def _read_capped(path: Path, max_bytes: int) -> tuple[str, int, bool]:
    """Read ``path`` (head-truncated to ``max_bytes``).

    Returns ``(text, bytes_total, was_truncated)``. The returned
    text's UTF-8 byte length is guaranteed ``≤ max_bytes`` — the
    truncation marker is accounted for in the head budget so the
    final slice never overflows the caller's allowance.
    """
    raw = path.read_bytes()
    total = len(raw)
    if total <= max_bytes:
        return raw.decode("utf-8", errors="replace"), total, False
    head_budget = max(0, max_bytes - _MARKER_OVERHEAD)
    head = raw[:head_budget].decode("utf-8", errors="replace")
    marker = f"\n\n[truncated — file was {total} bytes, kept {head_budget}]\n"
    text = head + marker
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:  # belt-and-braces: hard-clamp the marker too
        text = encoded[:max_bytes].decode("utf-8", errors="replace")
    return text, total, True


def _matching_skill_yaml_paths(
    repo_root: Path, scope_paths: list[str]
) -> list[Path]:
    """Skill YAMLs whose ``id`` is mentioned by any scope path.

    Two kinds of match:
      * Direct: ``scope_paths`` lists the YAML path verbatim.
      * Substring: the skill ``id`` appears in any scope path's
        basename (catches ``runtime/tools/echo_tool.py`` for
        ``echo.yaml``).
    """
    catalog = repo_root / _SKILLS_RELDIR
    if not catalog.is_dir():
        return []
    scope_set = set(scope_paths)
    scope_basenames = [Path(p).name for p in scope_paths]
    matches: list[Path] = []
    for yaml_path in sorted(catalog.glob("*/skill.yaml")):
        rel = str(yaml_path.relative_to(repo_root))
        if rel in scope_set:
            matches.append(yaml_path)
            continue
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        skill_id = str(data.get("id") or "").strip()
        if not skill_id:
            continue
        if any(skill_id in base for base in scope_basenames):
            matches.append(yaml_path)
    return matches


def gather_context(
    repo_root: Path,
    scope_paths: list[str],
    *,
    max_total_bytes: int = DEFAULT_TOTAL_BYTES,
    max_per_file_bytes: int = DEFAULT_PER_FILE_BYTES,
) -> ContextBundle:
    """Build a ``ContextBundle`` for the given task scope.

    Order: in-scope files first (in ``scope_paths`` order), then any
    matching skill YAMLs (alphabetical). Each entry obeys the
    per-file cap; the running total obeys the per-bundle cap.
    """
    rr = repo_root.resolve()
    files: list[FileSlice] = []
    skills: list[SkillSlice] = []
    used = 0
    truncated_overall = False
    seen_paths: set[str] = set()

    for rel in scope_paths:
        candidate = _safe_resolve(rr, rel)
        if candidate is None:
            truncated_overall = True
            continue
        if not candidate.is_file():
            continue
        budget = min(max_per_file_bytes, max_total_bytes - used)
        if budget < _MIN_USEFUL_BYTES:
            truncated_overall = True
            continue
        try:
            text, total, was_trunc = _read_capped(candidate, budget)
        except OSError:
            continue
        files.append(
            FileSlice(
                path=rel, content=text,
                bytes_total=total, was_truncated=was_trunc,
            )
        )
        used += len(text.encode("utf-8"))
        seen_paths.add(rel)
        if was_trunc:
            truncated_overall = True

    for yaml_path in _matching_skill_yaml_paths(rr, scope_paths):
        rel = str(yaml_path.relative_to(rr))
        if rel in seen_paths:  # already pulled in as a regular file
            continue
        budget = min(max_per_file_bytes, max_total_bytes - used)
        if budget < _MIN_USEFUL_BYTES:
            truncated_overall = True
            continue
        try:
            text, total, was_trunc = _read_capped(yaml_path, budget)
        except OSError:
            continue
        skills.append(
            SkillSlice(
                path=rel, content=text,
                bytes_total=total, was_truncated=was_trunc,
            )
        )
        used += len(text.encode("utf-8"))
        seen_paths.add(rel)
        if was_trunc:
            truncated_overall = True

    return ContextBundle(
        files=files, skills=skills,
        truncated=truncated_overall, total_bytes=used,
    )
