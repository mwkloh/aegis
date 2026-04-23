"""SkillRegistry.from_directory — directory-per-skill layout + source_dir tracking."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.skills.registry import SkillRegistry


def _write_skill_yaml(path: Path, skill_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
id: {skill_id}
version: 0.1.0
description: Test skill {skill_id}.
intents: [{skill_id}]
tool: t
args_schema:
  type: object
  additionalProperties: false
requires_tier1: false
""".strip()
    )


def test_loads_directory_per_skill(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "alpha" / "skill.yaml", "alpha")
    _write_skill_yaml(tmp_path / "beta" / "skill.yaml", "beta")

    registry = SkillRegistry.from_directory(tmp_path)

    assert {d.id for d in registry.all()} == {"alpha", "beta"}


def test_source_dir_of_returns_skill_directory(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "alpha" / "skill.yaml", "alpha")

    registry = SkillRegistry.from_directory(tmp_path)

    assert registry.source_dir_of("alpha") == tmp_path / "alpha"
    assert registry.source_dir_of("missing") is None


def test_still_loads_flat_layout_for_back_compat(tmp_path: Path) -> None:
    # Legacy: <catalog>/<id>.yaml with no subdirectory.
    flat = tmp_path / "gamma.yaml"
    _write_skill_yaml(flat, "gamma")

    registry = SkillRegistry.from_directory(tmp_path)

    assert {d.id for d in registry.all()} == {"gamma"}
    # Flat-layout source_dir is the catalog dir itself (no per-skill home).
    assert registry.source_dir_of("gamma") == tmp_path


def test_directory_wins_on_duplicate_id(tmp_path: Path) -> None:
    _write_skill_yaml(tmp_path / "dup" / "skill.yaml", "dup")
    _write_skill_yaml(tmp_path / "dup.yaml", "dup")

    registry = SkillRegistry.from_directory(tmp_path)

    # Directory layout is the authoritative form; flat legacy file is ignored.
    assert registry.source_dir_of("dup") == tmp_path / "dup"


def test_missing_catalog_returns_empty_registry(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    registry = SkillRegistry.from_directory(missing)
    assert registry.all() == []
