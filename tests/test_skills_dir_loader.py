"""SkillRegistry.from_directory — directory-per-skill layout + source_dir tracking."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.chat.telegram.bot import build_skill_arg_resolver
from runtime.config import AegisConfig
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


def test_skill_dir_placeholder_is_allowed_on_descriptor(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: alpha
description: uses skill_dir
tool: t
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{skill_dir}/run.py"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    # Loading must succeed — {skill_dir} is not declared in args_schema
    # but it's an infrastructure placeholder, not a user arg.
    registry = SkillRegistry.from_directory(tmp_path)
    assert registry.get("alpha") is not None


def test_resolver_injects_skill_dir(tmp_path: Path, monkeypatch) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: alpha
description: uses skill_dir
tool: t
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{skill_dir}/run.py"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    registry = SkillRegistry.from_directory(tmp_path)
    resolver = build_skill_arg_resolver(
        AegisConfig(),
        registry=registry,
        python_executable="/usr/bin/python3",
    )
    descriptor = registry.get("alpha")
    assert descriptor is not None

    argv = resolver(descriptor)

    assert argv == ["/usr/bin/python3", f"{skill_dir}/run.py"]


def test_resolver_rejects_unknown_placeholder(tmp_path: Path) -> None:
    skill_dir = tmp_path / "beta"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        """
id: beta
description: references an undeclared placeholder
tool: t
args_schema:
  type: object
  properties:
    known:
      type: string
  required: [known]
  additionalProperties: false
tools:
  - name: t
    argv_template: [python, "{known}"]
    timeout_ms: 1000
    allow_net: false
""".strip()
    )

    registry = SkillRegistry.from_directory(tmp_path)
    resolver = build_skill_arg_resolver(AegisConfig(), registry=registry)
    descriptor = registry.get("beta")
    assert descriptor is not None

    # `known` is declared in args_schema but the resolver has no value for
    # it (not vault_root / skill_dir) → unresolvable → None.
    assert resolver(descriptor) is None
