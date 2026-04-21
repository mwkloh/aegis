"""Catalog-side sanity for the SYS-* recurring jobs (Track D, D3).

Each scheduled system job (morning_brief, vault_reindex, tier2_compress,
reflection_sweep) ships a YAML in ``runtime/skills/catalog/`` declaring its
argv template. These tests assert:

* The shipped catalog loads without errors (YAML valid, placeholders resolve).
* The SYS-* ``skill`` field on ``SystemJobSpec`` matches a catalog ``id``.
* The argv template for each job uses the ``python -m runtime.skills.scripts.*``
  convention so subprocesses hit the same interpreter as the bot (via
  ``build_skill_arg_resolver``).

Tests are tolerant of skills that haven't landed yet (D3c, D3d) — they skip
rather than fail so partial progress doesn't red-light CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.scheduler.seed import SYSTEM_JOBS
from runtime.skills import SkillRegistry

pytestmark = pytest.mark.unit


CATALOG = Path(__file__).parent.parent / "runtime" / "skills" / "catalog"


def test_catalog_directory_loads() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    # Every skill id is unique and validates via model schema.
    assert registry.get("morning_brief") is not None


def test_vault_reindex_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    desc = registry.get("vault_reindex")
    assert desc is not None, "vault_reindex.yaml missing from catalog"
    assert desc.tool == "vault_reindex"
    assert len(desc.tools) == 1
    tool = desc.tools[0]
    assert tool.argv_template[:3] == ["python", "-m", "runtime.skills.scripts.vault_reindex"]
    # No placeholders — scheduler never has to substitute config for this skill.
    assert tool.placeholders() == set()
    # Long timeout — bge-m3 embedding a vault with thousands of notes is slow.
    assert tool.timeout_ms >= 60_000
    # Reindex is fully local; no network required.
    assert tool.allow_net is False


def test_tier2_compress_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    desc = registry.get("tier2_compress")
    assert desc is not None, "tier2_compress.yaml missing from catalog"
    assert desc.tool == "tier2_compress"
    assert len(desc.tools) == 1
    tool = desc.tools[0]
    assert tool.argv_template[:3] == [
        "python",
        "-m",
        "runtime.skills.scripts.tier2_compress",
    ]
    assert tool.placeholders() == set()
    assert tool.allow_net is False


def test_reflection_sweep_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    desc = registry.get("reflection_sweep")
    assert desc is not None, "reflection_sweep.yaml missing from catalog"
    assert desc.tool == "reflection_sweep"
    assert len(desc.tools) == 1
    tool = desc.tools[0]
    # Unlike the other scheduled jobs, reflection reuses its existing
    # library CLI entry point rather than a dedicated script under
    # runtime/skills/scripts/ — avoid duplicating the argparse surface.
    assert tool.argv_template[:3] == [
        "python",
        "-m",
        "runtime.reflection.cli",
    ]
    # The --quiet flag is what makes stdout empty on success (silent-success
    # contract). Without it the scheduler's push layer would forward progress
    # lines as a bogus "result" message on every run.
    assert "--quiet" in tool.argv_template
    assert tool.placeholders() == set()
    assert tool.allow_net is False


def test_echo_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    desc = registry.get("echo")
    assert desc is not None, "echo.yaml missing from catalog"
    assert desc.tool == "echo"
    assert len(desc.tools) == 1
    tool = desc.tools[0]
    assert tool.argv_template[:3] == ["python", "-m", "runtime.skills.scripts.echo"]
    # No placeholders — scheduler appends positional args from the job's args tuple.
    assert tool.placeholders() == set()
    assert tool.allow_net is False


def test_system_job_skills_all_resolvable() -> None:
    """Every SYSTEM_JOBS.skill loads from the catalog (D3a/b/c/d all done).
    Catches typos in ``seed.SYSTEM_JOBS``.
    """
    registry = SkillRegistry.from_directory(CATALOG)
    for spec in SYSTEM_JOBS:
        desc = registry.get(spec.skill)
        assert desc is not None, (
            f"SYS job {spec.id!r} references unknown skill {spec.skill!r}"
        )
        assert desc.tool == spec.skill
