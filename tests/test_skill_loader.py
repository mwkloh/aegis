from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from runtime.skills import SkillLoader

pytestmark = pytest.mark.unit


def _write_skill(
    catalog: Path,
    filename: str,
    *,
    skill_id: str,
    intents: list[str],
    tool: str = "respond",
    description: str = "Test skill.",
    extra: dict[str, object] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "id": skill_id,
        "version": "0.1.0",
        "description": description,
        "intents": intents,
        "tool": tool,
        "args_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    }
    if extra:
        payload.update(extra)
    path = catalog / filename
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


# --- basic lookups -----------------------------------------------------------


def test_for_intent_returns_none_when_catalog_missing(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path / "does_not_exist")
    assert loader.for_intent("echo") is None
    assert loader.get("echo") is None


def test_for_intent_finds_registered_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo", "ping"])
    loader = SkillLoader(tmp_path)
    desc = loader.for_intent("echo")
    assert desc is not None
    assert desc.id == "echo"
    assert loader.for_intent("ping") is desc  # same cached object
    assert loader.for_intent("unknown") is None


def test_get_finds_registered_skill_by_id(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    loader = SkillLoader(tmp_path)
    assert loader.get("echo") is not None
    assert loader.get("missing") is None


# --- per-file parse cache ----------------------------------------------------


def test_parse_cache_returns_same_object_until_mtime_changes(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    loader = SkillLoader(tmp_path)
    first = loader.get("echo")
    second = loader.get("echo")
    assert first is second  # cache hit → identical object


def test_parse_cache_invalidates_on_mtime_change(tmp_path: Path) -> None:
    path = _write_skill(
        tmp_path, "echo.yaml", skill_id="echo", intents=["echo"],
        description="first",
    )
    loader = SkillLoader(tmp_path)
    first = loader.get("echo")
    assert first is not None
    assert first.description == "first"

    # Bump the file's mtime by a full second so cache invalidation fires even
    # on filesystems with second-granularity mtime.
    time.sleep(1.1)
    _write_skill(
        tmp_path, "echo.yaml", skill_id="echo", intents=["echo"],
        description="second",
    )
    # Belt-and-braces: force mtime forward explicitly.
    future = path.stat().st_mtime + 2
    os.utime(path, (future, future))

    second = loader.get("echo")
    assert second is not None
    assert second.description == "second"
    assert second is not first


# --- directory index invalidation --------------------------------------------


def test_new_file_picked_up_after_dir_mtime_changes(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    loader = SkillLoader(tmp_path)
    assert loader.for_intent("echo") is not None
    assert loader.for_intent("time") is None

    # Force directory mtime forward so _ensure_index rebuilds on next access.
    _write_skill(tmp_path, "time.yaml", skill_id="time", intents=["time"])
    future = tmp_path.stat().st_mtime + 2
    os.utime(tmp_path, (future, future))

    assert loader.for_intent("time") is not None


def test_refresh_forces_index_rebuild(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    loader = SkillLoader(tmp_path)
    assert loader.for_intent("echo") is not None

    # Add a file but simulate a tool that somehow preserves dir mtime
    # (e.g., atomic rename in the same second). refresh() is the explicit
    # escape hatch operators can rely on.
    _write_skill(tmp_path, "extra.yaml", skill_id="extra", intents=["extra"])
    loader.refresh()
    assert loader.for_intent("extra") is not None


# --- determinism & error tolerance -------------------------------------------


def test_duplicate_intent_first_sorted_filename_wins(tmp_path: Path) -> None:
    _write_skill(tmp_path, "a_echo.yaml", skill_id="echo_a", intents=["echo"])
    _write_skill(tmp_path, "b_echo.yaml", skill_id="echo_b", intents=["echo"])
    loader = SkillLoader(tmp_path)
    desc = loader.for_intent("echo")
    assert desc is not None
    assert desc.id == "echo_a"


def test_malformed_yaml_skipped_in_index(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("not: [valid", encoding="utf-8")
    _write_skill(tmp_path, "ok.yaml", skill_id="ok", intents=["ok"])
    loader = SkillLoader(tmp_path)
    assert loader.for_intent("ok") is not None
    # Broken file never made it into either index, so nothing resolves to it.
    assert loader.get("broken") is None


def test_full_parse_failure_raises_only_at_access(tmp_path: Path) -> None:
    # Header looks fine (id + intents valid strings) but args_schema is missing
    # required pydantic constraint satisfaction — e.g., unknown top-level key.
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": "bad",
                "version": "0.0.1",
                "description": "x",
                "intents": ["bad"],
                "tool": "x",
                "unknown_extra_key": 1,
            }
        ),
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path)
    with pytest.raises(ValidationError):
        loader.get("bad")


def test_missing_file_between_index_and_parse_is_graceful(tmp_path: Path) -> None:
    path = _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    loader = SkillLoader(tmp_path)
    # Index built; now delete the file before parsing.
    path.unlink()
    assert loader.get("echo") is None


# --- progressive disclosure --------------------------------------------------


def test_render_for_prompt_excludes_noise_fields(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "echo.yaml",
        skill_id="echo",
        intents=["echo", "ping"],
        extra={"requires_tier1": True},
    )
    loader = SkillLoader(tmp_path)
    desc = loader.get("echo")
    assert desc is not None
    rendered = SkillLoader.render_for_prompt(desc)
    parsed = yaml.safe_load(rendered)

    assert parsed["id"] == "echo"
    assert parsed["description"]
    assert parsed["tool"]
    assert "args_schema" in parsed
    # Noise fields filtered out:
    assert "version" not in parsed
    assert "intents" not in parsed
    assert "requires_tier1" not in parsed
    # No tools declared → no `tools` key either
    assert "tools" not in parsed


def test_render_for_prompt_includes_tools_when_declared(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "vault_search.yaml",
        skill_id="vault_search",
        intents=["vault_search"],
        tool="vault_search",
        extra={
            "tools": [
                {
                    "name": "search",
                    "argv_template": ["aegis", "vault", "search", "{message}"],
                    "timeout_ms": 5000,
                    "allow_net": False,
                    "schema": {"type": "object"},
                }
            ]
        },
    )
    loader = SkillLoader(tmp_path)
    desc = loader.get("vault_search")
    assert desc is not None
    rendered = SkillLoader.render_for_prompt(desc)
    parsed = yaml.safe_load(rendered)

    assert parsed["tools"][0]["name"] == "search"
    assert parsed["tools"][0]["argv_template"][-1] == "{message}"
    assert parsed["tools"][0]["timeout_ms"] == 5000
    assert parsed["tools"][0]["allow_net"] is False
    assert parsed["tools"][0]["schema"] == {"type": "object"}


def test_progressive_disclosure_only_matched_skill_rendered(tmp_path: Path) -> None:
    _write_skill(tmp_path, "echo.yaml", skill_id="echo", intents=["echo"])
    _write_skill(tmp_path, "time.yaml", skill_id="time_query", intents=["time"])
    _write_skill(
        tmp_path, "other.yaml", skill_id="other_thing", intents=["other"]
    )
    loader = SkillLoader(tmp_path)

    desc = loader.for_intent("echo")
    assert desc is not None
    rendered = SkillLoader.render_for_prompt(desc)

    # Invariant: no sibling skill ids or intents leak into the prompt blob.
    assert "echo" in rendered
    assert "time_query" not in rendered
    assert "other_thing" not in rendered
