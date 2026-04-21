from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.skills import SkillDescriptor, SkillRegistry, ToolSpec

pytestmark = pytest.mark.unit


CATALOG = Path(__file__).parent.parent / "runtime" / "skills" / "catalog"


def test_loads_echo_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    echo = registry.get("echo")
    assert echo is not None
    assert echo.tool == "echo"
    assert "echo" in echo.intents
    assert "ping" in echo.intents
    assert echo.requires_tier1 is False
    assert len(echo.tools) == 1
    assert echo.tools[0].argv_template[:3] == ["python", "-m", "runtime.skills.scripts.echo"]


def test_for_intent_resolves_to_descriptor() -> None:
    registry = SkillRegistry.from_directory(CATALOG)
    assert registry.for_intent("echo") is not None
    assert registry.for_intent("nope") is None


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "id: bad\nversion: 0.0.1\ndescription: x\nintents: [x]\ntool: x\nunknown: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        SkillRegistry.from_directory(tmp_path)


# --- ToolSpec + SkillDescriptor.tools (Phase 8 C1) ---------------------------


def _base_descriptor(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "vault_search",
        "version": "0.1.0",
        "description": "Search the operator's Obsidian vault.",
        "intents": ["vault_search"],
        "tool": "vault_search",
        "args_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    }
    data.update(overrides)
    return data


def test_tools_defaults_to_empty_list() -> None:
    desc = SkillDescriptor(**_base_descriptor())  # type: ignore[arg-type]
    assert desc.tools == []


def test_toolspec_valid_minimal() -> None:
    spec = ToolSpec(name="search", argv_template=["aegis", "vault", "search"])
    assert spec.timeout_ms == 30_000
    assert spec.allow_net is False
    assert spec.schema_ is None
    assert spec.placeholders() == set()


def test_toolspec_placeholders_extracted_from_tokens() -> None:
    spec = ToolSpec(
        name="search",
        argv_template=[
            "aegis",
            "vault",
            "search",
            "--query={query}",
            "--limit",
            "{limit}",
        ],
    )
    assert spec.placeholders() == {"query", "limit"}


def test_toolspec_rejects_empty_argv() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="x", argv_template=[])


def test_toolspec_rejects_empty_token() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="x", argv_template=["aegis", ""])


def test_toolspec_timeout_bounds() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="x", argv_template=["aegis"], timeout_ms=50)  # below floor
    with pytest.raises(ValidationError):
        ToolSpec(name="x", argv_template=["aegis"], timeout_ms=600_001)  # above ceil


def test_toolspec_name_pattern_enforced() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="BadName", argv_template=["aegis"])


def test_toolspec_schema_aliased_from_schema_field() -> None:
    # YAML uses `schema:` but Pydantic stores it under `schema_` to avoid
    # clashing with BaseModel.schema().
    spec = ToolSpec.model_validate(
        {
            "name": "search",
            "argv_template": ["aegis"],
            "schema": {"type": "object"},
        }
    )
    assert spec.schema_ == {"type": "object"}


def test_descriptor_tool_placeholder_must_resolve() -> None:
    with pytest.raises(ValidationError, match="unknown_arg"):
        SkillDescriptor(
            **_base_descriptor(  # type: ignore[arg-type]
                tools=[
                    {
                        "name": "search",
                        "argv_template": ["aegis", "--bad", "{unknown_arg}"],
                    }
                ],
            ),
        )


def test_descriptor_tool_placeholder_resolves_against_args_schema() -> None:
    desc = SkillDescriptor(
        **_base_descriptor(  # type: ignore[arg-type]
            tools=[
                {
                    "name": "search",
                    "argv_template": [
                        "aegis",
                        "vault",
                        "search",
                        "--query",
                        "{query}",
                        "--limit",
                        "{limit}",
                    ],
                    "timeout_ms": 15_000,
                    "allow_net": False,
                }
            ],
        ),
    )
    assert len(desc.tools) == 1
    assert desc.tools[0].timeout_ms == 15_000
    assert desc.tools[0].allow_net is False


def test_descriptor_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValidationError, match="duplicate tool name"):
        SkillDescriptor(
            **_base_descriptor(  # type: ignore[arg-type]
                tools=[
                    {"name": "search", "argv_template": ["aegis"]},
                    {"name": "search", "argv_template": ["aegis", "search"]},
                ],
            ),
        )


def test_descriptor_with_no_args_schema_rejects_any_placeholder() -> None:
    data = _base_descriptor(args_schema={})
    data["tools"] = [
        {"name": "x", "argv_template": ["aegis", "{query}"]}
    ]
    with pytest.raises(ValidationError, match="query"):
        SkillDescriptor(**data)  # type: ignore[arg-type]


def test_loads_descriptor_with_tools_from_yaml(tmp_path: Path) -> None:
    yaml_text = """
id: vault_search
version: 0.1.0
description: Search the operator's Obsidian vault.
intents:
  - vault_search
tool: vault_search
args_schema:
  type: object
  properties:
    query:
      type: string
tools:
  - name: search
    argv_template:
      - aegis
      - vault
      - search
      - "{query}"
    timeout_ms: 5000
    allow_net: false
    schema:
      type: object
      properties:
        matches:
          type: array
"""
    (tmp_path / "vault_search.yaml").write_text(yaml_text, encoding="utf-8")
    registry = SkillRegistry.from_directory(tmp_path)
    desc = registry.get("vault_search")
    assert desc is not None
    assert len(desc.tools) == 1
    tool = desc.tools[0]
    assert tool.name == "search"
    assert tool.argv_template[-1] == "{query}"
    assert tool.timeout_ms == 5000
    assert tool.allow_net is False
    assert tool.schema_ is not None
    assert tool.schema_["type"] == "object"


def test_descriptor_field_reassignment_blocked() -> None:
    desc = SkillDescriptor(**_base_descriptor())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        desc.tools = [ToolSpec(name="x", argv_template=["aegis"])]
