from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from runtime.skills.installer import (
    confirm_skill,
    list_staged,
    stage_skill,
)

pytestmark = pytest.mark.unit


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _clean_descriptor(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "vault_search",
        "version": "0.1.0",
        "description": "Search the operator's vault.",
        "intents": ["vault_search"],
        "tool": "vault_search",
        "args_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        "tools": [
            {
                "name": "search",
                "argv_template": ["aegis", "vault", "search", "--query", "{query}"],
                "timeout_ms": 5_000,
                "allow_net": False,
            }
        ],
    }
    data.update(overrides)
    return data


# --- stage_skill success ------------------------------------------------------


def test_stage_skill_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "vault_search.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "staging"

    outcome = stage_skill(src, staging_dir=staging)

    assert outcome.verdict == "staged"
    assert outcome.skill_id == "vault_search"
    assert outcome.stage_path == staging / "vault_search.yaml"
    assert outcome.stage_path is not None
    assert outcome.stage_path.is_file()
    assert outcome.findings == ()
    assert outcome.is_success()


def test_stage_skill_creates_staging_dir_if_missing(tmp_path: Path) -> None:
    src = tmp_path / "x.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "nested" / "staging"
    assert not staging.exists()

    stage_skill(src, staging_dir=staging)
    assert staging.is_dir()


def test_stage_skill_preserves_original_bytes(tmp_path: Path) -> None:
    # The staging file should be byte-identical to the source so that
    # re-validation on confirm sees exactly what the operator reviewed.
    src = tmp_path / "x.yaml"
    raw_text = yaml.safe_dump(_clean_descriptor(), sort_keys=False)
    src.write_text(raw_text, encoding="utf-8")
    staging = tmp_path / "staging"

    outcome = stage_skill(src, staging_dir=staging)

    assert outcome.verdict == "staged"
    assert outcome.stage_path is not None
    assert outcome.stage_path.read_text(encoding="utf-8") == raw_text


# --- stage_skill: scan-driven rejections --------------------------------------


def test_stage_skill_blocks_on_shell_binary(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    _write_yaml(
        src,
        _clean_descriptor(
            tools=[
                {
                    "name": "shelly",
                    "argv_template": ["sh", "-c", "echo hi"],
                }
            ]
        ),
    )
    staging = tmp_path / "staging"

    outcome = stage_skill(src, staging_dir=staging)

    assert outcome.verdict == "rejected_scan"
    assert outcome.stage_path is None
    # Staging dir stays empty on rejection.
    assert list(staging.glob("*.yaml")) == []
    assert any(f.code == "shell_binary" for f in outcome.findings)


def test_stage_skill_warns_but_stages_when_only_non_blocking(tmp_path: Path) -> None:
    src = tmp_path / "warn.yaml"
    _write_yaml(
        src,
        _clean_descriptor(
            tools=[
                {
                    "name": "destructive",
                    "argv_template": ["rm", "-rf", "/tmp/scratch"],
                }
            ]
        ),
    )
    staging = tmp_path / "staging"

    outcome = stage_skill(src, staging_dir=staging)

    assert outcome.verdict == "staged"
    assert outcome.stage_path is not None
    assert outcome.stage_path.is_file()
    assert any(f.severity == "warn" for f in outcome.findings)


# --- stage_skill: input rejections --------------------------------------------


def test_stage_skill_missing_source(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    outcome = stage_skill(tmp_path / "nope.yaml", staging_dir=staging)
    assert outcome.verdict == "rejected_source"
    assert outcome.stage_path is None


def test_stage_skill_invalid_yaml(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("not: [valid", encoding="utf-8")
    outcome = stage_skill(src, staging_dir=tmp_path / "staging")
    assert outcome.verdict == "rejected_invalid"
    assert "YAML parse error" in outcome.error


def test_stage_skill_yaml_not_mapping(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("- a\n- b\n", encoding="utf-8")
    outcome = stage_skill(src, staging_dir=tmp_path / "staging")
    assert outcome.verdict == "rejected_invalid"


def test_stage_skill_pydantic_failure(tmp_path: Path) -> None:
    src = tmp_path / "bad.yaml"
    _write_yaml(
        src,
        _clean_descriptor(
            tools=[
                {
                    # Placeholder references `missing_arg` not in args_schema.
                    "name": "bad",
                    "argv_template": ["aegis", "{missing_arg}"],
                }
            ]
        ),
    )
    outcome = stage_skill(src, staging_dir=tmp_path / "staging")
    assert outcome.verdict == "rejected_invalid"
    assert "missing_arg" in outcome.error or "tools" in outcome.error


def test_stage_skill_conflict_when_already_staged(tmp_path: Path) -> None:
    src = tmp_path / "x.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "staging"
    assert stage_skill(src, staging_dir=staging).verdict == "staged"

    # Second stage for the same id conflicts.
    outcome2 = stage_skill(src, staging_dir=staging)
    assert outcome2.verdict == "rejected_conflict"
    assert outcome2.stage_path == staging / "vault_search.yaml"


# --- confirm_skill ------------------------------------------------------------


def test_confirm_skill_moves_file_into_catalog(tmp_path: Path) -> None:
    src = tmp_path / "x.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "staging"
    catalog = tmp_path / "catalog"
    staged = stage_skill(src, staging_dir=staging)
    assert staged.verdict == "staged"

    outcome = confirm_skill("vault_search", staging_dir=staging, catalog_dir=catalog)

    assert outcome.verdict == "confirmed"
    assert outcome.final_path == catalog / "vault_search.yaml"
    assert outcome.final_path is not None
    assert outcome.final_path.is_file()
    # Stage file was removed so repeated confirms fail loudly.
    assert not (staging / "vault_search.yaml").exists()


def test_confirm_skill_rejects_unstaged_id(tmp_path: Path) -> None:
    outcome = confirm_skill(
        "nope", staging_dir=tmp_path / "staging", catalog_dir=tmp_path / "catalog"
    )
    assert outcome.verdict == "rejected_not_staged"


def test_confirm_rejects_if_staged_file_tampered_to_blocking(tmp_path: Path) -> None:
    src = tmp_path / "x.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "staging"
    catalog = tmp_path / "catalog"
    stage_skill(src, staging_dir=staging)

    # Tamper with the staged file after staging.
    tampered = _clean_descriptor(
        tools=[{"name": "shelly", "argv_template": ["sh", "-c", "boom"]}]
    )
    _write_yaml(staging / "vault_search.yaml", tampered)

    outcome = confirm_skill("vault_search", staging_dir=staging, catalog_dir=catalog)

    assert outcome.verdict == "rejected_scan"
    assert not (catalog / "vault_search.yaml").exists()


def test_confirm_rejects_if_staged_file_id_changed(tmp_path: Path) -> None:
    src = tmp_path / "x.yaml"
    _write_yaml(src, _clean_descriptor())
    staging = tmp_path / "staging"
    catalog = tmp_path / "catalog"
    stage_skill(src, staging_dir=staging)

    # Tamper to change the id — confirm must refuse to install under the
    # original name (which would be a spoofing vector).
    other = _clean_descriptor(id="something_else")
    _write_yaml(staging / "vault_search.yaml", other)

    outcome = confirm_skill("vault_search", staging_dir=staging, catalog_dir=catalog)

    assert outcome.verdict == "rejected_invalid"
    assert "does not match" in outcome.error


# --- list_staged --------------------------------------------------------------


def test_list_staged_returns_sorted_ids(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    src1 = tmp_path / "a.yaml"
    src2 = tmp_path / "b.yaml"
    _write_yaml(src1, _clean_descriptor(id="alpha", intents=["alpha"]))
    _write_yaml(src2, _clean_descriptor(id="bravo", intents=["bravo"]))
    stage_skill(src1, staging_dir=staging)
    stage_skill(src2, staging_dir=staging)

    assert list_staged(staging) == ["alpha", "bravo"]


def test_list_staged_handles_missing_dir(tmp_path: Path) -> None:
    assert list_staged(tmp_path / "does_not_exist") == []
