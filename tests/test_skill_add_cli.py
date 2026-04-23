from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.skill_add import main

pytestmark = pytest.mark.unit


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


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- argparse / routing -----------------------------------------------------


def test_no_args_returns_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err


# --- stage: happy path + JSON mode ------------------------------------------


def test_stage_happy_path_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "vault_search.yaml", _clean_descriptor())
    staging = tmp_path / "staging"

    rc = main([str(src), "--staging-dir", str(staging)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Staged" in out
    assert "vault_search" in out
    assert (staging / "vault_search.yaml").is_file()


def test_stage_emits_json_when_flag_passed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "x.yaml", _clean_descriptor())
    staging = tmp_path / "staging"

    rc = main([str(src), "--staging-dir", str(staging), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["verdict"] == "staged"
    assert payload["skill_id"] == "vault_search"
    assert payload["stage_path"].endswith("vault_search.yaml")
    assert payload["findings"] == []


def test_stage_warn_findings_are_listed_but_still_stages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `rm` is warn-severity (sensitive_binary) — should stage + show warning.
    src = _write(
        tmp_path / "warn.yaml",
        _clean_descriptor(
            tools=[
                {
                    "name": "clean",
                    "argv_template": ["rm", "-rf", "/tmp/scratch"],
                }
            ]
        ),
    )
    staging = tmp_path / "staging"

    rc = main([str(src), "--staging-dir", str(staging)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Staged" in out
    assert "warn" in out
    assert "sensitive_binary" in out


# --- stage: rejections ------------------------------------------------------


def test_stage_missing_source_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main([str(tmp_path / "nope.yaml"), "--staging-dir", str(tmp_path / "s")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "rejected_source" in out


def test_stage_blocking_scan_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(
        tmp_path / "bad.yaml",
        _clean_descriptor(
            tools=[{"name": "shelly", "argv_template": ["sh", "-c", "echo hi"]}]
        ),
    )
    staging = tmp_path / "staging"

    rc = main([str(src), "--staging-dir", str(staging)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "rejected_scan" in out
    assert "shell_binary" in out
    # Staging dir stays empty on rejection.
    assert list(staging.glob("*.yaml")) == []


def test_stage_invalid_yaml_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "bad.yaml"
    src.write_text("not: [valid", encoding="utf-8")

    rc = main([str(src), "--staging-dir", str(tmp_path / "s")])

    out = capsys.readouterr().out
    assert rc == 1
    assert "rejected_invalid" in out


# --- --confirm --------------------------------------------------------------


def test_confirm_moves_into_catalog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "x.yaml", _clean_descriptor())
    staging = tmp_path / "staging"
    catalog = tmp_path / "catalog"

    rc_stage = main([str(src), "--staging-dir", str(staging)])
    capsys.readouterr()  # drop stage output
    assert rc_stage == 0

    rc_confirm = main(
        [
            "--confirm",
            "vault_search",
            "--staging-dir",
            str(staging),
            "--catalog-dir",
            str(catalog),
        ]
    )

    out = capsys.readouterr().out
    assert rc_confirm == 0
    assert "Installed" in out
    assert (catalog / "vault_search" / "skill.yaml").is_file()
    assert not (staging / "vault_search.yaml").exists()


def test_confirm_unstaged_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "--confirm",
            "ghost",
            "--staging-dir",
            str(tmp_path / "staging"),
            "--catalog-dir",
            str(tmp_path / "catalog"),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "rejected_not_staged" in out


# --- --list -----------------------------------------------------------------


def test_list_empty_is_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--list", "--staging-dir", str(tmp_path / "staging")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "No skills staged" in out


def test_list_shows_staged_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "x.yaml", _clean_descriptor())
    staging = tmp_path / "staging"
    main([str(src), "--staging-dir", str(staging)])
    capsys.readouterr()  # drop stage output

    rc = main(["--list", "--staging-dir", str(staging)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "vault_search" in out


def test_list_json_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "x.yaml", _clean_descriptor())
    staging = tmp_path / "staging"
    main([str(src), "--staging-dir", str(staging)])
    capsys.readouterr()

    rc = main(["--list", "--staging-dir", str(staging), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"staged": ["vault_search"]}
