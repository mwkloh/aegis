"""CommandsConfig — Pydantic model for the argv-only run_command tool."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from runtime import config as config_mod
from runtime.config import AegisConfig, CommandsConfig, _coerce_commands

pytestmark = pytest.mark.unit


def test_commands_config_defaults() -> None:
    cfg = CommandsConfig()
    assert cfg.allowed_binaries == ("ls", "cat", "head", "tail", "wc", "grep", "file")
    assert cfg.timeout_ms == 15_000
    assert cfg.max_output_bytes == 32_768


def test_commands_config_timeout_ms_floor_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandsConfig(timeout_ms=99)


def test_commands_config_timeout_ms_ceiling_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandsConfig(timeout_ms=120_001)


def test_commands_config_timeout_ms_bounds_accepted() -> None:
    assert CommandsConfig(timeout_ms=100).timeout_ms == 100
    assert CommandsConfig(timeout_ms=120_000).timeout_ms == 120_000


def test_commands_config_max_output_bytes_floor_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandsConfig(max_output_bytes=1023)


def test_commands_config_max_output_bytes_ceiling_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandsConfig(max_output_bytes=262_145)


def test_commands_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        CommandsConfig(surprise=1)  # type: ignore[call-arg]


def test_commands_config_is_frozen() -> None:
    cfg = CommandsConfig()
    with pytest.raises(ValidationError):
        cfg.timeout_ms = 5_000  # type: ignore[misc]


def test_commands_config_allowed_binaries_overridable() -> None:
    cfg = CommandsConfig(allowed_binaries=("ls", "sleep"))
    assert cfg.allowed_binaries == ("ls", "sleep")


def test_aegis_config_has_commands_field() -> None:
    cfg = AegisConfig()
    assert hasattr(cfg, "commands")
    assert isinstance(cfg.commands, CommandsConfig)


def test_coerce_commands_returns_default_when_raw_is_none() -> None:
    assert _coerce_commands(None) == CommandsConfig()


def test_coerce_commands_falls_back_on_non_dict() -> None:
    assert _coerce_commands(["ls"]) == CommandsConfig()


def test_coerce_commands_parses_allowed_binaries_override() -> None:
    cfg = _coerce_commands({"allowed_binaries": ["ls", "rg", "python"]})
    assert cfg.allowed_binaries == ("ls", "rg", "python")


def test_coerce_commands_accepts_camelcase_key() -> None:
    cfg = _coerce_commands({"allowedBinaries": ["ls", "rg"]})
    assert cfg.allowed_binaries == ("ls", "rg")


def test_coerce_commands_parses_numeric_bounds() -> None:
    cfg = _coerce_commands({"timeout_ms": 5000, "max_output_bytes": 65536})
    assert cfg.timeout_ms == 5000
    assert cfg.max_output_bytes == 65536


def test_coerce_commands_degrades_per_field_on_invalid_binaries() -> None:
    # A non-list allowed_binaries is ignored; other valid keys still apply.
    cfg = _coerce_commands({"allowed_binaries": "ls", "timeout_ms": 5000})
    assert cfg.allowed_binaries == CommandsConfig().allowed_binaries
    assert cfg.timeout_ms == 5000


def test_coerce_commands_out_of_range_value_falls_back_to_default() -> None:
    # timeout_ms below the floor fails model validation → whole-config default.
    cfg = _coerce_commands({"timeout_ms": 1})
    assert cfg == CommandsConfig()


def test_coerce_commands_wires_into_aegis_config_load() -> None:
    # The `aegis_sandbox` conftest fixture points AEGIS_ROOT at a tmp dir, so
    # writing config.json there and reloading proves the commands block is
    # actually threaded through get_config() (not just default_factory).
    root = config_mod._aegis_root()
    (root / "config.json").write_text(
        json.dumps({"commands": {"allowed_binaries": ["ls", "rg"]}}),
        encoding="utf-8",
    )
    config_mod.reset_config()
    try:
        loaded = config_mod.get_config()
        assert loaded.commands.allowed_binaries == ("ls", "rg")
    finally:
        config_mod.reset_config()
