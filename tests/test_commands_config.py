"""CommandsConfig — Pydantic model for the argv-only run_command tool."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.config import AegisConfig, CommandsConfig

pytestmark = pytest.mark.unit


def test_commands_config_defaults() -> None:
    cfg = CommandsConfig()
    assert cfg.allowed_binaries == ("ls", "cat", "head", "tail", "wc", "grep", "find", "file")
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
