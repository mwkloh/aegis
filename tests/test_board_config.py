"""Pydantic validation for BoardConfig and its panelist/synthesis children."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.board.config import BoardConfig, PanelistConfig, SynthesisConfig

pytestmark = pytest.mark.unit


def test_panelist_config_happy_path() -> None:
    p = PanelistConfig(
        name="Analyst",
        model="minimax/minimax-m2.7",
        provider="openrouter",
        persona="Be rigorous.",
    )
    assert p.name == "Analyst"
    assert p.max_tokens == 1024


def test_panelist_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PanelistConfig(
            name="x",
            model="y",
            provider="z",
            persona="p",
            bogus=True,  # type: ignore[call-arg]
        )


def test_panelist_config_is_frozen() -> None:
    p = PanelistConfig(name="x", model="y", provider="z", persona="p")
    with pytest.raises(ValidationError):
        p.name = "other"  # type: ignore[misc]


def test_synthesis_config_default_persona_is_non_empty() -> None:
    s = SynthesisConfig(model="m", provider="openrouter")
    assert "synthesis" in s.persona.lower()
    assert s.max_tokens == 512


def test_board_config_defaults() -> None:
    cfg = BoardConfig()
    assert cfg.panelists == []
    assert cfg.synthesis is None
    assert cfg.output_dir == Path.home() / ".aegis" / "boards"
    assert cfg.excerpt_chars == 300
    assert cfg.panelist_timeout_s == 60.0


def test_board_config_excerpt_chars_bounds() -> None:
    with pytest.raises(ValidationError):
        BoardConfig(excerpt_chars=10)
    with pytest.raises(ValidationError):
        BoardConfig(excerpt_chars=5000)


def test_board_config_timeout_bounds() -> None:
    with pytest.raises(ValidationError):
        BoardConfig(panelist_timeout_s=1.0)
    with pytest.raises(ValidationError):
        BoardConfig(panelist_timeout_s=400.0)


def test_board_config_expands_output_dir_tilde() -> None:
    cfg = BoardConfig(output_dir=Path("~/obsidian/Boards"))
    assert "~" not in str(cfg.output_dir)
    assert cfg.output_dir.is_absolute()
