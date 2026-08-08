"""Integration test for build_board_stack wiring BoardResearcher."""
from __future__ import annotations

import pytest

from runtime.board.config import BoardConfig, PanelistConfig, ResearchConfig
from runtime.chat.telegram.long_running import InFlightRegistry
from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
    VaultIndexingConfig,
)

pytestmark = pytest.mark.unit


def _cfg_with_research(*, brave_key: str | None) -> AegisConfig:
    research = ResearchConfig(brave_api_key=brave_key) if brave_key else None
    board = BoardConfig(
        panelists=[
            PanelistConfig(
                name="A",
                model="llama3.2:1b",
                provider="ollama",
                persona="You are an analyst.",
            )
        ],
        research=research,
    )
    return AegisConfig(
        models=ModelConfig(),
        providers=ProviderConfig(),
        telegram=TelegramConfig(),
        storage=StorageConfig(),
        vault_indexing=VaultIndexingConfig(),
        board=board,
    )


def test_build_board_stack_wires_researcher_when_key_present() -> None:
    from runtime.chat.telegram.board_handler import BoardRunner
    from runtime.chat.telegram.bot import build_board_stack

    cfg = _cfg_with_research(brave_key="BSA-test")
    runner = build_board_stack(cfg, registry=InFlightRegistry())
    assert isinstance(runner, BoardRunner)
    assert runner._researcher is not None


def test_build_board_stack_researcher_is_none_when_no_research_config() -> None:
    from runtime.chat.telegram.board_handler import BoardRunner
    from runtime.chat.telegram.bot import build_board_stack

    cfg = _cfg_with_research(brave_key=None)
    runner = build_board_stack(cfg, registry=InFlightRegistry())
    assert isinstance(runner, BoardRunner)
    assert runner._researcher is None
