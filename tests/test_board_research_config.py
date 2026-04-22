"""Unit tests for ResearchConfig and BoardConfig.research field."""
import pytest
from pydantic import ValidationError

from runtime.board.config import BoardConfig, ResearchConfig

pytestmark = pytest.mark.unit


def test_research_config_defaults() -> None:
    rc = ResearchConfig(brave_api_key="BSA-test")
    assert rc.top_k == 5
    assert rc.timeout_s == 10.0


def test_research_config_rejects_empty_key() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig(brave_api_key="")


def test_research_config_rejects_top_k_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ResearchConfig(brave_api_key="key", top_k=0)
    with pytest.raises(ValidationError):
        ResearchConfig(brave_api_key="key", top_k=11)


def test_board_config_research_defaults_to_none() -> None:
    cfg = BoardConfig()
    assert cfg.research is None


def test_board_config_accepts_research_block() -> None:
    cfg = BoardConfig(research=ResearchConfig(brave_api_key="BSA-test", top_k=3))
    assert cfg.research is not None
    assert cfg.research.top_k == 3
