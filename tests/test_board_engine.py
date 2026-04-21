"""BoardEngine — parallel fan-out, timeout handling, optional synthesis."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from runtime.board.config import BoardConfig, PanelistConfig, SynthesisConfig
from runtime.board.engine import (
    BoardConfigError,
    BoardEngine,
    BoardResult,
    PanelistResponse,
)
from runtime.model_router.clients.base import ChatRequest, ChatResponse

pytestmark = pytest.mark.unit


# --- Fakes -------------------------------------------------------------


class _FakeClient:
    """Canned `ChatResponse`, optional delay, optional raise."""

    def __init__(
        self,
        *,
        content: str = "ok",
        delay: float = 0.0,
        raises: Exception | None = None,
    ) -> None:
        self._content = content
        self._delay = delay
        self._raises = raises
        self.calls: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return ChatResponse(content=self._content, model=request.model)

    async def health(self) -> bool:
        return True


def _panelist(name: str, *, provider: str = "ollama", model: str = "m") -> PanelistConfig:
    return PanelistConfig(name=name, model=model, provider=provider, persona="p")


# --- Tests -------------------------------------------------------------


def test_panelist_response_is_frozen_dataclass() -> None:
    r = PanelistResponse(
        name="a", model="m", provider="p", response="x", latency_ms=0, error=None
    )
    with pytest.raises(Exception):
        r.name = "b"  # type: ignore[misc]


def test_board_result_holds_tuple_of_responses() -> None:
    responses = (
        PanelistResponse(name="a", model="m", provider="p", response="r", latency_ms=1, error=None),
    )
    result = BoardResult(
        board_id="BOARD-abcd",
        question="q",
        created_at=datetime(2026, 4, 21, tzinfo=UTC),
        panelist_responses=responses,
        synthesis=None,
    )
    assert result.board_id == "BOARD-abcd"
    assert result.panelist_responses == responses
