"""`/board` engine — fan-out to panelists, optional synthesis.

Pure async, no Telegram or subprocess deps. Timeouts and per-panelist
failures degrade to `PanelistResponse(error=...)` so one flaky model
can never fail the whole board.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from runtime.model_router.clients.base import ModelClient


class BoardConfigError(ValueError):
    """Raised at `BoardEngine.__init__` when a panelist provider is unknown."""


@dataclass(frozen=True)
class PanelistResponse:
    name: str
    model: str
    provider: str
    response: str
    latency_ms: int
    error: str | None


@dataclass(frozen=True)
class BoardResult:
    board_id: str
    question: str
    created_at: datetime
    panelist_responses: tuple[PanelistResponse, ...]
    synthesis: str | None


ClientFactory = Callable[[str, str], ModelClient]


class BoardEngine:
    """Placeholder — full implementation in Tasks 4-5."""
