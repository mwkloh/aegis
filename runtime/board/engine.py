"""`/board` engine — fan-out to panelists, optional synthesis.

Pure async, no Telegram or subprocess deps. Timeouts and per-panelist
failures degrade to `PanelistResponse(error=...)` so one flaky model
can never fail the whole board.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from runtime.board.config import BoardConfig
from runtime.model_router.clients.base import ModelClient

logger = logging.getLogger(__name__)


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
    """Parallel fan-out engine for the `/board` feature."""

    def __init__(
        self,
        config: BoardConfig,
        *,
        client_factory: ClientFactory,
        known_providers: frozenset[str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._factory = client_factory
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._validate_providers(known_providers)
        self._panelist_clients: list[tuple[object, ModelClient]] = [
            (p, client_factory(p.provider, p.model)) for p in config.panelists
        ]
        self._synth_client: ModelClient | None = (
            client_factory(config.synthesis.provider, config.synthesis.model)
            if config.synthesis is not None
            else None
        )

    @property
    def panelist_count(self) -> int:
        return len(self._config.panelists)

    def _validate_providers(self, known: frozenset[str]) -> None:
        unknown: set[str] = set()
        for p in self._config.panelists:
            if p.provider not in known:
                unknown.add(p.provider)
        if self._config.synthesis is not None and self._config.synthesis.provider not in known:
            unknown.add(self._config.synthesis.provider)
        if unknown:
            raise BoardConfigError(
                f"unknown provider(s) {sorted(unknown)}; known: {sorted(known)}"
            )
