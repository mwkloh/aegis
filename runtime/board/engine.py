"""`/board` engine — fan-out to panelists, optional synthesis.

Pure async, no Telegram or subprocess deps. Timeouts and per-panelist
failures degrade to `PanelistResponse(error=...)` so one flaky model
can never fail the whole board.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from runtime.board.config import BoardConfig, PanelistConfig
from runtime.model_router.clients.base import ChatMessage, ChatRequest, ChatResponse, ModelClient

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
        self._panelist_clients: list[tuple[PanelistConfig, ModelClient]] = [
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

    async def run(self, question: str) -> BoardResult:
        created_at = self._clock()
        board_id = "BOARD-" + secrets.token_hex(2)
        tasks = [
            self._call_panelist(panelist, client, question)
            for panelist, client in self._panelist_clients
        ]
        responses: tuple[PanelistResponse, ...] = tuple(
            await asyncio.gather(*tasks)
        ) if tasks else ()
        synthesis = await self._maybe_synthesise(question, responses)
        return BoardResult(
            board_id=board_id,
            question=question,
            created_at=created_at,
            panelist_responses=responses,
            synthesis=synthesis,
        )

    async def _call_panelist(
        self, panelist: PanelistConfig, client: ModelClient, question: str
    ) -> PanelistResponse:
        name = panelist.name
        model = panelist.model
        provider = panelist.provider
        persona = panelist.persona
        max_tokens = panelist.max_tokens
        timeout = getattr(self, "_timeout_override", None) or self._config.panelist_timeout_s
        request = ChatRequest(
            model=model,
            messages=[
                ChatMessage(role="system", content=persona),
                ChatMessage(role="user", content=question),
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        started = time.perf_counter()
        try:
            response: ChatResponse = await asyncio.wait_for(
                client.chat(request), timeout=timeout
            )
        except TimeoutError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "board.panelist.timeout",
                extra={"panelist_name": name, "model": model, "latency_ms": latency_ms},
            )
            return PanelistResponse(
                name=name, model=model, provider=provider,
                response="", latency_ms=latency_ms, error="timeout",
            )
        except Exception:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                "board.panelist.client_error",
                extra={"panelist_name": name, "model": model},
            )
            return PanelistResponse(
                name=name, model=model, provider=provider,
                response="", latency_ms=latency_ms, error="client_error",
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PanelistResponse(
            name=name, model=model, provider=provider,
            response=response.content, latency_ms=latency_ms, error=None,
        )

    async def _maybe_synthesise(
        self, question: str, responses: tuple[PanelistResponse, ...]
    ) -> str | None:
        if self._synth_client is None or self._config.synthesis is None:
            return None
        successful = [r for r in responses if r.error is None]
        if not successful:
            return None
        blocks = [f"# {r.name}\n{r.response}" for r in successful]
        user_prompt = (
            f"Question: {question}\n\n"
            f"Panelist perspectives:\n\n" + "\n\n---\n\n".join(blocks)
        )
        request = ChatRequest(
            model=self._config.synthesis.model,
            messages=[
                ChatMessage(role="system", content=self._config.synthesis.persona),
                ChatMessage(role="user", content=user_prompt),
            ],
            max_tokens=self._config.synthesis.max_tokens,
            temperature=0.3,
        )
        try:
            response = await self._synth_client.chat(request)
        except Exception:
            logger.exception("board.synthesis.failed")
            return None
        return response.content
