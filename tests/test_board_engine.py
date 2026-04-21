"""BoardEngine — parallel fan-out, timeout handling, optional synthesis."""
from __future__ import annotations

import asyncio
import dataclasses
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
    with pytest.raises(dataclasses.FrozenInstanceError):
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


def test_engine_rejects_unknown_provider() -> None:
    cfg = BoardConfig(panelists=[_panelist("A", provider="nope")])
    factory: Any = lambda provider, model: _FakeClient()  # noqa: E731
    with pytest.raises(BoardConfigError) as exc_info:
        BoardEngine(cfg, client_factory=factory, known_providers=frozenset({"ollama"}))
    assert "nope" in str(exc_info.value)


def test_engine_accepts_known_panelist_and_synthesis_providers() -> None:
    cfg = BoardConfig(
        panelists=[_panelist("A", provider="ollama")],
        synthesis=SynthesisConfig(model="m", provider="openrouter"),
    )
    engine = BoardEngine(
        cfg,
        client_factory=lambda provider, model: _FakeClient(),
        known_providers=frozenset({"ollama", "openrouter"}),
    )
    assert engine.panelist_count == 1


def test_engine_rejects_unknown_synthesis_provider() -> None:
    cfg = BoardConfig(
        panelists=[_panelist("A", provider="ollama")],
        synthesis=SynthesisConfig(model="m", provider="mystery"),
    )
    with pytest.raises(BoardConfigError):
        BoardEngine(
            cfg,
            client_factory=lambda provider, model: _FakeClient(),
            known_providers=frozenset({"ollama"}),
        )


def _engine(
    cfg: BoardConfig, clients: dict[str, _FakeClient]
) -> BoardEngine:
    # Factory returns the same fake per panelist name; tests stash
    # clients in a dict keyed by model so we can inspect calls after.
    def factory(provider: str, model: str) -> Any:
        return clients[model]

    return BoardEngine(
        cfg,
        client_factory=factory,
        known_providers=frozenset({"ollama", "openrouter"}),
        clock=lambda: datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
    )


async def test_run_executes_panelists_in_parallel_and_returns_all_responses() -> None:
    clients = {
        "m1": _FakeClient(content="A says"),
        "m2": _FakeClient(content="B says"),
    }
    cfg = BoardConfig(
        panelists=[
            _panelist("A", provider="ollama", model="m1"),
            _panelist("B", provider="ollama", model="m2"),
        ]
    )
    engine = _engine(cfg, clients)
    result = await engine.run("Should we migrate?")
    assert len(result.panelist_responses) == 2
    names = {r.name for r in result.panelist_responses}
    assert names == {"A", "B"}
    assert all(r.error is None for r in result.panelist_responses)
    assert result.synthesis is None
    assert result.board_id.startswith("BOARD-")
    assert len(result.board_id) == len("BOARD-") + 4
    assert result.question == "Should we migrate?"


async def test_run_timeouts_become_error_responses_and_others_proceed() -> None:
    clients = {
        "fast": _FakeClient(content="fast"),
        "slow": _FakeClient(content="slow", delay=1.0),
    }
    cfg = BoardConfig(
        panelists=[
            _panelist("Fast", provider="ollama", model="fast"),
            _panelist("Slow", provider="ollama", model="slow"),
        ],
        panelist_timeout_s=5.0,  # min allowed
    )
    engine = _engine(cfg, clients)
    engine._timeout_override = 0.05  # type: ignore[attr-defined]
    result = await engine.run("q?")
    by_name = {r.name: r for r in result.panelist_responses}
    assert by_name["Fast"].error is None
    assert by_name["Fast"].response == "fast"
    assert by_name["Slow"].error == "timeout"
    assert by_name["Slow"].response == ""


async def test_run_client_exception_becomes_error_response() -> None:
    clients = {
        "ok": _FakeClient(content="ok"),
        "bad": _FakeClient(raises=RuntimeError("boom")),
    }
    cfg = BoardConfig(
        panelists=[
            _panelist("Good", provider="ollama", model="ok"),
            _panelist("Bad", provider="ollama", model="bad"),
        ]
    )
    engine = _engine(cfg, clients)
    result = await engine.run("q?")
    by_name = {r.name: r for r in result.panelist_responses}
    assert by_name["Good"].error is None
    assert by_name["Bad"].error == "client_error"


async def test_run_calls_synthesis_with_panelist_texts_when_configured() -> None:
    clients = {
        "p1": _FakeClient(content="perspective 1"),
        "p2": _FakeClient(content="perspective 2"),
        "synth": _FakeClient(content="synthesised bottom line"),
    }
    cfg = BoardConfig(
        panelists=[
            _panelist("A", provider="ollama", model="p1"),
            _panelist("B", provider="ollama", model="p2"),
        ],
        synthesis=SynthesisConfig(model="synth", provider="openrouter"),
    )
    engine = _engine(cfg, clients)
    result = await engine.run("should we?")
    assert result.synthesis == "synthesised bottom line"
    synth_call = clients["synth"].calls[0]
    user_msg = synth_call.messages[-1].content
    assert "A" in user_msg
    assert "B" in user_msg
    assert "perspective 1" in user_msg
    assert "perspective 2" in user_msg


async def test_run_synthesis_skipped_when_all_panelists_failed() -> None:
    clients = {"bad": _FakeClient(raises=RuntimeError("boom")), "synth": _FakeClient(content="x")}
    cfg = BoardConfig(
        panelists=[_panelist("A", provider="ollama", model="bad")],
        synthesis=SynthesisConfig(model="synth", provider="openrouter"),
    )
    engine = _engine(cfg, clients)
    result = await engine.run("q?")
    assert result.synthesis is None
    assert clients["synth"].calls == []


async def test_run_synthesis_failure_is_graceful() -> None:
    clients = {
        "ok": _FakeClient(content="ok"),
        "synth": _FakeClient(raises=RuntimeError("synth down")),
    }
    cfg = BoardConfig(
        panelists=[_panelist("A", provider="ollama", model="ok")],
        synthesis=SynthesisConfig(model="synth", provider="openrouter"),
    )
    engine = _engine(cfg, clients)
    result = await engine.run("q?")
    assert result.synthesis is None
    assert result.panelist_responses[0].error is None


async def test_run_synthesis_timeout_is_graceful() -> None:
    clients = {
        "ok": _FakeClient(content="ok"),
        "synth": _FakeClient(content="x", delay=1.0),
    }
    cfg = BoardConfig(
        panelists=[_panelist("A", provider="ollama", model="ok")],
        synthesis=SynthesisConfig(model="synth", provider="openrouter"),
    )
    engine = _engine(cfg, clients)
    engine._timeout_override = 0.05  # type: ignore[attr-defined]
    result = await engine.run("q?")
    assert result.synthesis is None
    assert result.panelist_responses[0].error is None
