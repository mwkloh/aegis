"""Ensure `/board` is dispatched through `BoardRunner` in `route_command`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from runtime.chat.telegram.bot import route_command
from runtime.chat.telegram.dispatch import Dispatcher
from runtime.chat.telegram.auth import Authorizer

pytestmark = pytest.mark.unit


@dataclass
class _FakeMessage:
    text: str
    replies: list[str] = field(default_factory=list)

    async def reply_text(self, text: str) -> Any:
        self.replies.append(text)
        return self


@dataclass
class _FakeUpdate:
    effective_chat: Any
    effective_user: Any
    effective_message: Any


class _SpyBoardRunner:
    commands = frozenset({"/board"})

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def run(self, *, chat_id: int, cmd: Any, message: Any) -> None:
        self.calls.append((chat_id, cmd.name))


async def test_board_slash_dispatches_through_board_runner() -> None:
    authorizer = Authorizer((42,))
    dispatcher = Dispatcher(authorizer=authorizer, handlers={})
    board_runner = _SpyBoardRunner()
    update = _FakeUpdate(
        effective_chat=type("C", (), {"id": 42})(),
        effective_user=type("U", (), {"id": 42})(),
        effective_message=_FakeMessage(text="/board is coffee good?"),
    )
    await route_command(
        update,
        None,
        dispatcher=dispatcher,
        board_runner=board_runner,  # type: ignore[call-arg]
    )
    assert board_runner.calls == [(42, "/board")]


async def test_board_slash_denied_for_unauthorized_chat() -> None:
    authorizer = Authorizer((42,))
    dispatcher = Dispatcher(authorizer=authorizer, handlers={})
    board_runner = _SpyBoardRunner()
    update = _FakeUpdate(
        effective_chat=type("C", (), {"id": 99})(),
        effective_user=type("U", (), {"id": 99})(),
        effective_message=_FakeMessage(text="/board q?"),
    )
    await route_command(
        update,
        None,
        dispatcher=dispatcher,
        board_runner=board_runner,  # type: ignore[call-arg]
    )
    assert board_runner.calls == []
