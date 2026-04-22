"""`/board` multi-panelist feature — engine, writer, and config.

Pure-async; no Telegram or subprocess deps. Wired into the Telegram
surface by `runtime/chat/telegram/board_handler.py`.

Imports are lazy (via ``__getattr__``) to avoid a circular-import cycle:
``runtime.config`` → ``runtime.board.config`` (package init) →
``runtime.board.engine`` → ``runtime.llm`` → ``runtime.config``.
Callers should import submodules directly, e.g.::

    from runtime.board.config import BoardConfig
    from runtime.board.engine import BoardEngine, BoardResult
    from runtime.board.writer import BoardWriter
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.board.config import BoardConfig, PanelistConfig, SynthesisConfig
    from runtime.board.engine import (
        BoardConfigError,
        BoardEngine,
        BoardResult,
        ClientFactory,
        PanelistResponse,
    )
    from runtime.board.writer import BoardWriter

__all__ = [
    "BoardConfig",
    "BoardConfigError",
    "BoardEngine",
    "BoardResult",
    "BoardWriter",
    "ClientFactory",
    "PanelistConfig",
    "PanelistResponse",
    "SynthesisConfig",
]


def __getattr__(name: str) -> object:
    """Lazy-load public symbols to prevent circular imports at package init."""
    if name in {"BoardConfig", "PanelistConfig", "SynthesisConfig"}:
        from runtime.board import config as _config  # noqa: PLC0415

        return getattr(_config, name)
    _ENGINE_EXPORTS = {
        "BoardConfigError",
        "BoardEngine",
        "BoardResult",
        "ClientFactory",
        "PanelistResponse",
    }
    if name in _ENGINE_EXPORTS:
        from runtime.board import engine as _engine  # noqa: PLC0415

        return getattr(_engine, name)
    if name == "BoardWriter":
        from runtime.board import writer as _writer  # noqa: PLC0415

        return _writer.BoardWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
