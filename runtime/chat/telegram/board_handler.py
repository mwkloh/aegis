"""`BoardRunner` — Telegram adapter for `/board`.

Thin layer that parses the slash args, guards against concurrent runs
with `InFlightRegistry`, drives `BoardEngine.run`, writes the Markdown
file, and edits the initial "Running board …" reply with the final
summary. All engine / writer failures degrade to a human-readable
reply; the runner itself never raises.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from runtime.board.engine import BoardResult
from runtime.chat.telegram.dispatch import ParsedCommand
from runtime.chat.telegram.long_running import InFlightRegistry

if TYPE_CHECKING:
    from runtime.board.researcher import BoardResearcher, ResearchContext

logger = logging.getLogger(__name__)

_USAGE = "Usage: /board [--research] <question>"
_NOT_CONFIGURED = (
    "/board is not configured — add `board.panelists` to ~/.aegis/config.json."
)
_NOT_CONFIGURED_RESEARCH = (
    "Research not configured — add BRAVE_SEARCH_API_KEY to ~/.aegis/.env"
)
_MAX_TELEGRAM_CHARS = 3500
_SYNTHESIS_EXCERPT_CHARS = 400


class _Editable(Protocol):
    async def edit_text(self, text: str) -> Any: ...


class _Replyable(Protocol):
    async def reply_text(self, text: str) -> _Editable: ...


class _EngineLike(Protocol):
    @property
    def panelist_count(self) -> int: ...
    async def run(self, question: str) -> BoardResult: ...


class _WriterLike(Protocol):
    @property
    def output_dir(self) -> Any: ...
    def write(self, result: BoardResult) -> Any: ...


class BoardRunner:
    """Coordinator for `/board`. Shares `InFlightRegistry` with `LongRunningRunner`."""

    commands = frozenset({"/board"})

    def __init__(
        self,
        *,
        engine: _EngineLike,
        writer: _WriterLike,
        registry: InFlightRegistry,
        excerpt_chars: int = 300,
        researcher: BoardResearcher | None = None,
    ) -> None:
        self._engine = engine
        self._writer = writer
        self._registry = registry
        self._excerpt_chars = excerpt_chars
        self._researcher = researcher

    async def run(
        self, *, chat_id: int, cmd: ParsedCommand, message: _Replyable
    ) -> None:
        args = list(cmd.args)
        research_mode = "--research" in args
        if research_mode:
            args.remove("--research")
        question = " ".join(args).strip()

        if not question:
            await message.reply_text(_USAGE)
            return
        if self._engine.panelist_count == 0:
            await message.reply_text(_NOT_CONFIGURED)
            return
        if research_mode and self._researcher is None:
            await message.reply_text(_NOT_CONFIGURED_RESEARCH)
            return
        if not self._registry.try_acquire(chat_id, "/board"):
            current = self._registry.current(chat_id) or "another command"
            await message.reply_text(
                f"Already running {current} in this chat. "
                "Wait for it to finish before starting another."
            )
            return

        mode_suffix = ", research on" if research_mode else ""
        sent = await message.reply_text(
            f"Running board ({self._engine.panelist_count} panelists{mode_suffix})..."
        )
        research_note: str | None = None
        try:
            if research_mode and self._researcher is not None:
                ctx: ResearchContext | None = await self._researcher.fetch(question)
                if ctx is not None and ctx.results:
                    question = self._researcher.format_context(ctx) + "\n\n" + question
                else:
                    research_note = "[Research unavailable — proceeding without context]"
            try:
                result = await self._engine.run(question)
            except Exception:
                logger.exception("telegram.board.engine_failed", extra={"chat_id": chat_id})
                await sent.edit_text("/board internal error")
                return
            path_or_none = self._try_write(result, chat_id=chat_id)
            body = self._format(result, path_or_none)
            if research_note:
                body = research_note + "\n\n" + body
            await sent.edit_text(_clip(body, _MAX_TELEGRAM_CHARS))
        finally:
            self._registry.release(chat_id)

    def _try_write(self, result: BoardResult, *, chat_id: int) -> Any:
        try:
            return self._writer.write(result)
        except Exception:
            logger.exception("telegram.board.write_failed", extra={"chat_id": chat_id})
            return None

    def _format(self, result: BoardResult, path: Any) -> str:
        lines: list[str] = []
        n = len(result.panelist_responses)
        total_ms = sum(r.latency_ms for r in result.panelist_responses)
        lines.append(f"Board {result.board_id} · {n} panelists · {total_ms // 1000}s")
        lines.append(f"Question: {result.question}")
        lines.append("")
        if result.synthesis is not None:
            lines.append("Synthesis")
            lines.append(_excerpt(result.synthesis, _SYNTHESIS_EXCERPT_CHARS))
            lines.append("")
        for r in result.panelist_responses:
            lines.append(f"── {r.name} ({r.model}) ──")
            if r.error is not None:
                lines.append(f"[Error: {r.error}]")
            else:
                lines.append(_excerpt(r.response, self._excerpt_chars))
            lines.append("")
        if path is None:
            lines.append("[file write failed — full board below]")
            lines.append("")
            lines.append(self._render_markdown_inline(result))
        else:
            lines.append(f"Full board → {path}")
        return "\n".join(lines).rstrip() + "\n"

    def _render_markdown_inline(self, result: BoardResult) -> str:
        parts = [f"# Board: {result.question}", ""]
        if result.synthesis is not None:
            parts += ["## Synthesis", "", result.synthesis, ""]
        for r in result.panelist_responses:
            parts += [f"## {r.name}", f"*{r.model} via {r.provider} · {r.latency_ms}ms*", ""]
            parts += [f"[Error: {r.error}]" if r.error else r.response, ""]
        return "\n".join(parts)


def _excerpt(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


__all__ = ["BoardRunner"]
