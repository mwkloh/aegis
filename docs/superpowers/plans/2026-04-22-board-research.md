# /board --research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--research` flag to `/board` that pre-fetches Brave Search snippets and injects them into the panelist prompt before fan-out.

**Architecture:** A new `BoardResearcher` class (with an inner `BraveSearchClient`) lives in `runtime/board/researcher.py`. `BoardRunner` parses the `--research` flag and calls the researcher before calling the engine. Engine, writer, and all existing board behaviour are untouched.

**Tech Stack:** `httpx` (already installed), Brave Web Search API, Pydantic v2, pytest-asyncio.

---

## File Map

| File | Action |
|---|---|
| `runtime/board/config.py` | Add `ResearchConfig`; add `BoardConfig.research: ResearchConfig | None` |
| `runtime/board/researcher.py` | NEW — `BraveSearchError`, `SearchResult`, `ResearchContext`, `BraveSearchClient`, `BoardResearcher` |
| `runtime/chat/telegram/board_handler.py` | Parse `--research` flag; accept `researcher` kwarg; degrade path |
| `runtime/config.py` | Update `_coerce_board` signature to accept `env`; read `BRAVE_SEARCH_API_KEY` |
| `runtime/chat/telegram/bot.py` | Update `build_board_stack` to construct `BoardResearcher` from config |
| `tests/test_board_researcher.py` | NEW — unit tests for `BraveSearchClient` and `BoardResearcher` |
| `tests/test_telegram_board.py` | Extend with 4 research handler tests |

---

## Task 1: Add `ResearchConfig` to `runtime/board/config.py`

**Files:**
- Modify: `runtime/board/config.py`

- [ ] **Step 1: Write the failing test**

Add a new file `tests/test_board_research_config.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/michaelloh/projects/aegis
.venv/bin/pytest tests/test_board_research_config.py -v
```

Expected: `ImportError` — `ResearchConfig` does not exist yet.

- [ ] **Step 3: Add `ResearchConfig` and `BoardConfig.research` field**

Edit `runtime/board/config.py`. Add `ResearchConfig` before `BoardConfig`, and add the `research` field to `BoardConfig`:

```python
class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brave_api_key: str = Field(min_length=1, repr=False)
    top_k: int = Field(default=5, ge=1, le=10)
    timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)


class BoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    panelists: list[PanelistConfig] = Field(default_factory=list)
    synthesis: SynthesisConfig | None = None
    research: ResearchConfig | None = None
    output_dir: Path = Field(
        default_factory=lambda: Path.home() / ".aegis" / "boards"
    )
    excerpt_chars: int = Field(default=300, ge=50, le=1000)
    panelist_timeout_s: float = Field(default=60.0, ge=5.0, le=300.0)

    @field_validator("output_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_board_research_config.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
.venv/bin/pytest --tb=short -q
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add runtime/board/config.py tests/test_board_research_config.py
git commit -m "feat(board): add ResearchConfig to BoardConfig"
```

---

## Task 2: Create `runtime/board/researcher.py` — `BraveSearchClient`

**Files:**
- Create: `runtime/board/researcher.py`
- Create: `tests/test_board_researcher.py`

- [ ] **Step 1: Write the failing tests for `BraveSearchClient`**

Create `tests/test_board_researcher.py`:

```python
"""Unit tests for BraveSearchClient and BoardResearcher."""
from __future__ import annotations

import pytest
import respx
import httpx

from runtime.board.researcher import (
    BoardResearcher,
    BraveSearchClient,
    BraveSearchError,
    ResearchContext,
    SearchResult,
)

pytestmark = pytest.mark.unit

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_SAMPLE_RESPONSE = {
    "web": {
        "results": [
            {
                "title": "Local LLMs in 2025",
                "url": "https://example.com/llms",
                "description": "A roundup of local LLM options.",
            },
            {
                "title": "Ollama Guide",
                "url": "https://example.com/ollama",
                "description": "How to run models with Ollama.",
            },
        ]
    }
}


@respx.mock
@pytest.mark.asyncio
async def test_client_parses_200_into_search_results() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json=_SAMPLE_RESPONSE))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    results = await client.search("local LLMs")
    assert len(results) == 2
    assert results[0] == SearchResult(
        title="Local LLMs in 2025",
        url="https://example.com/llms",
        description="A roundup of local LLM options.",
    )


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_401() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    client = BraveSearchClient("bad-key", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_500() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(500, json={}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_timeout() -> None:
    respx.get(_BRAVE_URL).mock(side_effect=httpx.TimeoutException("timed out"))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    with pytest.raises(BraveSearchError):
        await client.search("anything")


@respx.mock
@pytest.mark.asyncio
async def test_client_returns_empty_list_when_no_results() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json={"web": {"results": []}}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    results = await client.search("obscure query")
    assert results == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_board_researcher.py -v
```

Expected: `ImportError` — `runtime.board.researcher` does not exist.

- [ ] **Step 3: Install `respx` if not already present**

```bash
.venv/bin/pip install respx
```

- [ ] **Step 4: Create `runtime/board/researcher.py` with `BraveSearchClient`**

```python
"""Brave Search pre-fetch for `/board --research`.

`BraveSearchClient` — thin async httpx wrapper around the Brave Web Search API.
`BoardResearcher` — orchestrates fetch and formats the context block for prompt injection.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchError(Exception):
    """Raised by `BraveSearchClient` on any non-200 or network failure."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str


@dataclass(frozen=True)
class ResearchContext:
    query: str
    results: tuple[SearchResult, ...]
    elapsed_ms: int


class BraveSearchClient:
    def __init__(
        self,
        api_key: str,
        *,
        top_k: int = 5,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._top_k = top_k
        self._timeout_s = timeout_s

    async def search(self, query: str) -> list[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        params = {"q": query, "count": self._top_k}
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            try:
                resp = await client.get(_BRAVE_SEARCH_URL, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                raise BraveSearchError("timeout") from exc
            except httpx.HTTPError as exc:
                raise BraveSearchError(f"http error: {exc}") from exc
        if resp.status_code >= 400:
            raise BraveSearchError(f"api error: {resp.status_code}")
        data = resp.json()
        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                )
            )
        return results
```

- [ ] **Step 5: Run `BraveSearchClient` tests**

```bash
.venv/bin/pytest tests/test_board_researcher.py -v -k "client"
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add runtime/board/researcher.py tests/test_board_researcher.py
git commit -m "feat(board): BraveSearchClient with httpx + BraveSearchError"
```

---

## Task 3: Add `BoardResearcher` to `runtime/board/researcher.py`

**Files:**
- Modify: `runtime/board/researcher.py`
- Modify: `tests/test_board_researcher.py`

- [ ] **Step 1: Write the failing tests for `BoardResearcher`**

Append to `tests/test_board_researcher.py`:

```python
@respx.mock
@pytest.mark.asyncio
async def test_researcher_fetch_returns_context_on_success() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(200, json=_SAMPLE_RESPONSE))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    researcher = BoardResearcher(client)
    ctx = await researcher.fetch("local LLMs")
    assert ctx is not None
    assert ctx.query == "local LLMs"
    assert len(ctx.results) == 2
    assert ctx.elapsed_ms >= 0


@respx.mock
@pytest.mark.asyncio
async def test_researcher_fetch_returns_none_on_api_failure() -> None:
    respx.get(_BRAVE_URL).mock(return_value=httpx.Response(500, json={}))
    client = BraveSearchClient("BSA-test", top_k=5, timeout_s=5.0)
    researcher = BoardResearcher(client)
    ctx = await researcher.fetch("anything")
    assert ctx is None


def test_format_context_produces_numbered_block() -> None:
    ctx = ResearchContext(
        query="local LLMs",
        results=(
            SearchResult(
                title="Local LLMs in 2025",
                url="https://example.com/llms",
                description="A roundup.",
            ),
        ),
        elapsed_ms=42,
    )
    researcher = BoardResearcher(BraveSearchClient("k"))
    text = researcher.format_context(ctx)
    assert "[Research context — Brave Search]" in text
    assert "1. Local LLMs in 2025" in text
    assert "https://example.com/llms" in text
    assert "A roundup." in text
    assert "---" in text


def test_format_context_handles_multiple_results() -> None:
    results = tuple(
        SearchResult(title=f"T{i}", url=f"https://u{i}.com", description=f"D{i}")
        for i in range(5)
    )
    ctx = ResearchContext(query="q", results=results, elapsed_ms=0)
    researcher = BoardResearcher(BraveSearchClient("k"))
    text = researcher.format_context(ctx)
    for i in range(5):
        assert f"{i + 1}. T{i}" in text
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_board_researcher.py -v -k "researcher"
```

Expected: FAIL — `BoardResearcher` not defined.

- [ ] **Step 3: Add `BoardResearcher` to `runtime/board/researcher.py`**

Append after the `BraveSearchClient` class:

```python
class BoardResearcher:
    """Fetch Brave Search snippets and format them for prompt injection."""

    def __init__(self, client: BraveSearchClient) -> None:
        self._client = client

    async def fetch(self, question: str) -> ResearchContext | None:
        started = time.perf_counter()
        try:
            results = await self._client.search(question)
        except BraveSearchError:
            logger.warning("board.researcher.fetch_failed", exc_info=True)
            return None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ResearchContext(
            query=question,
            results=tuple(results),
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def format_context(ctx: ResearchContext) -> str:
        lines = ["[Research context — Brave Search]"]
        for i, r in enumerate(ctx.results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   {r.url}")
            lines.append(f"   {r.description}")
        lines.append("---")
        return "\n".join(lines)


__all__ = [
    "BoardResearcher",
    "BraveSearchClient",
    "BraveSearchError",
    "ResearchContext",
    "SearchResult",
]
```

- [ ] **Step 4: Run all researcher tests**

```bash
.venv/bin/pytest tests/test_board_researcher.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add runtime/board/researcher.py tests/test_board_researcher.py
git commit -m "feat(board): BoardResearcher with fetch and format_context"
```

---

## Task 4: Update `BoardRunner` to handle `--research` flag

**Files:**
- Modify: `runtime/chat/telegram/board_handler.py`
- Modify: `tests/test_telegram_board.py`

- [ ] **Step 1: Write the 4 failing handler tests**

Append to `tests/test_telegram_board.py`:

```python
# ── research flag tests ──────────────────────────────────────────────────────


class _StubResearcher:
    """Minimal stand-in for BoardResearcher."""

    def __init__(
        self,
        ctx: "ResearchContext | None",
    ) -> None:
        self._ctx = ctx
        self.calls: list[str] = []

    async def fetch(self, question: str) -> "ResearchContext | None":
        self.calls.append(question)
        return self._ctx

    def format_context(self, ctx: "ResearchContext") -> str:
        return "[RESEARCH BLOCK]"


def _make_research_ctx() -> "ResearchContext":
    from runtime.board.researcher import ResearchContext, SearchResult
    return ResearchContext(
        query="q",
        results=(SearchResult(title="T", url="https://u.com", description="D"),),
        elapsed_ms=10,
    )


async def test_research_flag_enriches_question_passed_to_engine(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())
    writer = BoardWriter(output_dir=tmp_path)
    researcher = _StubResearcher(ctx=_make_research_ctx())
    runner = BoardRunner(
        engine=engine, writer=writer, registry=InFlightRegistry(), researcher=researcher
    )
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1,
        cmd=ParsedCommand(name="/board", args=("--research", "local", "LLMs")),
        message=msg,
    )
    assert len(engine.calls) == 1
    assert "[RESEARCH BLOCK]" in engine.calls[0]
    assert "local LLMs" in engine.calls[0]
    assert "research on" in msg.replies[0].initial_text


async def test_research_flag_with_no_researcher_returns_not_configured(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())
    writer = BoardWriter(output_dir=tmp_path)
    runner = BoardRunner(
        engine=engine, writer=writer, registry=InFlightRegistry(), researcher=None
    )
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1,
        cmd=ParsedCommand(name="/board", args=("--research", "q")),
        message=msg,
    )
    assert engine.calls == []
    assert "BRAVE_SEARCH_API_KEY" in msg.replies[0].initial_text


async def test_research_fetch_failure_runs_engine_with_original_question(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())
    writer = BoardWriter(output_dir=tmp_path)
    researcher = _StubResearcher(ctx=None)  # simulates fetch failure
    runner = BoardRunner(
        engine=engine, writer=writer, registry=InFlightRegistry(), researcher=researcher
    )
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1,
        cmd=ParsedCommand(name="/board", args=("--research", "local", "LLMs")),
        message=msg,
    )
    assert engine.calls == ["local LLMs"]
    final = msg.replies[0].edits[-1]
    assert "unavailable" in final.lower()


async def test_board_without_research_flag_never_calls_researcher(tmp_path: Path) -> None:
    engine = _StubEngine(result=_result())
    writer = BoardWriter(output_dir=tmp_path)
    researcher = _StubResearcher(ctx=_make_research_ctx())
    runner = BoardRunner(
        engine=engine, writer=writer, registry=InFlightRegistry(), researcher=researcher
    )
    msg = _FakeReplyable()
    await runner.run(
        chat_id=1,
        cmd=ParsedCommand(name="/board", args=("Should", "we", "migrate?")),
        message=msg,
    )
    assert researcher.calls == []
    assert engine.calls == ["Should we migrate?"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_telegram_board.py -v -k "research"
```

Expected: FAIL — `BoardRunner` doesn't accept `researcher` kwarg.

- [ ] **Step 3: Update `runtime/chat/telegram/board_handler.py`**

Replace the full file content with:

```python
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
```

- [ ] **Step 4: Run research handler tests**

```bash
.venv/bin/pytest tests/test_telegram_board.py -v -k "research"
```

Expected: 4 new tests PASS.

- [ ] **Step 5: Run full board handler test suite**

```bash
.venv/bin/pytest tests/test_telegram_board.py -v
```

Expected: all tests PASS (existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add runtime/chat/telegram/board_handler.py tests/test_telegram_board.py
git commit -m "feat(board): parse --research flag in BoardRunner"
```

---

## Task 5: Update `_coerce_board` in `runtime/config.py` to read `BRAVE_SEARCH_API_KEY`

**Files:**
- Modify: `runtime/config.py`

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_board_research_config.py`:

```python
from runtime.config import _coerce_board


def test_coerce_board_builds_research_config_from_env() -> None:
    raw = {
        "research": {
            "brave_api_key": "${BRAVE_SEARCH_API_KEY}",
            "top_k": 3,
        },
        "panelists": [],
    }
    env = {"BRAVE_SEARCH_API_KEY": "BSA-real-key"}
    cfg = _coerce_board(raw, env)
    assert cfg.research is not None
    assert cfg.research.brave_api_key == "BSA-real-key"
    assert cfg.research.top_k == 3


def test_coerce_board_drops_research_when_key_absent() -> None:
    raw = {
        "research": {
            "brave_api_key": "${BRAVE_SEARCH_API_KEY}",
        },
        "panelists": [],
    }
    env: dict[str, str] = {}
    cfg = _coerce_board(raw, env)
    assert cfg.research is None


def test_coerce_board_no_research_block_gives_none() -> None:
    raw = {"panelists": []}
    cfg = _coerce_board(raw, {})
    assert cfg.research is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_board_research_config.py -v -k "coerce"
```

Expected: FAIL — `_coerce_board` takes 1 argument, not 2.

- [ ] **Step 3: Update `_coerce_board` in `runtime/config.py`**

Replace the existing `_coerce_board` function:

```python
def _coerce_board(raw: Any, env: dict[str, str]) -> BoardConfig:
    """Build a `BoardConfig` from `config.json` → `board` + env.

    Reads BRAVE_SEARCH_API_KEY from env and injects it into the research
    block (overriding the ${VAR} placeholder). Key absent → research=None.
    Missing / non-dict raw → default (empty panelists, no research).
    """
    if not isinstance(raw, dict):
        return BoardConfig()
    board_raw = dict(raw)
    research_raw = board_raw.get("research")
    brave_key = env.get("BRAVE_SEARCH_API_KEY")
    if isinstance(research_raw, dict):
        if brave_key:
            research_copy = dict(research_raw)
            research_copy["brave_api_key"] = brave_key
            board_raw["research"] = research_copy
        else:
            board_raw.pop("research", None)
    try:
        return BoardConfig.model_validate(board_raw)
    except Exception:
        return BoardConfig()
```

Also update the call site inside `_coerce` (search for `_coerce_board(cfg.get("board"))`):

```python
board = _coerce_board(cfg.get("board"), env)
```

- [ ] **Step 4: Run config tests**

```bash
.venv/bin/pytest tests/test_board_research_config.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/pytest --tb=short -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add runtime/config.py tests/test_board_research_config.py
git commit -m "feat(config): _coerce_board reads BRAVE_SEARCH_API_KEY from env"
```

---

## Task 6: Update `build_board_stack` to wire `BoardResearcher`

**Files:**
- Modify: `runtime/chat/telegram/bot.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_build_board_stack_research.py`:

```python
"""Integration test for build_board_stack wiring BoardResearcher."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from runtime.board.config import BoardConfig, PanelistConfig, ResearchConfig
from runtime.config import AegisConfig, ModelConfig, ProviderConfig, StorageConfig, TelegramConfig, VaultIndexingConfig
from runtime.chat.telegram.long_running import InFlightRegistry

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
    from runtime.chat.telegram.bot import build_board_stack
    from runtime.chat.telegram.board_handler import BoardRunner

    cfg = _cfg_with_research(brave_key="BSA-test")
    runner = build_board_stack(cfg, registry=InFlightRegistry())
    assert isinstance(runner, BoardRunner)
    assert runner._researcher is not None


def test_build_board_stack_researcher_is_none_when_no_research_config() -> None:
    from runtime.chat.telegram.bot import build_board_stack
    from runtime.chat.telegram.board_handler import BoardRunner

    cfg = _cfg_with_research(brave_key=None)
    runner = build_board_stack(cfg, registry=InFlightRegistry())
    assert isinstance(runner, BoardRunner)
    assert runner._researcher is None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/pytest tests/test_build_board_stack_research.py -v
```

Expected: FAIL — `BoardRunner` doesn't receive `researcher` from `build_board_stack`.

- [ ] **Step 3: Update `build_board_stack` in `runtime/chat/telegram/bot.py`**

Replace the existing `build_board_stack` function:

```python
def build_board_stack(
    cfg: AegisConfig,
    *,
    registry: InFlightRegistry,
) -> Any | None:
    """Assemble `BoardRunner` from config, or return `None` when no panelists.

    Builds a production `ClientFactory` that maps `"ollama"` → `OllamaClient`
    and `"openrouter"` → `OpenRouterClient`. Returns `None` when
    `cfg.board.panelists` is empty so `/board` gets a clear
    "not configured" reply via `BoardRunner` only when panelists are
    present — otherwise we skip construction entirely and the slash
    falls through to the sync dispatcher's `unknown_command`.
    """
    from runtime.board import BoardEngine, BoardWriter  # noqa: PLC0415
    from runtime.board.config import BoardConfig  # noqa: PLC0415
    from runtime.board.researcher import BoardResearcher, BraveSearchClient  # noqa: PLC0415
    from runtime.chat.telegram.board_handler import BoardRunner  # noqa: PLC0415

    board_cfg: BoardConfig = cfg.board
    if not board_cfg.panelists:
        return None

    def _factory(provider: str, model: str) -> ModelClient:
        if provider == "ollama":
            return OllamaClient(cfg)
        if provider == "openrouter":
            return OpenRouterClient(cfg)
        raise ValueError(f"unknown provider {provider!r}")

    try:
        engine = BoardEngine(
            board_cfg,
            client_factory=_factory,
            known_providers=frozenset({"ollama", "openrouter"}),
        )
    except OpenRouterConfigError:
        logger.info("board.disabled", extra={"reason": "openrouter_config"})
        return None

    researcher: BoardResearcher | None = None
    if board_cfg.research is not None:
        brave_client = BraveSearchClient(
            board_cfg.research.brave_api_key,
            top_k=board_cfg.research.top_k,
            timeout_s=board_cfg.research.timeout_s,
        )
        researcher = BoardResearcher(brave_client)

    writer = BoardWriter(output_dir=board_cfg.output_dir)
    return BoardRunner(
        engine=engine,
        writer=writer,
        registry=registry,
        excerpt_chars=board_cfg.excerpt_chars,
        researcher=researcher,
    )
```

- [ ] **Step 4: Run wiring test**

```bash
.venv/bin/pytest tests/test_build_board_stack_research.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/pytest --tb=short -q
```

Expected: all tests pass. Note the total count — it should be the prior baseline + new tests.

- [ ] **Step 6: Run lint and type-check**

```bash
.venv/bin/ruff check runtime/board/researcher.py runtime/board/config.py runtime/chat/telegram/board_handler.py runtime/chat/telegram/bot.py runtime/config.py
.venv/bin/mypy runtime/board/researcher.py runtime/board/config.py runtime/chat/telegram/board_handler.py --ignore-missing-imports
```

Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add runtime/chat/telegram/bot.py tests/test_build_board_stack_research.py
git commit -m "feat(board): wire BoardResearcher in build_board_stack"
```

---

## Post-implementation: Update `.env.example`

- [ ] **Step 1: Add `BRAVE_SEARCH_API_KEY` to `.env.example`**

Open `.env.example` and append:

```
# Brave Search API key — required for /board --research
# Get a free key at https://api.search.brave.com/
BRAVE_SEARCH_API_KEY=
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add BRAVE_SEARCH_API_KEY to .env.example"
```

---

## Self-Review Checklist

The plan was reviewed against the spec. All spec sections are covered:

| Spec requirement | Task |
|---|---|
| `--research` opt-in flag | Task 4 |
| `BraveSearchClient` with httpx | Task 2 |
| `SearchResult`, `ResearchContext` dataclasses | Task 2 |
| `BoardResearcher.fetch()` degrades on failure | Task 3 |
| `format_context()` prompt injection format | Task 3 |
| `ResearchConfig` in `BoardConfig` | Task 1 |
| `_coerce_board` reads `BRAVE_SEARCH_API_KEY` | Task 5 |
| `build_board_stack` wires researcher | Task 6 |
| "not configured" reply when key absent | Task 4 |
| Engine runs with original question on fetch failure | Task 4 |
| Telegram note on unavailable research | Task 4 |
| `--research` flag has no effect without researcher | Task 4 |
