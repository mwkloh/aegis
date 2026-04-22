# /board --research: Pre-fetch Research Phase

**Status:** Approved  
**Date:** 2026-04-22  
**Builds on:** `docs/superpowers/specs/2026-04-21-board-design.md`

## 1. Overview

Adds an opt-in `--research` flag to `/board` that runs a Brave Search pre-fetch before
fan-out to panelists. The top 5 result snippets (title, URL, description) are injected
into the question as a context block. Engine, writer, and existing board behaviour are
untouched.

## 2. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Trigger | `--research` opt-in flag | Zero latency impact on existing `/board` usage |
| Data source | Brave Web Search API | Free tier (2k req/month), clean JSON, no scraping |
| Fetch depth | Snippets only (top 5) | Tiny models have small context windows; snippets sufficient for grounding |
| Architecture | Dedicated `BoardResearcher` class | Engine stays pure; HTTP logic independently testable |

## 3. Architecture

### 3.1 New file: `runtime/board/researcher.py`

```
BraveSearchClient
  - __init__(api_key: str, *, top_k: int = 5, timeout_s: float = 10.0)
  - async search(query: str) -> list[SearchResult]
  - Uses httpx.AsyncClient with Authorization: Bearer <key>
  - Raises BraveSearchError on 4xx/5xx/timeout

SearchResult (frozen dataclass)
  - title: str
  - url: str
  - description: str

ResearchContext (frozen dataclass)
  - query: str
  - results: tuple[SearchResult, ...]
  - elapsed_ms: int

BoardResearcher
  - __init__(client: BraveSearchClient)
  - async fetch(question: str) -> ResearchContext | None
    - Returns None on any failure (degrades gracefully)
  - format_context(ctx: ResearchContext) -> str
    - Returns formatted prompt block for injection
```

### 3.2 Modified: `runtime/board/config.py`

New `ResearchConfig` model nested under `BoardConfig`:

```python
class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    brave_api_key: str = Field(min_length=1, repr=False)
    top_k: int = Field(default=5, ge=1, le=10)
    timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)

class BoardConfig(BaseModel):
    ...
    research: ResearchConfig | None = None
```

### 3.3 Modified: `runtime/chat/telegram/board_handler.py`

- `BoardRunner.__init__` accepts `researcher: BoardResearcher | None = None`
- `BoardRunner.run` parses `--research` from `cmd.args`:
  - If flag present and `researcher is None` → reply "Research not configured — add `BRAVE_SEARCH_API_KEY` to `~/.aegis/.env`" and return
  - If flag present and researcher configured → call `researcher.fetch(question)`, build enriched question
  - Update running message: `Running board (N panelists, research on)...`
- Engine always called with final question string (enriched or original)

### 3.4 Modified: `runtime/chat/telegram/bot.py`

`build_board_stack` constructs `BoardResearcher` when `config.board.research` is set,
passes it to `BoardRunner`. Same `OpenRouterConfigError`-style guard pattern used for
missing keys.

### 3.5 Modified: `runtime/config.py`

`_coerce_board` reads `research.brave_api_key` from env (`BRAVE_SEARCH_API_KEY`) with
the same substitution pattern used for `TELEGRAM_BOT_TOKEN`. Key absence → `research=None`.

## 4. Data Flow

```
/board --research <question>
  │
  ├─ 1. Parse --research flag, extract question
  │
  ├─ 2. BoardResearcher.fetch(question)
  │      → BraveSearchClient.search(query, top_k=5)
  │      → ResearchContext(results=[...], elapsed_ms=N)
  │
  ├─ 3. Format enriched_question:
  │
  │      [Research context — Brave Search]
  │      1. <title>
  │         <url>
  │         <description>
  │      2. ...
  │      ---
  │      Question: <question>
  │
  ├─ 4. engine.run(enriched_question)   ← engine untouched
  │
  └─ 5. Normal synthesis → writer → Telegram reply
```

The `BoardResult.question` field stores `enriched_question`, so the written Markdown
file reflects exactly what panelists received.

## 5. Error Handling

| Condition | Behaviour |
|---|---|
| `research` block absent / `BRAVE_SEARCH_API_KEY` unset | `researcher=None`; `--research` flag replies "Research not configured" and returns |
| Brave API 4xx (bad key) | `fetch()` returns `None`; Telegram note `[Research unavailable — proceeding without context]`; engine runs with original question |
| Brave API 5xx / unreachable | Same as above |
| Timeout (>10s) | Same as above |
| Zero results returned | Proceed without context block; note in Telegram reply |

No failure mode in the research phase can prevent the board from running.

## 6. Config Example

`~/.aegis/.env`:
```
BRAVE_SEARCH_API_KEY=BSA...
```

`~/.aegis/config.json`:
```json
"board": {
  "research": {
    "brave_api_key": "${BRAVE_SEARCH_API_KEY}",
    "top_k": 5,
    "timeout_s": 10.0
  },
  "panelists": [...]
}
```

## 7. Testing

### `tests/test_board_researcher.py` (new, ~8 tests)

- `BraveSearchClient` parses 200 response into `SearchResult` list
- `BraveSearchClient` raises `BraveSearchError` on 401/403
- `BraveSearchClient` raises on timeout
- `BraveSearchClient` handles empty results array
- `BoardResearcher.fetch()` returns formatted `ResearchContext`
- `BoardResearcher.fetch()` returns `None` on API failure
- `format_context()` produces expected prompt injection string
- `format_context()` handles single result and max results

### `tests/test_telegram_board.py` (extend, ~4 new tests)

- `--research` with configured researcher → enriched question passed to engine
- `--research` with `researcher=None` → "not configured" reply, engine not called
- `--research` with fetch failure → engine called with original question + Telegram note
- `/board` without `--research` → researcher never called (baseline latency preserved)

## 8. Files Changed

| File | Change |
|---|---|
| `runtime/board/researcher.py` | NEW |
| `runtime/board/config.py` | Add `ResearchConfig`, `BoardConfig.research` |
| `runtime/chat/telegram/board_handler.py` | Parse `--research`, wire researcher |
| `runtime/chat/telegram/bot.py` | Construct `BoardResearcher` in `build_board_stack` |
| `runtime/config.py` | `_coerce_board` reads `BRAVE_SEARCH_API_KEY` |
| `tests/test_board_researcher.py` | NEW |
| `tests/test_telegram_board.py` | Extend with 4 research tests |
