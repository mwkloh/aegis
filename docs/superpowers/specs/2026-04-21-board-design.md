# `/board` Multi-Panelist Board Feature — Design Spec

**Date:** 2026-04-21  
**Status:** Implemented  
**Use cases:** Research, brainstorming, decision-making

---

## 1. Overview

`/board <question>` fans out a question to N configurable panelists — each a distinct model+persona combination — in parallel, optionally runs a synthesis pass, and delivers:

- A Telegram summary (synthesis excerpt + per-panelist excerpts + file path)
- A full Markdown document saved to a configurable output directory (default `~/.aegis/boards/`, supports Obsidian vault paths)

The feature is local-first: synthesis is optional and the board works fully offline if all panelists use Ollama.

---

## 2. Config Schema

Added to `AegisConfig` as `board: BoardConfig`. Loaded from the `board` key in `~/.aegis/config.json`.

```python
class PanelistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str                   # Display name, e.g. "Analyst"
    model: str                  # Model identifier, e.g. "minimax/minimax-m2.7"
    provider: str               # Free string — validated by BoardEngine at construction
    persona: str                # System prompt for this panelist
    max_tokens: int = 1024      # Per-response token budget


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    provider: str               # Free string — same validation as PanelistConfig
    persona: str = (
        "You are a synthesis assistant. Given multiple expert perspectives on a "
        "question, identify areas of agreement, key tensions, and produce a "
        "concise bottom-line summary."
    )
    max_tokens: int = 512


class BoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    panelists: list[PanelistConfig] = Field(default_factory=list)
    synthesis: SynthesisConfig | None = None   # absent = format-only, no extra model call
    output_dir: Path = Field(default_factory=lambda: Path.home() / ".aegis" / "boards")
    excerpt_chars: int = Field(default=300, ge=50, le=1000)
    panelist_timeout_s: float = Field(default=60.0, ge=5.0, le=300.0)
```

`AegisConfig` gains: `board: BoardConfig = Field(default_factory=BoardConfig)`.

If `panelists` is empty, `/board` replies with a clear not-configured message — no crash.

### Example `config.json` snippet

```json
{
  "board": {
    "panelists": [
      {
        "name": "Analyst",
        "model": "minimax/minimax-m2.7",
        "provider": "openrouter",
        "persona": "You are a rigorous analyst. Identify risks, gaps, and second-order effects. Be concise and direct."
      },
      {
        "name": "Strategist",
        "model": "llama3.1:8b",
        "provider": "ollama",
        "persona": "You are a strategic thinker. Focus on opportunities, positioning, and long-term leverage."
      },
      {
        "name": "Pragmatist",
        "model": "gemma4:e2b",
        "provider": "ollama",
        "persona": "You are a pragmatist. Focus on execution feasibility, near-term actions, and resource constraints."
      }
    ],
    "synthesis": {
      "model": "minimax/minimax-m2.7",
      "provider": "openrouter"
    },
    "output_dir": "~/obsidian/Boards"
  }
}
```

---

## 3. Module Structure

```
runtime/board/
  __init__.py          # exports BoardEngine, BoardResult, BoardWriter
  config.py            # PanelistConfig, SynthesisConfig, BoardConfig
  engine.py            # BoardEngine — pure async, no Telegram dependencies
  writer.py            # BoardWriter — formats BoardResult → Markdown file

runtime/chat/telegram/
  board_handler.py     # BoardRunner — thin Telegram adapter
```

`runtime/config.py` — add `BoardConfig` import and `board` field to `AegisConfig`.

---

## 4. Engine & Data Flow

### Data types (`engine.py`)

```python
@dataclass(frozen=True)
class PanelistResponse:
    name: str
    model: str
    provider: str
    response: str        # full text; empty string on error
    latency_ms: int
    error: str | None    # None = success; short classifier on failure ("timeout", "client_error")

@dataclass(frozen=True)
class BoardResult:
    board_id: str                               # "BOARD-a3f2"
    question: str
    created_at: datetime                        # tz-aware UTC
    panelist_responses: tuple[PanelistResponse, ...]
    synthesis: str | None                       # None = skipped or failed gracefully
```

### `ClientFactory` protocol

```python
ClientFactory = Callable[[str, str], ModelClient]  # (provider, model) → ModelClient
```

Production factory (built from `AegisConfig`) maps known provider strings to `OllamaClient` or `OpenRouterClient`. At `BoardEngine.__init__`, each panelist's `(provider, model)` is resolved; an unrecognised provider raises `BoardConfigError` with the list of known providers.

Tests inject a stub factory that accepts any provider name and returns canned `ChatResponse` objects — no network required.

### Execution flow

```
BoardEngine.run(question: str) -> BoardResult
  │
  ├── asyncio.gather(*[_call_panelist(p, question) for p in panelists])
  │     Each call:
  │       ChatRequest(model=p.model, messages=[system(p.persona), user(question)],
  │                   max_tokens=p.max_tokens, temperature=0.7)
  │       → client.chat() with asyncio.wait_for(timeout=panelist_timeout_s)
  │       Timeout / exception → PanelistResponse(error="timeout"|"client_error", response="")
  │
  ├── if synthesis configured AND ≥1 panelist succeeded:
  │     Build synthesis user prompt from all PanelistResponse entries
  │     ChatRequest → synthesis client.chat()
  │     Failure → synthesis=None  (graceful degrade, never a hard error)
  │
  └── return BoardResult(board_id, question, created_at, panelist_responses, synthesis)
```

`board_id` is `"BOARD-" + secrets.token_hex(2)` (4 hex chars, same length convention as `JOB-` ids).

---

## 5. Writer (`writer.py`)

### File naming

```
YYYY-MM-DD-BOARD-<id>-<slug>.md
```

`<slug>` = first 6 words of question, lowercased, non-alphanumeric replaced with `-`, max 60 chars. Example: `2026-04-21-BOARD-a3f2-should-we-migrate-to-postgres.md`.

### Markdown structure

```markdown
---
board_id: BOARD-a3f2
question: "Should we migrate to Postgres?"
date: 2026-04-21
panelists:
  - name: Analyst
    model: minimax/minimax-m2.7
    provider: openrouter
  - ...
synthesis_model: minimax/minimax-m2.7
---

# Board: Should we migrate to Postgres?

*2026-04-21 · 3 panelists · BOARD-a3f2*

## Synthesis

<synthesis text, or *(synthesis not configured)*>

---

## Analyst
*minimax/minimax-m2.7 via openrouter · 14 232ms*

<full panelist response>

---

## Strategist
*llama3.1:8b via ollama · 8 891ms*

<full panelist response>

---

## Pragmatist
*gemma4:e2b via ollama · 3 210ms*

[Error: timeout]

---
```

`BoardWriter(output_dir: Path)` — `output_dir` is baked in at construction. `write(result: BoardResult) → Path` creates `output_dir` if missing, writes the file, and returns the path.

---

## 6. Telegram Integration

### `BoardRunner` (`board_handler.py`)

```python
class BoardRunner:
    commands = frozenset({"/board"})

    def __init__(self, *, engine: BoardEngine, writer: BoardWriter,
                 registry: InFlightRegistry) -> None: ...

    async def run(self, *, chat_id: int, cmd: ParsedCommand,
                  message: _Replyable) -> None:
        # 1. Empty question → usage reply, return
        # 2. No panelists configured → not-configured reply, return
        # 3. try_acquire(chat_id) → "already running" reply if busy
        # 4. try/finally: release on exit
        # 5. edit "Running board (N panelists)..."
        # 6. result = await engine.run(question)
        # 7. path = writer.write(result)  — or None on IOError (output_dir baked into writer)
        # 8. edit message with Telegram summary
```

`BoardRunner` shares `InFlightRegistry` with `LongRunningRunner` so `/board` respects the one-in-flight-per-chat guard already in place.

### Telegram reply format

```
Board BOARD-a3f2 · 3 panelists · 42s
Question: Should we migrate to Postgres?

Synthesis
<up to 400 chars of synthesis text>

── Analyst (minimax-m2.7) ──
<first excerpt_chars chars of response>…

── Strategist (llama3.1:8b) ──
<first excerpt_chars chars>…

── Pragmatist (gemma4:e2b) ──
[Error: timeout]

Full board → ~/obsidian/Boards/2026-04-21-BOARD-a3f2-should-we-migrate.md
```

If synthesis is absent, the Synthesis block is omitted. If file write failed, the "Full board →" line is replaced with the full Markdown inline (no silent data loss).

### `bot.py` wiring

`build_application()` instantiates `BoardRunner` and registers it alongside the existing `LongRunningRunner`. The Telegram update handler checks `board_runner.commands` before falling through to the sync dispatcher.

---

## 7. Error Handling

| Condition | Behaviour |
|-----------|-----------|
| No panelists in config | Friendly "not configured" reply; engine not called |
| Empty question | Usage hint; engine not called |
| Board already in flight | Standard in-flight message |
| Individual panelist timeout | `PanelistResponse(error="timeout")`; others proceed |
| Individual panelist exception | `PanelistResponse(error="client_error")`; others proceed |
| All panelists fail | `BoardResult` with all errors; synthesis skipped; file written with error markers |
| Synthesis fails | `synthesis=None`; result still written and delivered |
| File write fails | Full Markdown sent inline to Telegram; no silent loss |
| Unknown provider at startup | `BoardConfigError` raised at `BoardEngine.__init__`; bot startup fails with clear message |

---

## 8. Testing

All tests marked `pytest.mark.unit`. No network; stub `ClientFactory` injected everywhere.

| File | Coverage |
|------|----------|
| `tests/test_board_config.py` | Pydantic validation; `output_dir` `~` expansion; empty panelists valid; `AegisConfig` defaults |
| `tests/test_board_engine.py` | Parallel execution; one panelist fails; all fail; synthesis called with all responses; synthesis fails gracefully; synthesis skipped when `None`; unknown provider raises `BoardConfigError` |
| `tests/test_board_writer.py` | Filename pattern; slug truncation; Markdown structure; frontmatter; error panelist renders `[Error: ...]`; synthesis absent; `output_dir` created if missing |
| `tests/test_telegram_board.py` | Not-configured reply; empty question reply; successful run edits message; excerpt truncation; file write failure sends inline |
