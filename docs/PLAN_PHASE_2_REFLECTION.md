# Phase 2 — Reflection (Read-Only)

> Status: **Draft, awaiting sign-off**. Builds on `PLAN_PHASE_1_MODEL_INTEGRATION.md`.
> Adds the second cognitive plane: read JSONL events, cluster them into
> patterns, and draft human-reviewable proposals. **No execution effects.**

## 1. Goal

After running for a while, AEGIS produces structured artifacts that say
*"these are the recurring frictions in your behaviour, and here is what
might be done about them."* A human (Phase 3) decides whether to act.

End-state demo:

```text
$ make reflect
[reflection] window=today  sessions=3  events=147
[reflection] patterns: 4 detected → ~/.aegis/workspace/reflection/PATTERNS.md
[reflection] proposals: 2 drafted → ~/.aegis/workspace/reflection/PROPOSALS.md
[reflection] tier=reflection model=gemma4:e4b  status=ok
```

## 2. Non-negotiables

1. **Plane isolation** — `runtime/reflection/` may import from `runtime/events/`
   (read-only) and `runtime/model_router/` (LLM client only). It MUST NOT
   import or call any harness/tool/skill code.
2. **No tool execution** — pattern + proposal pipeline never invokes the
   harness, never opens network sockets except the Reflection LLM.
3. **No canon mutation** — never touches `AGENTS.md`, `USER.md`, `IDENTITY.md`,
   `SOUL.md`, `HEARTBEAT.md`, or any catalog skill YAML.
4. **Reflection LLM is optional** — missing `gemma4:e4b` produces a structural
   placeholder, never a hard failure (consistent with Phase 1 doctor warning).
5. **Append-only outputs** — every reflection run appends a dated section to
   `PATTERNS.md` / `PROPOSALS.md`. We never rewrite history.
6. **All boundaries Pydantic-validated.** Pattern records and proposal drafts
   are typed at construction.

## 3. Deliverables

### 3.1 Event reader (`runtime/reflection/event_reader.py`)

| Symbol | Role |
| --- | --- |
| `Event` (Pydantic) | `ts`, `session_id`, `type`, `payload` — frozen, extra=forbid |
| `read_window(sessions_dir, since: date \| None)` | Yields validated events from JSONL. Skips malformed lines, emits a single warning aggregate |

Defaults to "today" when `since` is None. Reads `<sessions_dir>/YYYY-MM-DD/*.jsonl`.

### 3.2 Pattern detector (`runtime/reflection/patterns.py`)

Pure-Python, no LLM. Deterministic clustering over the event stream.

| Detector | Trigger |
| --- | --- |
| `unknown_intent` | `intent.classified` with `intent=="unknown"` ≥ 3 in window |
| `low_confidence` | `intent.classified` with `confidence < 0.5` ≥ 3 in window |
| `tool_error` | `tool.result` with `status != "ok"` clustered by tool name |
| `tier1_unavailable` | Any `pattern.observed:tier1_missing` recorded during the window |
| `model_latency` | `model.call.end` p95 latency > 5000 ms by tier |

Each detector returns 0..N `PatternRecord`:

```python

class PatternRecord(BaseModel, frozen=True, extra="forbid"):
    detector: str            # detector id
    severity: Literal["low","medium","high"]
    count: int
    sample_session_ids: list[str]   # at most 3
    summary: str             # one short line, no payload bodies
```

**No raw user text or payload bodies leak into pattern records.** Only counts,
session ids, and the structural summary. (Same discipline as Phase 1
event payloads — model prompts never logged.)

### 3.3 Proposal drafter (`runtime/reflection/proposals.py`)

Uses the **Reflection** model (`gemma4:e4b` via `OllamaClient`, separate
`InstrumentedModelClient` with `tier="reflection"`).

| Symbol | Role |
| --- | --- |
| `Proposal` (Pydantic) | `id`, `pattern_detector`, `affected: list[str]`, `change: str`, `risk: Literal["low","medium","high"]`, `rationale: str` (max 2 KB) |
| `draft(patterns, client, model)` | One LLM call per pattern, JSON-only output, parsed under strict schema |
| `prompts/proposal_drafter.txt` | System prompt; placeholders `{pattern_json}`, `{available_skills}`. Demands single JSON object reply |

Failure modes:

- LLM unavailable → emit a stub `Proposal` with `change="(reflection LLM unavailable — manual review)"`, `risk="low"`. Never raise.
- Malformed JSON → log `pattern.observed:proposal_parse_failed`, skip pattern. Same instrumentation discipline as Phase 1.
- Extra/missing keys → rejected by Pydantic (`extra="forbid"`).

### 3.4 Markdown writer (`runtime/reflection/writer.py`)

Renders `PatternRecord` and `Proposal` lists into the existing
`PATTERNS.md` / `PROPOSALS.md` files at:

```text
~/.aegis/workspace/reflection/PATTERNS.md
~/.aegis/workspace/reflection/PROPOSALS.md
```

Append-only. Each run gets a header:

```markdown
## 2026-04-18T18:42Z — window=today, sessions=3, events=147
```

The reflection directory is created on first run. **Outside canon** —
not in `REQUIRED_FILES` for the doctor, not subject to canon protection.

### 3.5 CLI driver (`runtime/reflection/cli.py`)

```text
python -m runtime.reflection.cli [--since YYYY-MM-DD] [--dry-run]
```

- `--dry-run` writes nothing, prints summary.
- Default window: today.
- Returns 0 on success (including "no patterns found"), 1 only on
  unrecoverable I/O / config error.

`make reflect` target wraps this.

### 3.6 Doctor extension

Add a `reflection:` section row that confirms `gemma4:e4b` presence
(already a warning today — no change to severity). New row:
`reflection:writable` checks `~/.aegis/workspace/reflection/` is creatable.

### 3.7 Tests (`tests/`)

| File | Coverage |
| --- | --- |
| `test_event_reader.py` | Reads multi-day JSONL, skips malformed, validates schema, applies date filter |
| `test_patterns_detectors.py` | One synthetic event chain per detector — assert `PatternRecord` count + severity |
| `test_proposal_drafter.py` | respx-mocked Ollama returning valid JSON / malformed JSON / 500 → expected Proposal or stub |
| `test_reflection_writer.py` | Append-only behaviour, idempotent header format, no canon writes |
| `test_reflection_cli_e2e.py` | End-to-end against a fixture session dir — produces both files, exit 0 |

All under `tests/`, marked `@pytest.mark.unit` except the e2e.

## 4. Open Questions for Sign-off

1. **Output location** — `~/.aegis/workspace/reflection/PATTERNS.md` (proposed)
   keeps Plane 2 outputs visibly separate from canon. OK?
2. **Reflection model identity** — re-use the existing `gemma4:e4b` config
   field (no new env var). OK?
3. **Pattern thresholds** — `≥ 3 occurrences` is the trigger floor for
   noisy detectors (`unknown_intent`, `low_confidence`, `tool_error`).
   Acceptable, or want it lower for faster signal in early use?
4. **Proposal token cap** — `max_tokens=512` like Tier 1 reasoner. OK?

## 5. Build order

1. `event_reader.py` + tests
2. `patterns.py` + per-detector tests
3. `writer.py` + append-only tests
4. `proposal_drafter.py` + mocked LLM tests
5. `cli.py` + e2e
6. Doctor row + Makefile `reflect` target
7. Full gate (ruff + mypy --strict + pytest + bandit)

Each step keeps the gate green before the next starts.

## 6. Out of scope (deferred to Phase 3+)

- Telegram / CLI approval of proposals
- Writing to `DECISIONS.md`
- Any code generation from approved proposals
- Cross-day pattern aggregation beyond `--since`
- Embedding-based clustering (deterministic detectors only for v1)
