# Phase 1 — Model Integration & Real Intent Classification

> Status: **Draft, awaiting sign-off**. Builds on `PLAN_PHASE_0_AND_WALKING_SKELETON.md`.
> Replaces deterministic prefix matching with real local-model inference; adds
> a Tier 1 reasoning path for skills that need a frontier model.

## 1. Goal

Take the walking skeleton from "echo only, rules only" to **"any short user message
can be classified by a local model and routed through a skill that may optionally
escalate to a frontier model — without breaking any Phase 0 invariant."**

End-state demo:

```text
> what time is it in Tokyo?
[intent.classified] intent=ask_question  confidence=0.81  model=gemma4:e2b
[skill.selected]    skill=ask_question
[reasoning.tier1]   model=openrouter/auto  tokens_in=120  tokens_out=64
[tool.invoked]      tool=respond
> Tokyo is currently 16 hours ahead of UTC...
```

The echo / ping path keeps working without any model call (zero latency floor).

## 2. Non-negotiables (carried from Phase 0)

1. Plane isolation — `runtime/` still imports nothing from `improvement/` or `coding_harness/`.
2. No mutation of canonical `.md` files.
3. Tier 1 (frontier model) is **never required for correctness** — every Tier 0
   path must still produce a valid contract on its own.
4. Every external boundary validated by Pydantic.
5. Single config gateway (`runtime/config.py`) — no scattered env reads.
6. `yaml.safe_load` only.

## 3. Deliverables

### 3.1 Model clients (`runtime/model_router/`)

| File | Role |
| --- | --- |
| `clients/__init__.py` | Re-exports `OllamaClient`, `OpenRouterClient`, `ModelClient` Protocol |
| `clients/base.py` | `ModelClient` Protocol + `ChatRequest` / `ChatResponse` Pydantic models |
| `clients/ollama_client.py` | Local HTTP client over `httpx.AsyncClient`. Hard-wired to `127.0.0.1` / `localhost`; refuses any other host |
| `clients/openrouter_client.py` | Frontier client. TLS-only. Reads `OPENROUTER_API_KEY` from `AegisConfig`. Honours `HTTP-Referer` + `X-Title` per OpenRouter docs |

Both clients share:

- httpx with `timeout=httpx.Timeout(connect=2, read=30, write=10, pool=5)`
- explicit `follow_redirects=False`
- bounded retries via `tenacity` (3 attempts, exponential backoff, only on connection / 5xx)
- a `health()` method the doctor can call

### 3.2 Real intent classifier (`runtime/intent/`)

| File | Role |
| --- | --- |
| `classifier.py` | Existing surface stays. Add `ModelBackedClassifier` that wraps `OllamaClient` |
| `prompts/intent_classifier.txt` | System prompt — fixed list of known intents loaded from registry |
| `parser.py` | Strict JSON parser with Pydantic; rejects anything that doesn't fit `{intent, confidence}` |

Behaviour:

1. Try the deterministic ruleset first (≤ 1 ms, no network).
2. Fallback to `gemma4:e2b` over Ollama when no rule fires.
3. If model returns non-JSON or unknown intent → `intent="unknown"` with `confidence=0`.
4. Hard ceiling on input length (e.g. 4 KB) before sending to model.

### 3.3 Tier 1 reasoning (`runtime/reasoning/`)

| File | Role |
| --- | --- |
| `skill_runner.py` | Existing surface stays. When `descriptor.requires_tier1=True`, route through `Tier1Reasoner` instead of raising |
| `tier1_reasoner.py` | Calls `OpenRouterClient` with a strict system prompt that demands a JSON `ToolIntent` payload |
| `prompts/tier1_skill.txt` | System prompt template — `{skill_id}`, `{tool}`, `{args_schema}`, `{user_text}` are the only interpolations |

Tier 1 path **only** runs when:

- `descriptor.requires_tier1 == True`
- `OPENROUTER_API_KEY` is present in `AegisConfig`
- Otherwise → graceful degradation: `ToolResult(status="error", error="tier1 unavailable: skill <id> requires it")` and a `pattern.tier1_missing` event for the reflection plane.

### 3.4 New skills (`runtime/skills/catalog/`)

| File | Skill | Tier 1? | Demonstrates |
| --- | --- | --- | --- |
| `time_query.yaml` | `ask_time` | No | Tier 0 → Python tool (no LLM) |
| `general_question.yaml` | `ask_question` | Yes | Tier 0 → Tier 1 → tool |

Plus the corresponding tools under `runtime/harness/tools/`:

- `time_tool.py` — pure stdlib `zoneinfo` lookup
- `respond_tool.py` — pass-through tool that wraps the Tier 1 reasoner output

### 3.5 Events (`runtime/events/`)

Add three event types:

- `MODEL_CALL_START` — `{tier, model, provider, prompt_tokens_estimate}`
- `MODEL_CALL_END` — `{tier, model, latency_ms, status, tokens_in, tokens_out}`
- `PATTERN_OBSERVED` — structured signal for the reflection plane (e.g. `tier1_missing`, `intent_unknown`)

These are the **instrumentation** Phase 1 of the IMPLEMENTATION_PLAN.md asks for —
real signals the future Reflection plane can cluster.

### 3.6 Doctor (`scripts/doctor.py`)

Add three checks:

1. `ollama:gemma4:e2b` — model is pulled (`/api/tags` lists it)
2. `ollama:gemma4:e4b` — reflection model is pulled
3. `openrouter:reachable` — only if `OPENROUTER_API_KEY` is set; performs a 2-token completion with `openrouter/auto` and reports latency

Failure of OpenRouter check is a **warning**, not an error (Tier 1 is optional).

### 3.7 Tests

| File | Coverage |
| --- | --- |
| `tests/test_ollama_client.py` | `respx`-mocked Ollama responses; verifies retry, timeout, host-allowlist enforcement |
| `tests/test_openrouter_client.py` | `respx`-mocked OpenRouter; verifies header injection, TLS-only refusal of `http://`, key redaction in logs |
| `tests/test_model_backed_classifier.py` | Asserts rule-first → model-fallback flow; verifies parser rejection of malformed JSON |
| `tests/test_tier1_reasoner.py` | `requires_tier1=True` path with mocked OpenRouter; degrades gracefully when key absent |
| `tests/test_time_skill_e2e.py` | E2E: `what time in Tokyo?` → ask_time → Python tool → reply, with full event chain |
| `tests/test_question_skill_e2e.py` | E2E: `requires_tier1` skill with mocked Tier 1, verifies model-call events |
| `tests/test_no_egress_without_key.py` | Asserts that with `OPENROUTER_API_KEY` unset, no httpx call ever leaves to OpenRouter |
| Update `tests/test_plane_isolation.py` | Add: no `httpx.Client` (sync) in runtime — async only |

Target: **100% of new code paths** behind a `respx` mock. Zero real network calls in CI.

## 4. Security defaults (per the security guidance)

- All HTTP calls use `httpx.AsyncClient` with explicit timeouts and `follow_redirects=False`.
- Ollama client refuses any host not in `{127.0.0.1, localhost, ::1}` (SSRF defense).
- OpenRouter client refuses any URL whose scheme is not `https`.
- Secrets never logged: `repr()` on `ProviderConfig` already redacts; events log model **names** never keys.
- Bandit + semgrep clean as a hard gate.
- Existing `defusedxml` / `bleach` deps stay reserved for skills that handle untrusted HTML/XML.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| `gemma4:e2b` returns malformed JSON often | Strict Pydantic parser → fall through to `intent="unknown"`; log a `pattern.intent_unparseable` event for Reflection |
| Frontier latency stalls UX | 30 s read timeout, async streaming reply optional; degrades to "thinking…" placeholder on slow paths |
| Token / cost runaway on OpenRouter | Hard cap `max_tokens=512` for Tier 1 in Phase 1; per-session counter logged |
| Network call escapes during tests | `tests/conftest.py` adds an autouse `respx_mock` that **fails any unmocked HTTP call** |
| Real skill emits unsafe HTML in reply | Defer skills that render external content to Phase 2; Phase 1 skills only return plain text |

## 6. Out of scope (Phase 1)

- Memory tier reads (sqlite-vec + bge-m3) — Phase 2.
- Telegram surface — Phase 3.
- OpenHarness subprocess wrapper — still in-process tool dispatch.
- Streaming token-by-token output — replies stay synchronous.
- Reflection plane logic — Phase 1 only **emits** the events, doesn't consume.

## 7. Definition of done

1. `make test` green (lint + mypy + pytest, all new code covered).
2. `make run`:
   - `echo hi` → echoes (no model call) — verified by absence of `model.call.*` events.
   - `ping` → pong (no model call).
   - `what time is it in Tokyo?` → real reply via `ask_time` skill, events show `model.call.start/end` for Tier 0 only.
3. With `OPENROUTER_API_KEY` set: a `requires_tier1` skill end-to-end with both Tier 0 and Tier 1 model events recorded.
4. With `OPENROUTER_API_KEY` unset: same skill returns a clear "tier1 unavailable" reply, emits `pattern.tier1_missing`, **does not** raise.
5. `aegis-doctor` shows green for Ollama + both gemma models pulled; warning (not failure) if OpenRouter key absent.
6. No new dependencies beyond what's already in `pyproject.toml`.
7. Plane isolation test still passes; new test confirms no sync `httpx.Client` leaked into runtime.

## 8. Build order

1. `OllamaClient` + tests (offline via `respx`).
2. `ModelBackedClassifier` + tests; wire into `build_pipeline`.
3. New events + update e2e tests to assert event chains.
4. `time_tool` + `time_query.yaml` skill + e2e test.
5. `OpenRouterClient` + tests.
6. `Tier1Reasoner` + `general_question.yaml` + tests for both with-key and without-key paths.
7. Doctor checks for both model pulls + optional OpenRouter ping.
8. Run full gate; fix; commit per logical chunk.
