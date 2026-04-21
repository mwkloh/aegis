# Phase 8 — Local-model readiness, structured output, skills + CLI tools

> Status: **Drafted 2026-04-19**. Replaces the earlier Phase 8 web-UI
> stub, which has moved to `PLAN_PHASE_9_WEB_UI.md`. Web UI is deferred
> until after we've proven the local-model thesis in live use.

## 1. Why this phase

Phases 0–7 built the **governance substrate**: three-plane trust model,
harness-verified code execution, canonical event log with idempotent
governance, tiered memory, Telegram operator surface, selective vault
indexing. What they did not yet prove:

- That a cheap or local 7B–13B model can actually drive the system
  reliably end-to-end.
- That skills + tool calls (beyond the code harness) can be added
  without re-introducing the context-bloat pain operators have felt
  with OpenClaw-style always-injected workspaces.

Phase 8 closes both gaps. The operator claim at the end of this phase
is: **"I can run AEGIS with a local Ollama model + bge-m3 embeddings,
install skills I actually use, and trust that tool calls either
succeeded verifiably or surfaced a structured failure."**

## 2. Non-negotiables (carry-forward + new)

Same five principles as Phase 7 §2 (local-first, structural reliability,
separation of concerns, model-agnostic, fail-closed auth), plus:

### 2.1 Progressive disclosure over injection

The skill descriptor + its tool list land in the prompt **only when
an intent match selects the skill**. Never all skills at once. This
is the load-bearing rule that separates AEGIS's skill system from
AGENTS.md-injection designs.

### 2.2 LLM output is never authoritative for tool success

Extends Phase 3/4 governance beyond code patches to any tool call.
Every invocation carries a verdict emitted by the harness, not the
model. Reply rendering consults the verdict, not the model's claim.

### 2.3 CLI > MCP for tool execution

When a capability can be expressed as an argv-shaped CLI invocation,
prefer that over standing up an MCP server. MCP stays optional for
cases where a CLI genuinely cannot represent the surface (e.g.
long-lived stateful connections), but the default is subprocess +
argv. Keeps the context budget tight; reuses the already-proven
`AsyncioSubprocessRunner` + `apply_cli`-style verdict pattern.

### 2.4 Schema-validated structured output, always

Every LLM call that consumes a structured answer goes through a
validator-retry-escalate wrapper. Local and cheap-frontier models
both benefit; frontier models lose nothing.

## 3. Tracks

### Track A — Local model plumbing

| Step | Deliverable |
|---|---|
| A1 | `runtime/model_router/clients/ollama_client.py` already conforms to the `ModelClient` Protocol. Verify + harden: streaming path (optional), JSON-mode happy path, non-loopback-host refusal, timeout classification. |
| A2 | `memory/embeddings.py` — implement `Bgem3Embedder` backed by Ollama's `/api/embeddings`. Config knob `providers.ollama_embedding_model` (default `bge-m3`). Dim detected from first response. L2-normalized. |
| A3 | `runtime/model_router/router.py` — add `fast_local` / `smart_local` tiers. Config precedence: if Ollama is reachable AND `models.prefer_local=True`, use local; else fall back to OpenRouter. Never raise on degrade. |

### Track B — Output reliability

| Step | Deliverable |
|---|---|
| B1 | `runtime/model_router/structured_output.py` — `request_structured(client, messages, schema, *, max_retries=2, escalate_to=None) -> tuple[dict, StructuredOutcome]`. Outcome carries `attempts`, `escalated`, `error_kind`. Corrective-prompt retry appends the failure reason and the JSON schema to the system prompt. Emits `llm.structured_retry` / `llm.tier_escalated` / `llm.structured_failed` events. |
| B2 | Route all existing structured LLM touchpoints through B1: intent classifier, reflection pattern detection, proposal drafting, critic. No bare `client.chat()` calls that expect JSON. |

### Track C — Skills + CLI-tool harness

| Step | Deliverable |
|---|---|
| C1 | Extend `skills/catalog/*.yaml` schema with a `tools:` list: `[{name, argv_template, schema, timeout_ms, allow_net}]`. `argv_template` uses `{arg}` placeholders pulled from the skill's `args_schema`. |
| C2 | `runtime/skills/loader.py` — on intent match, pulls ONLY that skill's descriptor + its tool list into the prompt. Progressive disclosure. Cached parse per skill yaml. |
| C3 | `runtime/tools/harness.py` — generalizes `applier`'s verdict pattern: `ToolVerdict = Literal["verified","argv_rejected","exit_nonzero","timeout","schema_violation","host_denied"]`. Reuses `AsyncioSubprocessRunner`. Hard deny on `shell=True`. Optional stdout JSON-schema validation. Caps stdout to 32 KB, tail-clipped for logging. |
| C4 | `/skills` slash command in Telegram: `list|show <skill>|enable <skill>|disable <skill>`. State stored per chat in a new `chat_skill_state.db` SQLite (or reuse Tier 2 sqlite with a new table). |
| C5 | `aegis skill add <path-or-git>` console script: copy/clone descriptor into `~/.aegis/skills/`, run an LLM-based safety scan against the descriptor (flags shell metacharacters, suspicious argv patterns, non-local network). Operator confirms install via `/skills confirm <id>`. |

### Track D — Tool-call trust layer

| Step | Deliverable |
|---|---|
| D1 | `runtime/tools/record.py` — `record_tool_call(skill, tool, argv_hash, verdict, outcome_bytes)`. Idempotent on `(session, imp_id, skill, tool, argv_hash)` — mirror of `record_decision`. Writes a `tool.invoked` event to the Plane-1 stream. Structural payload only, no stdout bodies. |
| D2 | Reply rendering gate: if the model's reply asserts an action was performed (phrase detection) but no matching `tool.invoked` event with `verdict="verified"` is on record for that turn, the chat pipeline annotates the reply: `⚠️ unverified tool claim — not executed`. |

### Track E — `/board` multi-agent (design-only this phase)

Full implementation deferred. This phase writes a design doc:

- Panel schema (N panelists, each a skill/model combo).
- Parallel fan-out via `asyncio.gather` with per-panelist tool harness.
- Critic pass + synthesis pass.
- Result persisted to `~/.aegis/boards/<id>.md` and indexed into Tier 2.

File: `docs/DESIGN_BOARD_MEETING.md`. No code in Phase 8.

## 4. Gate / exit criteria

- `make test` green (ruff + bandit + mypy + pytest) across all new
  modules.
- Manual smoke on a local Ollama install: a Telegram `/status` works,
  free-form chat works using `llama3.1:8b` or similar, one installed
  skill executes a real CLI (suggest: a simple `search_vault` skill
  that calls `aegis vault search <query>`) and surfaces a verdict.
- Dropping `OPENROUTER_API_KEY` does not degrade the surface — all
  conversational + skill paths must run on local Ollama alone.
- Progressive-disclosure invariant: a grep of one turn's context
  payload shows only the matched skill descriptor, not the full
  catalog.

## 5. Sequencing

A1 → A2 → A3 → B1 → B2 → C1 → C2 → C3 → C4 → C5 → D1 → D2. Each track
is shippable on its own — A+B alone gives us reliable local chat; C+D
layer on the skill surface.

## 6. Post-Phase-8 decision point

After live use on local models, we revisit the Strategy A vs B fork
(see 2026-04-19 conversation memory):

- **Strategy A:** keep AEGIS as the reference governance substrate,
  proceed to Phase 9 (web UI).
- **Strategy B:** extract governance + tiered memory + harness as an
  importable library so OpenClaw / Kai / Atamai-likes can adopt the
  pattern. Web UI becomes a proof-point, not the product.

Phase 8 is the work that makes Strategy B credible. Phase 9 is the
work that makes Strategy A complete.
