# Phase 7 — Telegram Operator Bot & Tiered Memory

> Status: **Drafted 2026-04-18**. Builds on
> `PLAN_PHASE_6_RECURSIVE_LOOP.md`. Gives the human operator a
> mobile, always-on surface for governance and session queries —
> without turning the bot into a general chat UI. The hard work of
> this phase is *not* the bot plumbing; it is the **tiered memory
> model** that keeps per-chat context from blowing up tokens as
> conversations accumulate.
>
> Prior art the user explicitly flagged as lessons:
> `~/projects/clawd-bot` and `atamai` — both shipped chatbot
> experiences where context tokens grew unbounded because there was
> no split between always-on persona/prefs and on-demand history.
> Phase 7 solves that *first*, then adds Telegram on top.

## 1. Goal

Give the operator (a single human, for now — `mwkloh@gmail.com`) a
way to:

1. **Approve or reject proposals from a phone.** Track B.
2. **Ask questions about sessions, decisions, and the Obsidian
   vault** without opening a laptop. Track C.
3. **Carry a chat history per Telegram `chat_id`** where each tenant
   has its own memory, and *total tokens sent to the model per turn
   stay bounded regardless of chat age*. Track A.

What Phase 7 is **not**:

- Not a public chat product. One operator, one authorized Telegram
  user ID per deployment. Multi-operator access is Phase 7.5.
- Not a web UI. The browser dashboard lives in Phase 8 behind
  Tailscale.
- Not an auto-approval surface. `/approve` still writes to
  `DECISIONS.md` via the same Phase 3 path. No new trust boundary
  is introduced.

End-state demos:

```text
# Track A — tiered memory enforces a bounded working set
$ .venv/bin/python -m runtime.chat.memory_stats --chat 123456789
tier 1 (identity+prefs): 2.1 KB   loaded every turn
tier 2 (episodic RAG):  47.3 KB   indexed — pulled per query
tier 3 (exec thread):   12.8 KB   last 12 turns — compresses at 24
                        oldest 112 turns archived into tier 2
```

```text
# Track B — operator governance from Telegram
You:       /pending
Bot:       2 proposals waiting.
            • P-007  risk:low   revisit IMP-a86b087a scope
            • P-008  risk:med   add regex cache to intent classifier
You:       /approve P-007 "scope too broad, split then retry"
Bot:       ✅ approved P-007 → DEC-042 @ 2026-04-18T14:03Z
           governance.decision emitted; applied_verdict pending
           next /harness run.
```

```text
# Track C — ask-anything over session notes + Obsidian vault
You:       what was I working on last Tuesday
Bot:       Pulled from tier-2 episodic (2026-04-11, 3 sessions):
           — finalized Phase 5 Track B critique flow (2 commits)
           — drafted IMP-a86b087a (applied_test_failed, reverted)
           — 1 vault note updated: "AEGIS Phase 6 scope.md"
           Want me to quote the vault note? /recall vault:phase-6-scope
```

## 2. Non-negotiables

Carry-forward from Phases 0–6 (unchanged).

**New for Phase 7.** The bot is a read-mostly interface:

1. The bot **never** writes to canon files directly. All writes go
   through existing Plane 3 APIs (`record_decision`, etc.) that
   already validate and emit `governance.decision`.
2. Every Telegram update carries an operator identity check at
   the edge. Unauthorized `chat_id` → single rate-limited refusal,
   no downstream processing.
3. **Token budget is enforced per turn, not per conversation.**
   Tier 1 + tier 3 trimmed thread + any on-demand tier 2 retrievals
   must sum to ≤ `TELEGRAM_TURN_TOKEN_BUDGET` (default 8 KB,
   configurable). If the budget would be exceeded, the lowest-tier
   content is dropped *before* the model call, not after.
4. Compression runs on a schedule, not inside a turn. A turn must
   never block on "compressing the last 50 messages."
5. Bot outages do not stall the loop. `make harness`,
   `make apply`, `make reflect` run independently from cron / the
   CLI. The bot only *observes* and *mediates approval*.
6. Multi-tenant boundary is the `chat_id`. A tenant's tier 2 and
   tier 3 stores are opaque to other tenants. One Telegram deployment
   can host N operators (Phase 7.5) each on their own `chat_id`.
7. **No PII in events.** Chat message bodies are *never* copied
   into Plane 1 events. Only structural signals
   (`{intent, token_budget_used, tier2_hits}`).
8. Stub-on-failure / never-raise discipline holds. A Telegram API
   outage returns a stub reply logged into the chat-local store;
   no exception bubbles to the operator.
9. **Transparency of recall.** Whenever the bot loads a verbatim
   raw thread from cold storage — whether via `/recall verbatim`
   or the §3.6 auto-recall policy — it **must** prepend a one-line
   announcement (`"Pulling raw thread from {date} ({n} turns,
   {kb} KB)…"`). Silent context injection is forbidden. The
   operator always sees what the model is about to see.

## 3. Tiered Memory Model (Track A — **built first**)

This is the core of Phase 7. Every other track assumes it works.

### 3.1 Three tiers

| Tier | Contents                                         | Lifetime       | Loaded per turn?    | Backing store                  |
|-----:|:-------------------------------------------------|:---------------|:--------------------|:-------------------------------|
| 1    | `IDENTITY.md`, `USER.md`, chat-local prefs JSON  | permanent      | **always**          | disk + in-memory cache         |
| 2    | session summaries, vault notes, decisions digest | months–years   | **on demand** (RAG) | sqlite-vec + bge-m3 embeddings |
| 3    | recent raw turns (user/bot messages)             | rolling window | **trimmed**         | chat-local sqlite table        |

Mapping onto `memory/tiers.py::Tier`:

- Tier 1 = `IDENTITY | PREFERENCES`
- Tier 2 = `EPISODIC | EXTERNAL`
- Tier 3 = `EXECUTION`

### 3.2 Compression contract

Tier 3 holds at most `_TIER3_KEEP_TURNS` raw exchanges (default 12).
When a turn lands as #13, a **background job** (not the request
path) runs:

1. Summarize the oldest N turns into a single `EpisodicMemory`
   record with fields `{chat_id, started_at, ended_at, summary,
   decisions_cited[], imp_ids_cited[]}`.
2. Embed the summary with `bge-m3` and write to tier 2.
3. Delete the raw turns from tier 3.

The summarizer runs on Tier 1 (local/cheap). Never a frontier model.
If the summarizer is unavailable, raw turns are **archived
verbatim** to tier 2 as a degraded-summary record — still
queryable, never dropped. This is the same stub-on-failure
discipline we use in the coding harness.

### 3.3 Retrieval contract

Per turn, the builder:

1. Loads tier 1 fully (small, fixed upper bound — ~4 KB).
2. Classifies the incoming message intent. If the intent is
   `recall | slash-query | ambiguous-reference`, runs a tier 2
   retrieval with top-k=5, re-ranks by recency+relevance, truncates
   to `_TIER2_BUDGET_BYTES`.
3. Appends the last ≤ `_TIER3_KEEP_TURNS` raw turns.
4. Sums bytes. If > `TELEGRAM_TURN_TOKEN_BUDGET`, drops tier 2
   results newest-first until budget fits. Tier 1 is never dropped.
5. Emits a `chat.turn.context` event carrying only structural
   counts (bytes per tier, number of retrievals, final budget
   usage). **No message bodies.**

### 3.4 Files

- `runtime/chat/memory/tier1.py` — loader for `IDENTITY.md` +
  `USER.md` + chat-local `prefs.json`. Pure file I/O, no network.
- `runtime/chat/memory/tier2.py` — sqlite-vec + bge-m3 wrappers.
  Reuses `memory/embeddings.py` and `memory/store_sqlite.py`.
- `runtime/chat/memory/tier3.py` — rolling-window raw-turn store,
  per `chat_id`.
- `runtime/chat/memory/compressor.py` — batch summarizer;
  runs via `make compress` and on a cron.
- `runtime/chat/memory/context_builder.py` — assembles per-turn
  payload; enforces the byte budget.
- `runtime/chat/memory/cold_storage.py` — `ColdRef` model + JSONL
  slice reader; verifies sha256 before handing bytes back.
- `runtime/chat/memory/recall.py` — auto-recall policy (intent
  flag → top-k RAG → score-based promotion → multi-match
  inline-keyboard); emits `chat.recall.auto`.
- `runtime/chat/memory/vault_indexer.py` — selective vault walker
  (see §5.1). Reads `VaultIndexingConfig`, walks `sources[]`, applies
  `glob`/`exclude`, embeds changed files into tier 2 with
  `{label, priority}`.
- `tests/test_chat_memory_*.py` — one per module, plus an
  integration test that proves total bytes stay ≤ budget under a
  synthetic 200-turn conversation, and a cold-ref round-trip test
  (write JSONL → archive → recall → byte-equal).

### 3.5 Cold storage & verbatim recall

**The compressor deletes raw turns from tier 3, but they are NOT
deleted from disk.** Plane 1 instrumentation already writes every
session to `cfg.storage.sessions_dir/<session_id>.jsonl` and that
file is the canonical archive. Phase 7 treats it as **cold storage**
and indexes pointers into it.

Every `EpisodicMemory` record carries a `cold_ref`:

```python
class ColdRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: str
    jsonl_path: str          # relative to cfg.storage.sessions_dir
    turn_range: tuple[int, int]  # [start_turn_idx, end_turn_idx) inclusive-exclusive
    sha256: str              # of the raw byte slice — detects later corruption
```

**Verbatim recall path** (operator types `/recall verbatim
<session_id>` *or* the auto-recall policy in §3.6 fires):

1. Look up the `EpisodicMemory` records for `session_id`.
2. Read the JSONL file, slice to `turn_range`, verify `sha256`.
3. Render the turns back into the chat (rate-limited; chunked if
   over Telegram's 4 KB message cap).
4. Emit a `chat.recall.verbatim` event with `{session_id,
   turn_range, bytes, auto: bool}` — no message bodies.

**Retention.** Default: keep cold storage forever (disk is cheap
relative to lost context). Configurable via
`AEGIS_RAW_RETENTION_DAYS` (e.g. 365 for compliance). When a JSONL
file ages out, the matching `EpisodicMemory` record is updated to
`cold_ref = None` and the summary becomes the only surviving form.

### 3.6 Auto-recall policy

The slash is the explicit override. The default is **the bot
notices recall references and pulls verbatim on its own**, then
*always tells the operator it did so*.

Per turn, after intent classification:

1. If intent ∈ `{recall, ambiguous-reference, direct-quote}`,
   run tier 2 RAG with top-k=5.
2. Inspect the top hit's similarity score:
   - **≥ `AUTO_RECALL_CONFIDENCE`** (default 0.78) **AND** the raw
     slice fits the per-turn byte budget → auto-promote to
     verbatim. Prepend a one-line "Pulling raw thread from
     {date} ({n} turns, {kb} KB)…" so the load is never silent.
   - **In `[0.55, 0.78)`** → return summary, attach a Telegram
     inline-keyboard button "Show raw thread" that issues
     `/recall verbatim <session_id>`. Operator-confirmed.
   - **< 0.55** → no recall surfaced; conversation proceeds with
     tier 1 + tier 3 only.
3. **Multi-match** (top 2 hits within 0.05 of each other) → ask
   once with an inline keyboard listing both candidates by date.
   Never guess.
4. **Budget collision.** When verbatim is loaded, tier 2 semantic
   retrieval for that turn drops from top-5 to top-1 to make room.
   Tier 1 is never dropped.

Threshold tuning: log every auto-recall decision (hit/miss
classification gathered from operator follow-up corrections) into
the `chat.recall.auto` event so we can re-tune
`AUTO_RECALL_CONFIDENCE` from real usage.

### 3.7 Why this is the recursive-loop step

Tier 2 is not just for chat. The same `sqlite-vec` corpus is
queryable by the Reflection plane:

- Proposal drafts can cite vault notes without pasting them.
- `reflect` clustering can join `PatternRecord`s with
  `EpisodicMemory` records on `imp_id` to generate higher-order
  patterns like "scope CT-001 keeps getting revisited in chat before
  each harness attempt."

We are not wiring that cross-plane join in Phase 7. We are *leaving
the seam open* — the tier 2 record shape must support it.

## 4. Track B — Operator Governance Commands

### 4.1 Authorized commands

| Slash                              | Action                                                  | Writes           |
|:-----------------------------------|:--------------------------------------------------------|:-----------------|
| `/pending`                         | list proposals with status `drafted`                    | —                |
| `/proposal P-NNN`                  | pretty-print one proposal (summary + risk + rationale)  | —                |
| `/approve P-NNN [rationale]`       | call `record_decision(approve, …)`                      | DECISIONS.md     |
| `/reject P-NNN [rationale]`        | same, `reject`                                          | DECISIONS.md     |
| `/defer P-NNN [rationale]`         | same, `defer`                                           | DECISIONS.md     |
| `/status`                          | last 24h: sessions, patterns, decisions, applies        | —                |
| `/decisions [N]`                   | tail `DECISIONS.md` N rows (default 10)                 | —                |
| `/recall verbatim <session_id>`    | replay raw turns from cold storage for `session_id`     | —                |
| `/recall vault:<slug>`             | quote a vault note verbatim                             | —                |
| `/apply CT-NNN`                    | kick `make apply ARGS="CT-NNN"`, stream verdict back    | standard apply   |
| `/harness CT-NNN [--with-context]` | kick `make harness` for one task                        | standard harness |

Non-goal: `/edit`, `/delete`, free-form "AEGIS, please fix X."
Those reopen trust boundaries we closed in Phases 3–5.

### 4.2 Rate-limiting & idempotency

- One command in flight per `chat_id` at a time. Stream progress via
  edited messages, not new ones (Telegram best practice).
- `/approve P-007` is idempotent — a second call after the decision
  is recorded replies with the existing `decided_at`, not a new row.

### 4.3 Files

- `runtime/chat/telegram/bot.py` — long-poll entrypoint.
- `runtime/chat/telegram/dispatch.py` — slash routing.
- `runtime/chat/telegram/formatters.py` — reply renderers.
- `runtime/chat/telegram/auth.py` — `chat_id` allow-list.
- `tests/test_telegram_dispatch.py` — no real Telegram; fake
  `Update` objects end-to-end.

## 5. Track C — Session & Vault Q&A

- Natural-language questions route through the tier 2 retriever.
- Vault access: read-only mount of the Obsidian vault directory.
  The bot *quotes* vault notes on explicit `/recall vault:<slug>`
  only — never paraphrases them into free-form generation. This is
  the same guardrail Phase 5 placed on context-mode drafts.

### 5.1 Selective vault indexing

Indexing the entire vault is wrong — Daily Notes, Inbox, and scratch
folders are noise. The operator selects which folders matter. Pattern
ported from `~/projects/atamai/packages/core/src/config/types.ts`
(`VaultIndexingConfigSchema` + `MemorySourceSchema`):

```yaml
# .aegis/config.yaml (chat-local prefs sit elsewhere; this is global)
vault_indexing:
  vault_root: ~/data/obsidian-vaults/secondbrain
  sources:
    - path: Research
      priority: 1.5         # rank above default in tier-2 retrieval
      glob: "**/*.md"
      exclude: ["**/Archive/**", "**/_drafts/**"]
      label: research
    - path: Projects/Docs
      priority: 1.0
      label: project-docs
    - path: Wiki
      priority: 1.2
      label: wiki
  reindex_interval_hours: 6   # heartbeat cron, not on every turn
```

Pydantic v2 model — keep the field shape compatible with atamai so
the user's existing config can be ported with a renamer:

```python
class VaultSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str                        # relative to vault_root
    priority: float = Field(default=1.0, ge=0.5, le=4.0)
    glob: str = "**/*.md"
    exclude: tuple[str, ...] = ()
    label: str | None = None

class VaultIndexingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    vault_root: str | None = None    # None ⇒ vault indexing disabled
    sources: tuple[VaultSource, ...] = ()
    reindex_interval_hours: int = Field(default=6, ge=1)
```

Indexer behaviour:

1. On startup AND on the heartbeat cron, walk each `source.path`
   under `vault_root`, apply `glob` then `exclude`, embed every
   `.md` file whose mtime > last-indexed-at, write into the same
   tier-2 `sqlite-vec` corpus with `{label, priority}` carried on
   the row.
2. Tier-2 retrieval multiplies similarity score by `priority`
   before ranking — so a `Research` hit at 0.72 beats a generic
   `notes` hit at 0.80.
3. Removed/renamed files are pruned on the next walk by comparing
   the indexed set against the current filesystem snapshot.
4. **Never write to the vault.** The mount is read-only at the
   filesystem layer too — the indexer process runs without write
   permission to `vault_root`.

Operator slashes for vault management:

| Slash                              | Action                                                  |
|:-----------------------------------|:--------------------------------------------------------|
| `/vault status`                    | last index run, file counts per source, drift size      |
| `/vault reindex [source]`          | force reindex of one source or all                      |
| `/vault sources`                   | list configured sources + priorities                    |

## 6. Build order

1. **Tier 3 store** (in-memory first, then sqlite). Add
   `chat_id → turns[]` with cap + eviction test.
2. **Tier 2 store**. Wire `memory/store_sqlite.py` + `embeddings.py`.
   One schema migration. `EpisodicMemory` + `VaultNote` tables.
3. **Tier 1 loader.** Deterministic, cached, tiny.
4. **Context builder** with byte-budget enforcement. Unit test:
   200-turn synthetic convo stays under budget.
5. **Compressor** (background job) **with `ColdRef` writes**.
   Unit test: archives age ≥ 12 turns into tier 2, deletes from
   tier 3, every new `EpisodicMemory` has a populated `cold_ref`,
   sha256 matches the source slice byte-for-byte.
6. **Cold-storage reader** (`cold_storage.py`). Unit test:
   round-trip `EpisodicMemory.cold_ref` → JSONL slice → original
   turns; sha256 mismatch raises a structured error (never silent).
7. **Auto-recall policy** (`recall.py`) wired into the context
   builder. Unit tests: high-confidence auto-promote, mid-band
   inline-button, low-band silence, multi-match keyboard, budget
   collision drops semantic top-5 → top-1.
8. **Telegram auth + dispatch skeleton** (no real network in tests).
9. **Read-only slashes**: `/pending`, `/proposal`, `/status`,
   `/decisions`, `/recall verbatim`, `/recall vault:`.
10. **Write slashes**: `/approve`, `/reject`, `/defer`. Each routes
    through `record_decision` — no new validation path.
11. **Long-running slashes**: `/harness`, `/apply` streamed via
    edited messages.
12. **Vault indexer** (§5.1) — selective folder walker driven by
    `VaultIndexingConfig.sources[]`. Unit tests: glob+exclude
    filtering, mtime-based incremental reindex, prune on file
    removal, priority weighting in retrieval ranking. Integration
    test: index a fake vault with two sources at different
    priorities, query, assert higher-priority source ranks first.
13. **E2E smoke**: fake Telegram server, full `/approve P-001` →
    `DECISIONS.md` + `governance.decision` event → `detect_all` sees
    it. Plus a recall smoke: ask a question that references a
    5-month-old (synthetic) session → bot announces the verbatim
    pull → renders the slice → emits `chat.recall.auto`.

## 7. Resolved decisions (design-time)

- **One operator per deployment in Phase 7.** Multi-operator (with
  per-operator decision authority) is Phase 7.5.
- **No message bodies in events.** Everything the Reflection plane
  sees is structural. This is non-negotiable because events feed
  pattern clustering that ships into proposal drafts — PII there is
  unacceptable.
- **Summarizer is local/Tier 1.** We never pay frontier tokens to
  compress a tenant's history. If the local model is down,
  compression degrades (raw archive) — it does not escalate.
- **Byte budget, not token count.** Bytes are deterministic;
  token counts depend on the tokenizer build. The builder enforces
  bytes; observability logs estimated tokens.
- **Tier 2 is shared substrate with Reflection.** Same
  `sqlite-vec` instance. Same `bge-m3` embeddings. Tenant isolation
  is enforced by a `chat_id` column, not separate databases.
- **Vault is read-only.** AEGIS never writes to Obsidian. Quoting
  is allowed; paraphrasing is allowed only with explicit
  `/recall vault:<slug>` permission.
- **Tailscale is Phase 8.** Bot authenticates by Telegram identity
  combined with an allow-listed `chat_id`. The web dashboard will
  bind to the Tailscale interface only.

## 8. Out of scope for Phase 7

- Web UI (Phase 8, Tailscale-gated).
- Multi-operator governance (Phase 7.5).
- Voice messages / image attachments.
- Tier 2 cross-plane joins into `PatternRecord`s (the seam is left
  open; actual wiring lands when a concrete pattern needs it).
- Auto-approving low-risk proposals. Operator still clicks.
- Pushing to `main` from the bot. `/apply` lands on an isolated
  branch just like `make apply`. A human pushes.

## 9. Exit criteria

- `make test` green including new `tests/test_chat_memory_*.py` and
  `tests/test_telegram_dispatch.py`.
- Synthetic 200-turn convo proven to stay under
  `TELEGRAM_TURN_TOKEN_BUDGET` across 50 random seeds.
- Operator can approve/reject/defer from their phone and the
  existing `DECISIONS.md` + `governance.decision` flow is
  unchanged.
- Compressor ran at least once in staging; tier 3 → tier 2 transfer
  verified by row counts + round-trip retrieval.
- `IMPLEMENTATION_PLAN.md` gains a Phase 7 row with Tracks A/B/C
  checked off and the exit-criteria statement reproduced.
