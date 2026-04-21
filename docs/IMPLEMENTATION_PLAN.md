# AEGIS — Phased Implementation Plan

This document defines a **safe, incremental rollout plan** for building AEGIS without breaking runtime stability.

---

## Phase 0 — Canon & Boundaries

**Goal:** Lock architecture and invariants before code changes.

- Commit ARCHITECTURE.md
- Commit IMPROVEMENT.md
- Confirm AGENTS.md, MEMORY.md, USER.md, IDENTITY.md, SOUL.md are canonical
- No runtime behavior change

✅ Exit criteria: Documentation approved and immutable

---

## Phase 1 — Instrumentation

**Goal:** Observe without acting.

- Log execution signals to EVENTS.md
- Capture tool failures, retries, corrections
- Do not summarize or interpret

✅ Exit criteria: EVENTS.md accumulating real data

---

## Phase 2 — Reflection (Read-Only)

**Goal:** Understand patterns.

- Cluster events into PATTERNS.md
- Generate PROPOSALS.md entries
- No code generation

✅ Exit criteria: At least 1 valid proposal drafted

---

## Phase 3 — Human Governance

**Goal:** Establish trust loop.

- Review proposals
- Record decisions in DECISIONS.md
- Approve at least one low-risk proposal

✅ Exit criteria: Approved proposal exists

---

## Phase 4 — Coding Harness

**Goal:** Draft safe improvements.

- Translate approved proposal into CODING_TASKS.md
- Run coding harness
- Generate diffs only

✅ Exit criteria: Reviewed diff ready to merge

---

## Phase 5 — Stabilization

**Goal:** Close the harness feedback loop and make drafts smarter
without weakening any safety property.

See `docs/PLAN_PHASE_5_STABILIZATION.md` for the full track-by-track
spec. Status as of 2026-04-18:

### Track A — Close the Loop (apply only what humans approved)

- ✅ A1 `ApplyOutcome` model + result writer (`coding_harness/result_writer.py`)
- ✅ A2 Applier preconditions + diff extraction
- ✅ A3 Subprocess runner (isolated `aegis/CT-NNN-<imp>` branch, `make test`)
- ✅ A4 Verdict literal extended to `applied_pass` / `applied_fail`
- ✅ A5 `apply_cli` (scriptable, `--status`, never pushes)
- ✅ A6 Doctor git rows + `make apply` target
- ✅ A7 Real-git e2e test for the apply flow

### Track B — Smarter Drafts (skill-aware context + critique-then-revise)

- ✅ B1 `ContextBundle` + `coding_harness/context.py` gatherer
      (15 KB total / 4 KB per file — Decision #5 sign-off)
- ✅ B2 `coder.draft_for(..., context=...)` wiring; Phase-4 prompt
      remains bit-identical when `context=None`
- ✅ B3 `coding_harness/critic.py` — bounded one-shot
      critique-then-revise; graceful degradation on every failure mode
- ✅ B4 `cli.py --with-context` flag (default OFF) emits
      `pattern.observed {pattern: harness_with_context, ...}` and runs
      the critique pass per draft

### Final gate

- ✅ Full `make test` (ruff + bandit + mypy + 237 unit + 12 e2e) green

✅ Exit criteria: Approved drafts can land on an isolated branch
   gated by `make test`; smarter drafts available behind an opt-in
   flag; no canon write paths added; no auto-merge; no push.

---

## Phase 6 — Recursive Loop

**Goal:** Close the cycle so apply outcomes feed Reflection and the
applier auto-cleans after itself — no human babysitting between runs.

See `docs/PLAN_PHASE_6_RECURSIVE_LOOP.md` for the full track-by-track
spec. Status as of 2026-04-19:

### Track B — Auto-revert in the applier (built first; ergonomics fix)

- ✅ B1 `_revert_apply` helper — `reset --hard HEAD` →
      `checkout <original>` → `branch -D <apply_branch>`
- ✅ B2 `apply_patch(..., auto_revert=True)` default;
      `_check_preconditions` returns `(reason, original_branch)` so
      revert has a target
- ✅ B3 Revert NEVER masks the verdict — failure is appended to
      `reason` as a soft warning suffix
- ✅ B4 `--no-revert` flag on `apply_cli` preserves Phase 5 behaviour
      for forensic inspection
- ✅ B5 Real-git e2e — failed tests → back on `feature/work`, apply
      branch deleted, working tree clean

### Track A — Outcome-driven pattern detectors (closes the loop)

- ✅ A1 `detect_apply_failed_repeat` — ≥2 non-clean apply verdicts
      for same `imp_id` → high (reads `governance.decision`)
- ✅ A2 `detect_harness_refused_repeat` — ≥2 `harness.refused` for
      same `imp_id` → medium
- ✅ A3 `detect_context_mode_helps` — `harness_with_context`
      followed by `applied_clean` for same `imp_id` → low (positive)
- ✅ A4 All three wired into `detect_all`
- ✅ A5 Unit tests pin behaviour (15 new tests in
      `tests/test_patterns_apply.py`)

### Final gate

- ✅ Full `make test` (ruff + bandit + mypy + 263 unit + 13 e2e) green

✅ Exit criteria: Apply verdicts re-enter the Reflection plane as
   structural patterns; the applier restores HEAD on its own after
   any failure that left a branch behind; both behaviours documented,
   tested, and reversible via `--no-revert`.

---

## Phase 7 — Telegram Operator Bot & Tiered Memory

**Goal:** Mobile operator surface (governance + recall) on top of a
strictly-bounded tiered memory model. Per-turn token budget is the
hard invariant.

See `docs/PLAN_PHASE_7_TELEGRAM.md` for the full track-by-track
spec. Status as of 2026-04-18:

### Track A — Tiered Memory (built first; everything else depends on it)

- ✅ A1 Tier 3 store (`runtime/chat/memory/tier3.py`) — in-memory
      rolling window, per-`chat_id` isolation, monotonic `turn_idx`,
      drainable eviction queue for the compressor (24 unit tests in
      `tests/test_chat_memory_tier3.py`)
- ✅ A2 Tier 2 store (`runtime/chat/memory/tier2.py`) — sqlite
      schema v2 migration in `memory/store_sqlite.py`,
      `Embedder` Protocol + deterministic `FakeEmbedder` +
      `Bgem3Embedder` skeleton in `memory/embeddings.py`,
      `EpisodicMemory` / `VaultNote` / `ColdRef` Pydantic models
      (frozen + `extra="forbid"`), cosine search over float32-BLOB
      embeddings (atamai pattern, no numpy dep), per-`chat_id`
      isolation on episodic search, `priority`-weighted vault
      ranking with optional `label_filter` (30 unit tests in
      `tests/test_chat_memory_tier2.py`)
- ⬜ A3 Tier 1 loader (IDENTITY/USER + chat-local prefs)
- ⬜ A4 Context builder + byte-budget enforcement
- ⬜ A5 Compressor (background job; writes `ColdRef` per §3.5)
- ⬜ A6 Cold-storage reader (`cold_storage.py`)
- ⬜ A7 Auto-recall policy (`recall.py`)

### Track B — Telegram Governance Commands

- ⬜ B1 Auth + dispatch skeleton
- ⬜ B2 Read-only slashes (`/pending`, `/proposal`, `/status`,
      `/decisions`, `/recall verbatim`, `/recall vault:`)
- ⬜ B3 Write slashes (`/approve`, `/reject`, `/defer`)
- ⬜ B4 Long-running slashes (`/harness`, `/apply` streamed)

### Track C — Vault Indexing

- ⬜ C1 Selective vault indexer (§5.1) with priority/glob/exclude

### Final gate

- ⬜ Full `make test` green; 200-turn synthetic convo proven under
      `TELEGRAM_TURN_TOKEN_BUDGET` across 50 random seeds; E2E
      `/approve P-001` → `DECISIONS.md` + `governance.decision` →
      `detect_all` sees it.
