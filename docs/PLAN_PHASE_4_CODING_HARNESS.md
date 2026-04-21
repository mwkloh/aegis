# Phase 4 — Coding Harness (Draft Only)

> Status: **Draft, awaiting sign-off**. Builds on
> `PLAN_PHASE_3_HUMAN_APPROVAL.md`. Adds the second half of Plane 3:
> **translate approved coding tasks into reviewable diffs**. Still no
> commits, no merges, no runtime mutation.

## 1. Goal

Phase 3 produced `~/.aegis/workspace/improvement/CODING_TASKS.md` —
human-approved work items, each anchored to an `IMP-id`. Phase 4 walks
that queue and, for every CT whose latest decision is `approve` and
which has no existing draft, asks a coding model to produce a unified
diff + summary + test notes + rollback. Drafts land under
`~/.aegis/workspace/coding_harness/diffs/`. **A human merges manually,
never the harness.**

End-state demo:

```text
$ make harness
[harness] loaded 1 approved CT, 0 already drafted
[harness] drafting CT-001 (IMP-a86b087a) → coding model openrouter:minimax/minimax-m2.7
[harness] wrote ~/.aegis/workspace/coding_harness/diffs/CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md
[harness] 1 drafted, 0 skipped, 0 refused
```

## 2. Non-negotiables

1. **Plane isolation** — `runtime/coding_harness/` may import from
   `runtime/config/`, `runtime/events/`, `runtime/model_router/`, and
   `runtime/improvement/` (read-only consumers: `load_tasks`,
   `latest_by_imp`). It MUST NOT import skills, harness execution code,
   chat, or intent. It NEVER calls `record_decision` or `queue_task`.
2. **No execution effects** — drafting writes a `.patch.md` file under
   the workspace. It does NOT apply the diff, run tests in the repo,
   commit, or push. Phase 5+ may add merge tooling; Phase 4 does not.
3. **Hard canon refusal** — if a CT's `scope` (or any path the LLM
   tries to touch in its diff) names a canonical file
   (`AGENTS.md`, `USER.md`, `IDENTITY.md`, `SOUL.md`, `HEARTBEAT.md`,
   or any file under `coding_harness/` itself), the harness refuses,
   writes a refusal note in place of a patch, and emits a
   `harness.refused` event. Refusals do NOT crash the run.
4. **Idempotent drafting** — a CT with an existing `.patch.md` under
   `diffs/` is skipped. `--force` redrafts (new timestamped file; the
   old one is kept as history — append-only directory).
5. **Latest-decision-wins** — only CTs whose latest `Decision.verdict`
   is `approve` are eligible. A CT later superseded by `reject` or
   `defer` is silently skipped (logged in stdout, not refused).
6. **Stub-on-failure** — if no coding client is configured or the LLM
   call/parse fails, write a structural stub patch.md flagged
   `[stub: <reason>]` instead of raising. The workflow always exits
   cleanly so partial progress is preserved.
7. **All boundaries Pydantic-validated** (`extra="forbid"`,
   `frozen=True` on inputs).
8. **No Telegram, no auto-merge, no auto-test execution** — Phase 5+.

## 3. Deliverables

### 3.1 Draft envelope (`runtime/coding_harness/draft.py`)

| Symbol | Role |
| --- | --- |
| `Draft` (Pydantic) | `ct_id`, `imp_id`, `model`, `summary` (≤512), `unified_diff` (≤32 KB), `test_notes` (≤2 KB), `rollback` (≤2 KB), `drafted_at`, `status: Literal["ok","stub","refused"]`, `reason: str` (empty when ok) |
| `_LLMReply` (Pydantic, internal) | mirror of `{summary, unified_diff, test_notes, rollback}` returned by the model |

The unified-diff field is treated as opaque text. We do NOT parse,
validate, or apply it; the patch.md is for a human to review.

### 3.2 Coding model wiring (`runtime/coding_harness/coder.py`)

| Symbol | Role |
| --- | --- |
| `draft_for(task, prior_decision, *, client, model, prompt_path, events) -> Draft` | One LLM call per CT. Returns `status="ok"` on success, `status="stub"` on failure, `status="refused"` if the CT touches canon. |
| `_DEFAULT_PROMPT` | `runtime/coding_harness/prompts/coder.txt` (renders CT scope, constraints, expected output, repo-style hints; never the full repo) |

Uses the existing `InstrumentedModelClient` with `tier="coding"`.
Reuses `OpenRouterClient` / `OllamaClient` — no new transport. The
model is read from a new `MODEL_CODING` config field (default = same
as `MODEL_SMART`, i.e. `minimax/minimax-m2.7` via OpenRouter). If
neither Ollama nor OpenRouter is reachable, `client` is `None` and
the draft is a stub.

### 3.3 Patch writer (`runtime/coding_harness/patch_writer.py`)

| Symbol | Role |
| --- | --- |
| `diffs_dir(workspace) -> Path` | `<workspace>/coding_harness/diffs/`, created on demand |
| `patch_filename(draft) -> str` | `CT-NNN__IMP-xxxxxxxx__YYYY-MM-DDTHHMMZ.patch.md` |
| `existing_drafts_for(workspace, ct_id) -> list[Path]` | All prior drafts for a CT (used by idempotency check) |
| `write_patch(workspace, draft) -> Path` | Renders markdown + writes |

Patch markdown layout:

```markdown
# CT-001 — IMP-a86b087a (status: ok)

- **Drafted:** 2026-04-18T12:07Z
- **Model:** openrouter:minimax/minimax-m2.7

## Summary
…

## Unified diff
```diff
…
```

## Test notes
…

## Rollback
…
```

Refusal/stub variants replace the diff block with a single fenced note.

### 3.4 CLI driver (`runtime/coding_harness/cli.py`)

```text
python -m runtime.coding_harness.cli              # draft for all eligible CTs
python -m runtime.coding_harness.cli --list       # show eligible CTs (no calls)
python -m runtime.coding_harness.cli --task CT-001
python -m runtime.coding_harness.cli --force      # redraft even if a patch exists
```

- "Eligible" = latest decision is `approve` AND scope is canon-clean.
- Returns 0 on clean exit (drafts, stubs, refusals all count as
  clean). Returns 1 only on I/O failure or unknown `--task` id.
- `make harness` wraps the no-arg form.

### 3.5 Config addition

`runtime/config.py` `ModelConfig` gains:

```python
coding: str = Field(
    default="minimax/minimax-m2.7",
    description="Plane 3 coding harness (OpenRouter by default).",
)
```

Read from `MODEL_CODING` env var. Falls back to `MODEL_SMART` when
unset (so existing `~/.aegis/.env` keeps working).

### 3.6 Doctor extension

Adds `coding_harness:writable` row (severity = warn, mirrors
`reflection:writable` and `improvement:writable`). No new model probe
— the coding model is the same OpenRouter endpoint already verified.

### 3.7 Workspace layout (additive)

```text
~/.aegis/workspace/
├── reflection/
├── improvement/
└── coding_harness/                        # NEW — Phase 4
    └── diffs/
        ├── CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md
        └── CT-002__IMP-7a3014d7__2026-04-19T0830Z.patch.md   # later
```

The repo-level `coding_harness/CODING_PROMPT.md`,
`coding_harness/README.md`, and `SAMPLE_CODING_TASK_CT-001.md` remain
canonical schemas (untouched).

### 3.8 Tests (`tests/`)

| File | Coverage |
| --- | --- |
| `test_coding_harness_draft.py` | `Draft` Pydantic validation, status literals, max-length enforcement |
| `test_coding_harness_patch_writer.py` | Filename format, dir creation, append-only behaviour, refusal/stub markdown variants |
| `test_coding_harness_coder.py` | Stub when client is None, stub on `httpx.HTTPError`, stub on schema rejection, `respx`-mocked happy path returns `status="ok"` |
| `test_coding_harness_canon_refusal.py` | Scope including `AGENTS.md` / `SOUL.md` etc. → `status="refused"` without an LLM call; `harness.refused` event emitted |
| `test_coding_harness_cli_scriptable.py` | `--task CT-001` drafts; `--task CT-999` → exit 1; idempotent skip when patch exists; `--force` writes a second patch |
| `test_coding_harness_cli_e2e.py` | Seed CT via Phase 3 helpers, run `main([])`, assert `.patch.md` present and well-formed (`@pytest.mark.e2e`) |

All under `tests/`, marked `@pytest.mark.unit` except the e2e.

### 3.9 Event types

`runtime/events/stream.py` adds:

```python
HARNESS_DRAFT_START   = "harness.draft.start"
HARNESS_DRAFT_END     = "harness.draft.end"   # status=ok|stub|refused
HARNESS_REFUSED       = "harness.refused"     # canon-touch attempt
```

Reflection plane will eventually pattern-match on
`harness.refused` to surface drift between proposals and canon.

## 4. Open questions for sign-off

1. **Coding model default** — default `MODEL_CODING` to
   `minimax/minimax-m2.7` (= same as `MODEL_SMART`, OpenRouter)?
   Alternative: keep coding off-by-default until the user sets
   `MODEL_CODING` explicitly, so a fresh install never spends tokens.
   I lean **default-on with stub-fallback** — matches the Phase 2
   pattern where missing config degrades gracefully.
2. **Canon refusal scope** — refuse if CT scope OR any path inside the
   produced unified diff hits canon? Diff parsing is fragile; I
   propose **scope-only check pre-call** plus a regex post-check on
   the returned diff (`^\+\+\+ b/(AGENTS|USER|IDENTITY|SOUL|HEARTBEAT)\.md`).
   Stub-or-refuse on hit.
3. **Patch-file history** — `--force` writes a new timestamped file
   alongside the old one (proposed). Alternative: rename the old one
   to `.superseded`. I lean append-only directory — simpler, mirrors
   the rest of the system.
4. **Skill-aware context** — the prompt currently sees only the CT's
   `scope`, `constraints`, `expected_output`, and a list of skill
   names. Should we also include the actual skill YAML for files in
   scope? I lean **not in Phase 4**: keeps token costs low and keeps
   the harness from drifting into "knows the whole repo" territory.
   Phase 5+ can add a `--with-context` opt-in.

## 5. Build order

1. `Draft` model + `patch_writer.py` + tests
2. `coder.py` (stub-only path first, then LLM path) + tests
3. Canon refusal path + tests
4. `MODEL_CODING` config addition + doctor row
5. `cli.py` (scriptable mode first, then full sweep) + tests
6. `Makefile` `harness` target + e2e test
7. Full gate (ruff + mypy --strict + pytest + bandit)

Each step keeps the gate green before the next starts.

## 6. Out of scope (deferred to Phase 5+)

- Applying / merging diffs
- Running tests against the proposed diff
- Multi-turn coding with tool use (read_file / write_file)
- Web/Telegram review UI for diffs
- Auto-rebasing drafts when underlying files change
- Cross-machine draft sync
