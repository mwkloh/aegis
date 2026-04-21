# Phase 3 — Human Approval Loop (Improvement Plane)

> Status: **Draft, awaiting sign-off**. Builds on
> `PLAN_PHASE_2_REFLECTION.md`. Adds the third cognitive plane:
> **govern** the proposals Reflection drafted. **Still no execution.**

## 1. Goal

Phase 2 produced `PROPOSALS.md` — candidate changes derived from observed
patterns. Phase 3 lets a human walk those proposals, decide
*approve / reject / defer*, and writes a tamper-evident `DECISIONS.md`.
Approved proposals are queued as `CODING_TASKS.md` entries — but
nothing is built or merged. Phase 4 (coding harness) consumes the queue.

End-state demo:

```text
$ make review
[improvement] loaded 2 proposals from reflection/PROPOSALS.md
[improvement] 2 pending  (0 approved, 0 rejected, 0 deferred)

──────────────────────────────────────────────
IMP-7f3a1c2b — unknown_intent (risk: low)
Affected:  intent_classifier
Change:    Increase the number of training samples for...
Rationale: The pattern record indicates underfitting...
──────────────────────────────────────────────
[a]pprove / [r]eject / [d]efer / [s]kip / [q]uit > a
rationale> seems safe; matches what we saw in the demo
[improvement] recorded approve  → CT-001 queued

...
[improvement] wrote ~/.aegis/workspace/improvement/DECISIONS.md
[improvement] wrote ~/.aegis/workspace/improvement/CODING_TASKS.md
```

## 2. Non-negotiables

1. **Plane isolation** — `runtime/improvement/` may import from
   `runtime/events/` (write decision events only) and `runtime/config/`.
   It MUST NOT import any harness/tool/skill code, and MUST NOT call
   any LLM. Phase 3 is deterministic, human-driven only.
2. **No execution effects** — approving a proposal queues a
   `CodingTask`; it does not run code, generate diffs, or touch
   runtime canon. Phase 4 owns code generation.
3. **No canon mutation** — never touches `AGENTS.md`, `USER.md`,
   `IDENTITY.md`, `SOUL.md`, `HEARTBEAT.md`, or any catalog skill YAML.
4. **Append-only governance log** — `DECISIONS.md` and `CODING_TASKS.md`
   are append-only. Re-reviewing a proposal updates *forward* (a new
   "supersedes" row), never overwrites history.
5. **Stable proposal IDs across runs** — `IMP-<hash>` derived from
   `(detector, affected, change)`. Same logical proposal across two
   reflection runs maps to the same `IMP-id`, so the user does not
   re-decide the same item.
6. **Idempotent task queueing** — approving a proposal that already has
   a `CT-id` is a no-op (logged, not duplicated).
7. **All boundaries Pydantic-validated** (`extra="forbid"`, `frozen=True`).

## 3. Deliverables

### 3.1 Proposal loader (`runtime/improvement/proposal_loader.py`)

Parses the markdown produced by `runtime/reflection/writer.py`.

| Symbol | Role |
| --- | --- |
| `LoadedProposal` (Pydantic) | `imp_id`, `pattern_detector`, `affected: list[str]`, `change: str`, `risk: Literal["low","medium","high"]`, `rationale: str`, `source_run: str` (timestamp header) |
| `load_proposals(workspace) -> list[LoadedProposal]` | Reads `<workspace>/reflection/PROPOSALS.md`, parses every `### P-NNN` block, deduplicates by content hash → `IMP-<8 hex>` |

`imp_id` derivation (deterministic):

```python
sha256(f"{detector}|{','.join(sorted(affected))}|{change}".encode()).hexdigest()[:8]
imp_id = f"IMP-{hash8}"
```

This means: if Phase 2 drafts the same proposal twice across two runs,
both map to the same `IMP-id`; the user only decides once.

### 3.2 Decision recorder (`runtime/improvement/decisions.py`)

| Symbol | Role |
| --- | --- |
| `Decision` (Pydantic) | `imp_id`, `verdict: Literal["approve","reject","defer"]`, `rationale: str` (max 1 KB), `decided_at: datetime`, `supersedes: str \| None` |
| `load_decisions(workspace) -> list[Decision]` | Reads `<workspace>/improvement/DECISIONS.md`, parses every dated row |
| `record_decision(workspace, decision)` | Appends one markdown row; emits a `governance.decision` event via `EventStream` for the audit trail |

Reviewing the same `IMP-id` again produces a new row with
`supersedes=<prior decided_at>` — earlier rows are kept as history.

### 3.3 Coding-task queue (`runtime/improvement/coding_tasks.py`)

| Symbol | Role |
| --- | --- |
| `CodingTask` (Pydantic) | `ct_id` (`CT-NNN`), `imp_id`, `scope: list[str]` (= proposal `affected`), `constraints: str` (canned: "do not modify canon"), `expected_output: str` (= proposal `change`), `queued_at: datetime` |
| `load_tasks(workspace) -> list[CodingTask]` | Reads existing `<workspace>/improvement/CODING_TASKS.md` |
| `queue_task(workspace, proposal) -> CodingTask \| None` | If no existing CT for this `imp_id`, append a new `CT-NNN`; otherwise return None |

CT numbering is monotonic across runs (highest existing `CT-NNN` + 1).

### 3.4 CLI driver (`runtime/improvement/cli.py`)

```text
python -m runtime.improvement.cli                          # interactive review
python -m runtime.improvement.cli --list                   # show pending only
python -m runtime.improvement.cli --decide IMP-7f3a1c2b \
    --verdict approve --rationale "safe; matches demo"     # scriptable single-decision mode
```

- Pending = proposals whose latest decision is *not* `approve` or `reject`.
  (`defer` is also pending unless `--include-deferred` is omitted.)
- Interactive prompts: `[a]pprove / [r]eject / [d]efer / [s]kip / [q]uit`
  followed by free-text `rationale>`.
- Returns 0 always when the loop exits cleanly; 1 only on I/O / config
  failure or unknown `--decide IMP-id`.

`make review` wraps the interactive form.

### 3.5 Doctor extension

Add `improvement:writable` row that confirms
`~/.aegis/workspace/improvement/` is creatable. Severity = warn (matches
`reflection:writable`).

### 3.6 Workspace layout (additive)

```text
~/.aegis/workspace/
├── reflection/
│   ├── PATTERNS.md          # Phase 2
│   └── PROPOSALS.md         # Phase 2
└── improvement/             # NEW — Phase 3
    ├── DECISIONS.md
    └── CODING_TASKS.md
```

The repo-level `improvement/DECISIONS.md` and
`improvement/CODING_TASKS.md` remain canonical *schemas* (untouched).
The workspace files are the live append-only logs.

### 3.7 Tests (`tests/`)

| File | Coverage |
| --- | --- |
| `test_proposal_loader.py` | Parse Phase 2 markdown, stable `IMP-id` across two identical proposals, ignore malformed sections |
| `test_decisions_log.py` | Append-only, `supersedes` chain when same `IMP-id` decided twice |
| `test_coding_tasks_queue.py` | Monotonic CT-NNN, idempotent on duplicate approve, no canon writes |
| `test_improvement_cli_interactive.py` | `monkeypatch.setattr("builtins.input", ...)` walks one approve, one reject, one defer; verifies decision rows + CT-001 only for the approve |
| `test_improvement_cli_scriptable.py` | `--decide IMP-... --verdict approve` records decision, queues CT, returns 0 |
| `test_improvement_cli_e2e.py` | End-to-end against a fixture `PROPOSALS.md` — `make review` equivalent, exit 0, both files written |

All under `tests/`, marked `@pytest.mark.unit` except the e2e.

## 4. Open questions for sign-off

1. **Decision storage location** — `~/.aegis/workspace/improvement/`
   (proposed) keeps Plane 3 outputs separate from canon, mirroring
   Phase 2's `reflection/` directory. The repo `improvement/*.md` files
   stay as schema templates only. OK?
2. **Stable IMP-id derivation** — `sha256(detector | affected | change)[:8]`
   means rewording a proposal slightly creates a *new* `IMP-id`. That's
   probably correct (different change = different decision), but flag if
   you'd rather hash by `(detector, affected)` only.
3. **Approval auto-queues a CodingTask** — proposal approved →
   `CT-NNN` appended in the same step. Alternative: a separate
   `make queue` step. I lean toward auto-queue for tightness; you can
   always `reject` later, which records a superseding decision but does
   *not* delete the queued CT (Phase 4 will skip CTs whose latest
   decision is `reject`).
4. **Telegram approval** — explicitly **out of scope** for Phase 3.
   Phase 5+ (Optional Acceleration) wraps the same scriptable CLI. OK?

## 5. Build order

1. `proposal_loader.py` + tests
2. `decisions.py` + append-only tests
3. `coding_tasks.py` + idempotency tests
4. `cli.py` (scriptable mode first, interactive second) + tests
5. Doctor `improvement:writable` row
6. `Makefile` `review` target
7. Full gate (ruff + mypy --strict + pytest + bandit)

Each step keeps the gate green before the next starts.

## 6. Out of scope (deferred to Phase 4+)

- Code generation from approved tasks (Phase 4 = coding harness)
- Diff review UI / web dashboard
- Telegram-mediated approval
- Auto-approval policies / risk-based gating
- Cross-machine decision sync
