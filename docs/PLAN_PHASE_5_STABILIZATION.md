# Phase 5 — Stabilization & Acceleration

> Status: **Approved 2026-04-18**. Builds on
> `PLAN_PHASE_4_CODING_HARNESS.md`. Adds the two halves that close
> the improvement loop:
>
> - **Track A — Close the Loop:** apply approved drafts onto an
>   isolated branch, run the test suite, record the verdict.
> - **Track B — Optional Acceleration:** opt-in skill-aware context
>   and a single critique-then-revise pass to improve draft quality.
>
> The runtime is still never modified by the runtime. A human still
> decides whether anything ships to `main`.

## 1. Goal

Phase 4 left us with `.patch.md` files and zero feedback on whether
they actually work. Phase 5 wires the two missing edges:

1. **A — Apply & Verify**: `make apply CT=CT-001` → fresh branch
   `aegis/CT-001-<imp>` → `git apply --check` → `git apply` → run
   `make test` with a timeout → write `.result.md` next to the patch
   → record outcome in `IMPROVEMENTS_DECISIONS.md` → leave the branch
   intact for human review. **No push, no merge, no rebase.**
2. **B — Smarter drafts**: `make harness --with-context` reads each
   in-scope file (capped) plus its skill YAML, then runs a one-shot
   critique pass (`draft → critique → revise`) before writing the
   patch. Default behaviour is unchanged — the flag is opt-in.

End-state demo:

```text
$ make apply CT=CT-001
[apply] CT-001 IMP-a86b087a — patch ~/.aegis/workspace/coding_harness/diffs/CT-001__…patch.md
[apply] working tree clean ✓
[apply] creating branch aegis/CT-001-a86b087a from HEAD
[apply] git apply --check ✓
[apply] git apply ✓ (3 files changed, 47 insertions, 12 deletions)
[apply] running: make test (timeout 300s)
[apply] tests passed ✓ (98 passed, 0 failed) in 41s
[apply] wrote ~/.aegis/workspace/coding_harness/diffs/CT-001__…result.md
[apply] decision logged: applied_clean
[apply] branch aegis/CT-001-a86b087a is ready for human review.
        Next: `git diff main...aegis/CT-001-a86b087a` then merge or `git branch -D` to discard.
```

```text
$ make harness ARGS="--task CT-002 --with-context"
[harness] CT-002 IMP-7a3014d7 — context-mode ON
[harness] gathered 2 in-scope files (1.4 KB) + 1 skill YAML (0.6 KB)
[harness] draft → critique → revise (2 model calls)
[harness] wrote ~/.aegis/workspace/coding_harness/diffs/CT-002__…patch.md
```

## 2. Non-negotiables

1. **Plane isolation preserved** —
   `runtime/coding_harness/applier.py` may shell out to `git` and
   `make` via `subprocess` but MUST refuse to operate on
   `main` / `master` / `staging`, MUST refuse if the working tree is
   dirty, and MUST NOT run with `--no-verify`, `--force`, or
   `--no-edit`. It NEVER calls `git push`.
2. **Branch-only, atomic** — apply lands on a fresh branch named
   `aegis/<CT-id>-<imp-id>`. If `git apply --check` fails, no branch
   is created. If `git apply` fails midway, the branch is left in the
   broken state with `.result.md` flagged `apply_conflict` (a human
   inspects and decides; we do NOT auto-reset).
3. **Test runner is opaque** — Phase 5 invokes `make test` with a
   timeout (default 300 s, override via `MAKE_TEST_TIMEOUT_SECS`) and
   reads its exit code + captured stdout/stderr. We do NOT parse
   pytest output; the verdict is `pass` (exit 0) / `fail` (non-zero)
   / `timeout`. Captured output is truncated at 8 KB into the
   `.result.md` (head + tail).
4. **One CT at a time** — `make apply` accepts exactly one
   `CT=CT-NNN`. No batch-apply (sequential conflicts get ugly fast).
5. **Decision log gets a new verdict, not a status field** —
   re-uses the existing `Decision` envelope from Phase 3. New
   verdicts: `applied_clean`, `applied_test_failed`,
   `apply_conflict`, `reverted`. Latest-decision-wins still applies.
   This keeps the audit trail in one append-only file.
6. **Track B is opt-in** — `--with-context` flag, default OFF.
   When OFF, the Phase 4 prompt and behaviour are bit-for-bit
   identical (existing tests still pass unchanged).
7. **Critique loop is bounded** — exactly one revise pass. No
   recursion, no convergence loop, no agentic exploration.
8. **Hard canon refusal still applies** — both pre-call (CT scope)
   and post-call (diff regex) checks from Phase 4 run on the
   *revised* draft. A revise that introduces a canon write is
   refused just like a draft that does.
9. **No Telegram, no auto-merge, no auto-revert** — Phase 5 leaves
   the branch in place. The human merges (PR, fast-forward, whatever)
   or runs `git branch -D aegis/CT-001-…` to discard.
10. **All boundaries Pydantic-validated** (`extra="forbid"`,
    `frozen=True` on inputs).

## 3. Deliverables (Track A — Close the Loop)

### 3.1 Applier (`runtime/coding_harness/applier.py`)

| Symbol | Role |
| --- | --- |
| `ApplyOutcome` (Pydantic) | `ct_id`, `imp_id`, `branch`, `verdict: Literal["applied_clean","applied_test_failed","apply_conflict","precondition_failed"]`, `reason: str`, `tests_exit_code: int \| None`, `tests_duration_s: float \| None`, `tests_stdout_tail: str` (≤8 KB), `applied_at: datetime`, `patch_path: str` |
| `apply_patch(repo_root, draft_path, *, runner, clock) -> ApplyOutcome` | Pure function: validates preconditions, creates branch, runs `git apply --check` then `git apply`, runs tests, returns outcome. NEVER raises on subprocess failure (returns `ApplyOutcome` with the appropriate verdict). |
| `_extract_unified_diff(patch_md_text) -> str \| None` | Parses the `## Unified diff` fenced block out of the patch.md. Returns None for stub/refused patches. |
| `_check_preconditions(repo_root) -> str \| None` | Returns reason string (working tree dirty / on protected branch / not a git repo) or None when safe. |

The `runner` parameter is a thin adapter so tests can inject a
fake subprocess executor. Default = `subprocess.run` with
`shell=False`, `text=True`, `check=False`, capped timeouts. Adapter
also handles the `make test` invocation.

Protected branches default to `("main", "master", "staging")`.
Override via `AEGIS_PROTECTED_BRANCHES` env var (comma-separated).

### 3.2 Result writer (`runtime/coding_harness/result_writer.py`)

| Symbol | Role |
| --- | --- |
| `result_filename(outcome) -> str` | `CT-NNN__IMP-xxxxxxxx__YYYY-MM-DDTHHMMZ.result.md` (mirrors `patch_filename`) |
| `write_result(workspace, outcome) -> Path` | Renders markdown + writes |
| `latest_result_for(workspace, ct_id) -> Path \| None` | For idempotency / status display |

Result markdown layout:

```markdown
# CT-001 — IMP-a86b087a (verdict: applied_clean)

- **Applied:** 2026-04-19T08:30Z
- **Branch:** aegis/CT-001-a86b087a
- **Patch:** CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md
- **Tests:** make test exit 0 in 41.2s

## Test output (last 8 KB)
```text
…tail of make test output…
```

## Next steps
- Review: `git diff main...aegis/CT-001-a86b087a`
- Ship:   merge / PR the branch
- Discard: `git branch -D aegis/CT-001-a86b087a`
```

`apply_conflict` and `applied_test_failed` variants get a clearly
labelled error block at the top.

### 3.3 Decision log extension

`runtime/improvement/decisions.py` `Verdict` literal grows:

```python
Verdict = Literal[
    "approve", "reject", "defer",                 # Phase 3
    "applied_clean", "applied_test_failed",       # Phase 5 (NEW)
    "apply_conflict", "reverted",                 # Phase 5 (NEW)
]
```

`record_decision` is unchanged. The CLI maps applier verdicts onto
this enum and appends one decision per apply. `latest_by_imp` keeps
its semantics (the apply outcome supersedes the prior `approve`).

### 3.4 CLI driver (`runtime/coding_harness/apply_cli.py`)

```text
python -m runtime.coding_harness.apply_cli CT-001
python -m runtime.coding_harness.apply_cli CT-001 --dry-run    # checks only, no branch
python -m runtime.coding_harness.apply_cli CT-001 --no-tests   # apply + log only
python -m runtime.coding_harness.apply_cli --status CT-001     # show latest .result.md
```

Exit codes: `0` on `applied_clean`, `1` on any other verdict
(non-zero so CI / shell users notice).

### 3.5 Makefile target

```makefile
apply:  ## Apply ONE drafted CT onto a fresh branch and run the test suite
	$(PY) -m runtime.coding_harness.apply_cli $(CT) $(ARGS)
```

Usage: `make apply CT=CT-001`, optional `ARGS=--dry-run`.

### 3.6 Doctor extension

Two new rows under `services:`:

| Label | Severity if missing |
| --- | --- |
| `git:available` | error (apply impossible without it) |
| `git:repo_clean` | warn (apply will refuse but doctor itself still passes) |

Runs `git --version` and `git status --porcelain` (latter only if
the cwd is inside a repo).

### 3.7 Event types

`runtime/events/stream.py` adds:

```python
APPLY_PRECHECK     = "apply.precheck"      # ok | dirty | protected_branch | not_a_repo
APPLY_DIFF_CHECK   = "apply.diff.check"    # ok | rejected
APPLY_DIFF_APPLY   = "apply.diff.apply"    # ok | conflict
APPLY_TESTS        = "apply.tests"         # ok | failed | timeout | skipped
APPLY_DECISION     = "apply.decision"      # logged verdict
```

### 3.8 Workspace layout (additive)

```text
~/.aegis/workspace/coding_harness/diffs/
├── CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md     # Phase 4
└── CT-001__IMP-a86b087a__2026-04-19T0830Z.result.md    # Phase 5 (NEW)
```

Patches and results sit side-by-side (no separate `results/` dir —
keeps the audit trail co-located).

## 4. Deliverables (Track B — Optional Acceleration)

### 4.1 Context gatherer (`runtime/coding_harness/context.py`)

| Symbol | Role |
| --- | --- |
| `gather_context(repo_root, scope_paths, *, max_total_bytes=15360) -> ContextBundle` | Reads each in-scope file (capped at 4 KB each) + the skill YAML for any skill name in `available_skills` whose YAML touches a scope path. Total bundle size capped at 15 KB. |
| `ContextBundle` (Pydantic) | `files: list[FileSlice]`, `skills: list[SkillSlice]`, `truncated: bool`, `total_bytes: int` |
| `FileSlice` / `SkillSlice` | `path`, `content` (≤4 KB), `bytes_total`, `was_truncated` |

The gatherer is read-only and refuses paths outside the repo root
(no `..` escape). Files larger than the cap are head-truncated with
a `[truncated …]` marker.

### 4.2 Critique pass (`runtime/coding_harness/critic.py`)

| Symbol | Role |
| --- | --- |
| `critique_then_revise(draft, task, context, *, client, model, prompt_paths) -> Draft` | One critique call (returns short list of issues) + one revise call (returns a new `_LLMReply`). Returns the revised `Draft`, or the original on any failure (graceful degradation). |
| `_DEFAULT_CRITIC_PROMPT` | `runtime/coding_harness/prompts/critic.txt` (asks: scope creep? canon writes? missing tests? rollback weak?) |
| `_DEFAULT_REVISE_PROMPT` | `runtime/coding_harness/prompts/revise.txt` (asks: re-emit the four-field JSON addressing each critique point) |

Bounded: exactly one critique + one revise. If the critique returns
"no issues", the revise call is skipped and the original draft
stands. Total Track-B model calls per CT: 1 (no flag) → 3 (flag on,
critique finds issues).

### 4.3 Coder integration

`coder.draft_for(...)` gains an optional `context: ContextBundle | None = None`
kwarg. When present, the rendered prompt includes a new
`{context_blob}` placeholder. The fast/no-context path is unchanged.

CLI integration (`coding_harness/cli.py`):

```text
python -m runtime.coding_harness.cli --with-context
python -m runtime.coding_harness.cli --task CT-001 --with-context
```

Default OFF. Emits `pattern.observed` event
`{"pattern": "harness_with_context", "ct_id": ..., "files": N, "skills": M}`
so reflection can later pattern-match on whether context-mode
correlates with `applied_clean` outcomes.

### 4.4 Event types

`runtime/events/stream.py` adds:

```python
HARNESS_CRITIQUE_START = "harness.critique.start"
HARNESS_CRITIQUE_END   = "harness.critique.end"     # status=ok|skipped|failed
HARNESS_REVISE_START   = "harness.revise.start"
HARNESS_REVISE_END     = "harness.revise.end"       # status=ok|failed
```

## 5. Tests (`tests/`)

| File | Coverage |
| --- | --- |
| `test_apply_outcome.py` | `ApplyOutcome` Pydantic validation, verdict literals, length caps |
| `test_apply_preconditions.py` | Refuses on protected branch, dirty tree, non-repo cwd; passes on clean repo at feature branch |
| `test_apply_extract_diff.py` | Parses unified-diff block out of valid patch.md; returns None for stub/refused; rejects malformed fences |
| `test_apply_runner_fake.py` | `apply_patch` with injected fake runner — covers all four verdicts (`applied_clean`, `applied_test_failed`, `apply_conflict`, `precondition_failed`) without touching real git |
| `test_apply_cli_scriptable.py` | `--dry-run`, `--no-tests`, `--status`, unknown CT (exit 1), missing patch (exit 1) |
| `test_apply_cli_e2e.py` | Real `git init`-ed temp repo, real patch, real `git apply`, fake `make test` runner — asserts branch exists + result.md written + decision logged (`@pytest.mark.e2e`) |
| `test_result_writer.py` | Filename format, all four verdict variants render correctly, latest_result_for returns most recent |
| `test_context_gatherer.py` | Cap enforcement (4 KB per file, 8 KB total), `..` rejection, missing files skipped, skills matched by scope intersection |
| `test_critic_revise.py` | Critique returns issues → revise called → revised draft returned; critique returns "no issues" → revise skipped; critique fails → original returned (graceful) |
| `test_coder_with_context.py` | `draft_for(..., context=bundle)` includes context_blob in rendered prompt; without bundle the prompt is bit-identical to Phase 4 |

All marked `@pytest.mark.unit` except the two `_e2e` files.

## 6. Resolved decisions (sign-off 2026-04-18)

| # | Decision | Resolution |
| --- | --- | --- |
| 1 | Apply scope | **Single CT only.** `--all-approved` deferred to Phase 6+. |
| 2 | Test runner | **`make test`** (full gate). Override via `AEGIS_APPLY_TEST_CMD`. |
| 3 | Conflict cleanup | **Leave the half-applied branch.** Failures are evidence. |
| 4 | Critique iterations | **Exactly one revise pass.** Cost-bounded, no recursion. |
| 5 | Context cap | **15 KB total / 4 KB per file** (raised from initial 8 KB). |
| 6 | Apply outcome logging | **Same `record_decision` API**, extended `Verdict` literal. |
| 7 | `--with-context` default | **OFF.** Flip later only when apply-outcome data justifies it. |

## 7. Open questions (historical — for context only)

1. **Apply scope** — accept `make apply CT=CT-001` only, or also
   `make apply --all-approved` (sequential)? I lean **single-CT
   only** — multi-apply needs conflict resolution we don't have.
   `--all-approved` becomes Phase 6 if you want it.
2. **Test-runner choice** — invoke `make test` (current proposal)
   or invoke `pytest` directly? `make test` runs the full gate
   (ruff + mypy + pytest), which catches more regressions but takes
   longer. I lean **`make test`** to match what the contributor sees
   locally; override via `AEGIS_APPLY_TEST_CMD` env var for users
   with a faster gate.
3. **`apply_conflict` cleanup** — leave the half-applied branch for
   human inspection (proposed) or auto-reset and discard the branch?
   I lean **leave it** — `git status` tells the human exactly what
   broke; auto-reset hides the failure mode.
4. **Critique-loop iteration count** — exactly one (proposed) or
   bounded loop until critique returns "no issues" (max 3)? I lean
   **exactly one** — bounded loops are fine in theory, but every
   extra pass is one more frontier-model call and the critic prompt
   itself can hallucinate "issues" indefinitely.
5. **Context gatherer cap** — 8 KB total / 4 KB per file (proposed)
   or scale with the model's context window? I lean **fixed cap** —
   we don't actually need the cheapest tier for coding; we need
   reproducible cost.
6. **Decision verdict overlap** — should `applied_clean` and
   `applied_test_failed` go through the same `record_decision`
   API, or do we add a sibling `record_apply_outcome`? I lean
   **same API** — the audit trail is one append-only log per IMP-id;
   forking it complicates `latest_by_imp` for no clear benefit.
7. **`--with-context` default for new installs** — keep OFF
   forever (proposed) or flip ON in Phase 6 once the
   correlation-with-`applied_clean` data accumulates? I lean
   **decide later, with data**. The pattern.observed event makes
   that decision evidence-based, not vibes-based.

## 8. Build order

Track A and B are independent. Build A first (it's the structural
gap), then B (quality knob). Each step keeps the gate green before
the next starts.

**Track A:**
1. `ApplyOutcome` model + `result_writer.py` + tests
2. `applier.py` precondition checks + `_extract_unified_diff` + tests (no subprocess yet)
3. `applier.py` runner adapter + fake-runner tests covering all verdicts
4. `Verdict` literal extension in `decisions.py` + tests
5. `apply_cli.py` (scriptable mode first, then `--status`) + tests
6. Doctor `git:*` rows
7. Real-git e2e test
8. `Makefile` `apply` target

**Track B:**
9. `ContextBundle` + `context.py` + tests (read-only, no model calls)
10. `coder.draft_for(..., context=...)` integration + prompt template update + tests
11. `critic.py` + critique/revise prompts + tests (with mocked client)
12. `cli.py` `--with-context` flag wiring + tests

**Final:**
13. Full gate (ruff + mypy --strict + pytest + bandit + semgrep)
14. Update `IMPLEMENTATION_PLAN.md` Phase 5 row from ⏳ to ✅

## 9. Out of scope (deferred to Phase 6+)

- Auto-merge to main (will never be in scope; humans only)
- `git push` of the apply branch
- Multi-CT batch apply with conflict resolution
- Auto-revert on test failure
- Re-running tests after a critique loop discovers an issue
- Frontier-model self-critique chains beyond one revise pass
- Telegram review UI for `.result.md`
- Cross-machine apply-result sync
- Pattern detector that uses apply outcomes as a training signal
  (the `pattern.observed` events land in EVENTS.md; Phase 6 reads them)
