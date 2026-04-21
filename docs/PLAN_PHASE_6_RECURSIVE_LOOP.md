# Phase 6 — Recursive Loop & Self-Cleanup

> Status: **Drafted 2026-04-18**. Builds on
> `PLAN_PHASE_5_STABILIZATION.md`. Closes the two gaps Phase 5 left
> open by design:
>
> - **Track A — Apply outcomes as a pattern signal.** Plane 2
>   (Reflection) starts reading the `governance.decision` and
>   `harness.refused` events that Plane 3 (Coding Harness) already
>   emits. Repeat failures become first-class `PatternRecord`s —
>   feeding the same proposal pipeline as runtime patterns.
> - **Track B — Auto-revert on failure.** Plane 3 stops leaving the
>   runner stranded on a broken apply branch. On
>   `applied_test_failed` / `apply_conflict`, the applier checks out
>   the original branch and deletes the apply branch.
>
> The runtime is still never modified by the runtime. A human still
> decides whether anything ships to `main`. We are wiring outcomes
> back into the loop, not unlocking auto-merge.

## 1. Goal

Phase 5 produced two unused signals:

1. Every `make apply` writes a `governance.decision` event with the
   verdict. Nothing reads them.
2. Every `apply_test_failed` / `apply_conflict` leaves an
   `aegis/CT-NNN-<imp>` branch behind in a state the human probably
   doesn't want to inspect (the test failure is in `.result.md`
   already; the branch is just clutter).

Phase 6 fixes both — independently, in one ship.

End-state demos:

```text
# Track A — outcome-driven detector
$ make reflect
[reflect] reading window since 2026-04-12 (90 sessions, 1241 events)
[reflect] patterns:
  - unknown_intent (medium) ×4
  - apply_failed_repeat (high) ×1   # NEW
        IMP-a86b087a: 3 non-clean apply outcomes in 6 attempts.
        sample: aegis/CT-001-a86b087a, aegis/CT-001-a86b087a-2
[reflect] wrote PATTERNS.md
[reflect] drafted 1 new proposal (P-007: revisit IMP-a86b087a scope)
```

```text
# Track B — auto-revert
$ make apply CT=CT-014
[apply] CT-014 — patch CT-014__IMP-a13f0042__2026-04-19T1422Z.patch.md
[apply] git apply --check ✓
[apply] creating branch aegis/CT-014-a13f0042 from feature/intent-aliases
[apply] git apply ✓ (1 file changed, 4 insertions, 0 deletions)
[apply] running: make test (timeout 300s)
[apply] tests FAILED ✗ (3 failed, 95 passed) in 38s
[apply] auto-revert: checking out feature/intent-aliases ✓
[apply] auto-revert: deleted branch aegis/CT-014-a13f0042 ✓
[apply] wrote ~/.aegis/workspace/coding_harness/diffs/CT-014__…result.md
[apply] decision logged: applied_test_failed
[apply] (test output captured in .result.md; branch reverted)
```

## 2. Non-negotiables

1. **No new write paths.** Phase 6 reads existing `governance.decision`
   / `harness.refused` events and writes the same `PATTERNS.md` /
   `PROPOSALS.md` files. No new event types, no new log files.
2. **Auto-revert never destroys uncommitted work.** It runs only
   *after* `apply_patch()` confirmed a clean tree at start (a Phase 5
   precondition). The reverted branch was created during this run —
   nothing pre-existing is touched.
3. **Auto-revert is opt-out, not opt-in.** Default `--auto-revert`
   ON. `--no-revert` preserves Phase 5 behaviour for users who want
   to inspect the broken branch by hand.
4. **Auto-revert never runs on `precondition_failed`.** That verdict
   means no branch was created — there is nothing to revert.
5. **Detectors stay deterministic.** Same shape as the existing five:
   pure functions over `Event` lists, threshold-gated, no LLM, no
   payload bodies in `summary`.
6. **`PatternRecord` schema unchanged.** No new severity levels, no
   schema migration. New detectors fit the existing envelope.
7. **Recursion bound is 1.** Phase 6 closes the *outcome → pattern →
   proposal* loop once. Phase 7+ may consume the resulting proposals
   through the existing `improvement` CLI. No reflection-on-reflection.
8. **Plane isolation preserved.** Plane 2 (Reflection) gains read
   access to `governance.decision` / `harness.refused` payloads
   (which it can already see — they're in the same `EventStream`).
   It does NOT import from `runtime/coding_harness/` or
   `runtime/improvement/`. Detectors stay in `runtime/reflection/`.
9. **Hard canon refusal still applies.** Auto-revert is a `git`
   subprocess sequence with the same Phase 5 discipline: never
   `--no-verify`, never `--force`, never `push`.
10. **All boundaries Pydantic-validated** (`extra="forbid"`,
    `frozen=True` on inputs).

## 3. Deliverables (Track A — Outcome-driven detectors)

### 3.1 New detectors in `runtime/reflection/patterns.py`

| Detector | Severity | Trigger | Reads |
| --- | --- | --- | --- |
| `apply_failed_repeat` | high | Same `imp_id` has ≥2 non-clean apply verdicts (`applied_test_failed` ∪ `apply_conflict` ∪ `reverted`) within the window | `governance.decision` events |
| `harness_refused_repeat` | medium | Same `imp_id` has ≥2 `harness.refused` events (model keeps trying scope/canon writes) | `harness.refused` events |
| `context_mode_helps` | low | Same `imp_id` has both a `harness_with_context` event AND a subsequent `applied_clean` decision | `pattern.observed` (where `pattern == "harness_with_context"`) + `governance.decision` |

**Why these three:**

- `apply_failed_repeat` is the killshot signal — drafts for this CT
  are not converging. The proposal model should escalate scope or
  re-decide.
- `harness_refused_repeat` says the model is misreading the CT
  scope. Either the CT is mis-scoped, or the prompt is too vague.
- `context_mode_helps` is the *positive* signal Phase 5 left a TODO
  for: "decide later, with data" (Decision #7). This detector
  generates the data.

`detect_all` gains the three new entries:

```python
out.extend(detect_apply_failed_repeat(materialized))
out.extend(detect_harness_refused_repeat(materialized))
out.extend(detect_context_mode_helps(materialized))
```

Constants:

```python
_APPLY_FAIL_THRESHOLD = 2     # 2 non-clean attempts on same imp = "stuck"
_REFUSE_THRESHOLD     = 2
_CONTEXT_HELP_MIN     = 1     # any positive evidence is worth surfacing
```

`summary` strings stay ≤240 chars and never include payload bodies —
just `imp_id`, count, and a one-line description (matches existing
detectors).

### 3.2 Proposal templates

`runtime/reflection/proposals.draft()` already runs one LLM call per
`PatternRecord`. No code changes needed — but the prompt template
already accepts `detector` and `summary`, so the new detectors flow
through without modification. We DO add three short detector
descriptions in `runtime/reflection/prompts/proposal.txt` so the
model knows what each new pattern means (one line each, no other
prompt changes).

If the prompt file doesn't have a detector glossary today, we add
one as an additive section — it's a documentation aid, not a
behavioural change for the existing five detectors.

### 3.3 Tests

| File | Coverage |
| --- | --- |
| `tests/test_patterns_apply.py` | All three new detectors: positive case (threshold met), negative case (one-off failure), boundary case (exactly threshold), empty-stream case |
| `tests/test_patterns_apply.py::test_no_payload_leak` | Asserts `summary` never contains rationale text or test stdout |

Reuses the existing `tests/test_patterns.py` fixtures + the
`Event(type=..., session_id=..., payload=...)` factory — no new
helpers.

## 4. Deliverables (Track B — Auto-revert)

### 4.1 Capture original branch before checkout

`runtime/coding_harness/applier._check_preconditions` already calls
`git rev-parse --abbrev-ref HEAD` to get the current branch (so it
can refuse on protected branches). Today it returns only a reason
string. We refactor it to return `(reason, current_branch)`:

```python
def _check_preconditions(repo_root: Path, *, git: Runner) -> tuple[str | None, str]:
    """Return (reason, current_branch). reason=None when safe to apply."""
```

Internal-only refactor (helper has a `_` prefix). No public API
change. Existing tests update to unpack the tuple.

### 4.2 Auto-revert in `apply_patch`

`apply_patch()` gains one parameter:

```python
def apply_patch(
    repo_root: Path,
    patch_path: Path,
    *,
    runner: Runner,
    clock: Callable[[], datetime],
    run_tests: bool = True,
    test_timeout_secs: float = 300.0,
    auto_revert: bool = True,           # NEW — default ON
) -> ApplyOutcome:
```

Behaviour additions:

- Capture `original_branch` from preconditions.
- On `applied_test_failed` or `apply_conflict` *after* the apply
  branch was created (i.e., after `checkout -b` succeeded), and
  when `auto_revert=True`:
  1. `git -C <repo> checkout <original_branch>` — undoes any
     uncommitted apply changes by switching branches (the apply
     branch retains them).

     Wait — `git checkout <other-branch>` refuses if the working
     tree has unstaged changes that conflict. Since `git apply`
     unstaged the changes, we MUST first `git -C <repo> reset
     --hard HEAD` on the apply branch to discard the apply, then
     checkout. The apply branch still exists at the point of
     creation (HEAD), so deleting it is safe.

     Order: `reset --hard HEAD` → `checkout <original>` →
     `branch -D <apply_branch>`.
  2. If any of those fail, the verdict is unchanged but the
     `reason` field gets a suffix: `; auto-revert failed: <stderr>`.
     We DO NOT change the verdict to mask the test failure. The
     human can clean up by hand.
- On `applied_clean` we never revert (success keeps the branch).
- On `precondition_failed` we never revert (no branch was created).
- On `auto_revert=False` we preserve Phase 5 behaviour exactly
  (used by existing tests + by users who want forensic state).

The revert sequence is wrapped in a small private helper:

```python
def _revert_apply(
    repo_root: Path, *, git: Runner, original_branch: str, apply_branch: str
) -> str | None:
    """Return None on success, error string on partial failure."""
```

### 4.3 CLI flag in `apply_cli.py`

```python
p.add_argument(
    "--no-revert",
    action="store_true",
    help=(
        "Do NOT auto-revert on test failure or apply conflict. "
        "Leaves the broken branch checked out for forensic inspection."
    ),
)
```

`run()` passes `auto_revert=not args.no_revert` through.

`_print_outcome()` gains two lines on revert success:

```text
[apply] auto-revert: checking out <original> ✓
[apply] auto-revert: deleted branch <apply_branch> ✓
```

…or one line on revert failure (which is a soft warning, not an
error):

```text
[apply] auto-revert: FAILED — <reason> (branch left in place for inspection)
```

### 4.4 Tests

| File | Coverage |
| --- | --- |
| `tests/test_applier_auto_revert.py` | `applied_test_failed` with `auto_revert=True` triggers reset/checkout/branch -D in correct order; `auto_revert=False` skips revert (Phase 5 baseline); `apply_conflict` after branch creation triggers revert; `apply_conflict` before branch creation skips revert; `precondition_failed` never reverts; revert helper failure surfaces in `reason` but verdict is unchanged |
| `tests/test_apply_cli_scriptable.py` (extend) | `--no-revert` flag plumbing — assert `auto_revert=False` reaches `apply_patch` |
| `tests/test_apply_cli_e2e.py` (extend) | Real-git: failing test → branch deleted, working tree returned to original branch, `git status` clean |

Uses the existing `ScriptedRunner` from `applier.py` for unit tests.

## 5. Resolved decisions (sign-off pending)

| # | Decision | Resolution |
| --- | --- | --- |
| 1 | Detector window | **Same window as runtime detectors.** Reuse `read_window(since=today)` — no special multi-day window for outcome detectors. |
| 2 | `apply_failed_repeat` threshold | **2 non-clean outcomes.** Three would let a stuck CT cycle for a long time before flagging. |
| 3 | `context_mode_helps` symmetry | **Only positive evidence emitted.** Negative evidence (`with_context` + `applied_test_failed`) is already covered by `apply_failed_repeat`. |
| 4 | Revert default | **ON.** Users who want forensic state opt out; the common case is "I ran apply, tests failed, I want my repo back." |
| 5 | Revert on `precondition_failed` | **Never.** No branch was created. |
| 6 | Revert helper failure mode | **Soft warning.** Don't change the verdict; the test failure IS the headline. |
| 7 | New event types | **None.** Track A reads only existing events. Track B writes nothing new — just additional console output. |

## 6. Build order

Track A and B are independent. Build B first (it's the urgent
ergonomics fix — anyone running `make apply` today gets stranded on
broken branches). Then A (the loop closure that compounds over time).

**Track B (auto-revert):**

1. Refactor `_check_preconditions` to return `(reason, branch)` —
   update existing tests.
2. Add `_revert_apply` helper + `auto_revert` parameter to
   `apply_patch()`.
3. Add `--no-revert` flag + plumbing in `apply_cli.py`.
4. Unit tests with `ScriptedRunner` covering all verdicts × revert
   on/off matrix.
5. E2E test — real `git init`, real failing patch, assert clean
   tree afterwards.

**Track A (outcome detectors):**

6. Add three new detectors to `runtime/reflection/patterns.py`.
7. Wire into `detect_all`.
8. Unit tests for each detector.
9. (Optional, additive) Update `proposal.txt` glossary section.

**Final:**

10. Full gate (`make test` — ruff + bandit + mypy + 237+ unit + 12 e2e).
11. Update `docs/IMPLEMENTATION_PLAN.md` Phase 6 row from absent to ✅.

## 7. Out of scope (deferred to Phase 7+)

- Telegram surface for reviewing patterns / proposals (Phase 7).
- Auto-merge of `applied_clean` branches (will never be in scope).
- Cross-machine sync of apply outcomes.
- Detectors that span multiple windows (multi-day trend analysis).
- Negative-evidence variant of `context_mode_helps` (covered by
  `apply_failed_repeat`; an explicit `context_mode_hurts` would
  double-count).
- Auto-reverting an `applied_clean` branch when a *later* signal
  invalidates it (recursion bound: 1).
- Reflection feedback into the harness prompt itself
  (proposal-driven prompt mutation is a separate, riskier track).
