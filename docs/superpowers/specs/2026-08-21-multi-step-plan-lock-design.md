# Multi-Step Plan Lock — Design Spec

## Context and motivation

AEGIS's multi-step tool-use loop (`HarnessDispatcher._run_multi_step`,
`runtime/chat/telegram/harness_dispatcher.py`) has no persisted plan object.
Every step, `Tier1Reasoner.plan_next` (`runtime/reasoning/tier1_reasoner.py`)
is called fresh, re-deriving "what's left to do" from the operator's
original request plus the raw `(tool, args) → result` history accumulated
so far this turn. The eval harness (`runtime/eval/`, `eval/tasks/`) measured
this design's actual reliability against AEGIS's own skill catalog and
found small local models fail multi-step chains specifically, even while
handling single-tool tasks reasonably well.

Two prior changes this session narrowed down why, empirically rather than
by argument:

1. **`PlanStep.remaining` (commit 97eb1da).** Added a field to the
   planner's JSON output schema, ordered before `kind` so that — because
   this schema is sent to Ollama as a grammar-constrained `format`
   decode — the model is forced to externalize which tool ids it still
   believes are needed before committing to its next action. Measured
   result against `gemma4:e2b-mlx`: overall TGC 46.7%→60.0%, SGC
   0.0%→33.3% (single-tool tasks improved), but **both genuine
   multi-step tasks (`list_then_read`, `search_then_read`) remained at
   0/2**, unchanged. 4B was unaffected (already near ceiling).

2. **`VariantResult.actual_calls` (commit 78f84e0).** Added raw
   observed-call capture to the eval report, since the JSON gave no way
   to tell *why* the remaining-field change didn't help. Re-running the
   2B suite with this diagnostic revealed three distinct failure
   patterns, not the two originally hypothesized:
   - **Pattern 1 — zero engagement.** `list_then_read`'s variants show an
     *empty* `actual_calls` list; the model never calls any tool at all,
     and `dispatch()` returns `PASS` with no reply sent from the harness
     itself.
   - **Pattern 2 — error regression.** `search_then_read` variant 1:
     `files_search` succeeds, `files_read` then fails on a malformed
     path, and instead of retrying `files_read` with corrected args the
     model calls `files_search` three more times — regressing to repeat
     an already-succeeded step rather than recovering forward.
   - **Pattern 3 — tool substitution under difficulty.** `search_then_read`
     variant 2: after `files_search` succeeds twice, the model calls
     `run_command` (`ls -t <path>`) — a registered-but-unintended tool —
     as an improvised workaround, then returns to `files_search` twice
     more.

A follow-up check (this session, not yet committed to any doc) confirmed
Pattern 1 is *not* a safety gap: when `_run_multi_step` bails to `PASS`,
the turn falls through to `ChatPipeline.turn()`, which independently
re-derives verified tool calls from the evidence ledger and runs the same
`annotate_unverified_claim` gate (`ChatPipeline._gate_reply`,
`runtime/chat/pipeline.py:221-246`) on whatever reply it generates. A
hallucinated completion claim in the Pattern-1 fallback path would still
be caught there. Pattern 1 is therefore a capability/completeness problem
with an existing backstop, not an unguarded safety hole — real, but a
different kind of problem from Patterns 2 and 3, which happen *after* the
model has already started executing and are specifically about a plan
going wrong mid-flight.

## Scope

**In scope:** Patterns 2 and 3 — constraining what the model can do once
multi-step execution has already started, so it cannot regress to a
completed step or substitute an unplanned tool.

**Explicitly out of scope:** Pattern 1 (zero engagement). It shares no
mechanism with a plan-enforcement fix — there is no plan to enforce
against a model that never proposes one — and is tracked as a separate,
later follow-up. The eval suite's `list_then_read` task is Pattern-1-shaped
and is **not expected to improve** from this change; a flat result there
must not be read as this design failing.

## Approaches considered

1. **Dedicated upfront planning call.** A new LLM round-trip, before any
   execution, that commits to an ordered tool-id plan. Cleaner separation
   of concerns, but adds latency/cost per multi-step turn on
   latency-constrained hardware, and — since Patterns 2/3 only matter
   after step 1 has already succeeded — a dedicated call wouldn't know
   anything the harness doesn't already have from the real first
   `plan_next` response. Rejected in favor of reuse.
2. **Reuse the existing first-step `remaining` output as the plan, with
   per-step tool restriction (chosen).** No new LLM call. Uses
   already-shipped infrastructure (`PlanStep.remaining`) for a second
   purpose it was partly designed to enable.
3. **Soft reminder only (no schema restriction).** Inject an explicit
   "already done X, current step Y" note into the prompt without
   narrowing the tool schema. Same category of fix as the `remaining`
   field already shipped and measured — a nudge the model can still
   ignore. Rejected: the `remaining` experiment already showed nudges
   don't fix the specific multi-step failures; a stronger nudge alone is
   not expected to behave differently.
4. **Full plan lock, no early bail** (considered as the strict end of
   approach 2's spectrum, rejected as the default). Never offering
   `respond`/`task_complete` until the plan is exhausted guarantees the
   plan runs to completion or fails via the step cap, but risks forcing a
   doomed tool call in genuinely ambiguous cases (e.g., a search that
   found nothing) instead of allowing an honest "nothing found" reply.
   Rejected as too rigid; `respond`/`task_complete` stay available at
   every step in the chosen design.

## Design

### Plan derivation

Entirely local to `HarnessDispatcher._run_multi_step`'s existing loop
(`runtime/chat/telegram/harness_dispatcher.py`). No new files, no change
to `Tier1Reasoner.plan_next`'s signature — the mechanism works by
narrowing what `_run_multi_step` passes as the *existing*
`available_skills` parameter, not by changing that parameter's shape.
`_build_planner_schema` (`runtime/reasoning/tier1_reasoner.py`) already
derives its `tool`/`remaining` enums from whatever `available_skills` it
receives, so no changes are needed there either.

After step 1's `plan_next` call returns `kind="tool_call"`:

```python
plan_ids = [plan.tool] + [t for t in plan.remaining if t != plan.tool]
cursor = 0
```

`plan.tool` leads the list regardless of whether the model's own
`remaining` output included it — a defensive normalization, since nothing
enforces that a model's self-reported `remaining` is internally
consistent with its own `tool` choice.

### Cursor advancement and tool restriction

On every `plan_next` call from step 2 onward, while `cursor < len(plan_ids)`,
the `available_skills` argument passed to `plan_next` is narrowed to just
the `SkillDescriptor` whose `.tool == plan_ids[cursor]`, not the full
registry. This alone makes an unplanned tool (e.g. `run_command`)
unreachable — absent from the schema's tool enum entirely, not merely
discouraged by prompt text (Pattern 3). Verified against the live catalog
(`~/.aegis/workspace/skills/`) that tool ids are unique per skill (13
skills, 13 unique `.tool` values) — the narrowed lookup always resolves to
exactly one descriptor, never zero or more than one.

The cursor advances only when the tool actually executed at
`plan_ids[cursor]` completes without error (same status convention used
elsewhere in this file, e.g. `verdict_for_result`). On success,
`cursor += 1`. If that reaches `len(plan_ids)`, the plan is exhausted and
the *next* `plan_next` call reverts to the full, unrestricted
`available_skills` — matching today's existing behavior once a plan
completes, since the model should then be free to decide it's done or
that something beyond the original plan is still needed.

On failure, the cursor does not advance, and the harness keeps offering
only that same restricted tool on the next call. The error is already
threaded into `call_history` via the existing mechanism, so the model
sees what went wrong and can retry the same step with corrected
arguments — but cannot regress to a prior, already-succeeded step
(Pattern 2), because that tool is not in the schema on this call.

`respond` and `task_complete` remain available in the schema at every
step regardless of restriction — the model can always terminate the
chain gracefully rather than being forced through a step that no longer
makes sense (e.g., a search that returned nothing).

### Escape valve

No new retry-count parameter. A step that keeps failing simply exhausts
the loop's existing `self._max_steps` cap, terminating with whatever
partial history was accumulated — identical to today's behavior when a
turn runs out of steps. Introducing a second, independent cap here would
risk disagreeing with the existing one; reusing it keeps termination
semantics singular.

### Accepted consequence

Once `available_skills` is narrowed to one descriptor, `remaining`'s
enum (derived from `available_skills` the same way `tool`'s is) is also
constrained to that one tool id or empty — its role narrows from
"self-reported full remaining plan" (meaningful on step 1) to
effectively "am I done with this one step" on restricted steps. This is
an accepted, unavoidable consequence of the restriction, not a defect.

## Testing

**Unit** (`tests/test_harness_dispatcher.py`, extending the existing
stub-planner multi-step tests):
- Step 2's `available_skills` is restricted to the single planned tool
  after step 1 succeeds.
- A failing step 2 keeps that same single-tool restriction on step 3
  (does not revert to full catalog, does not advance the cursor).
- A successful, fully-consumed plan reverts to full `available_skills` on
  the next call after the plan completes.
- Defensive normalization: a step-1 `remaining` that omits the model's own
  `tool`, or is empty, still produces a `plan_ids` list led by the actual
  tool called.

**Live validation:** re-run `runtime/eval/cli.py` against
`gemma4:e2b-mlx` and `gemma4:e4b-mlx`.
- Target: `search_then_read` specifically. Success is `actual_calls` in
  the resulting JSON no longer showing a repeated already-succeeded step
  (Pattern 2) or an unplanned tool such as `run_command` (Pattern 3).
- `list_then_read` is not expected to move (out of scope, Pattern-1-shaped)
  — a flat result there is not a failure of this design.
- No regression on any currently-passing task at 4B.

## Known limitation (measured after implementation)

The lock's efficacy is entirely gated on the model's step-1 `remaining`
self-report being complete. `plan_ids` is derived from `plan.tool` plus
whatever `plan.remaining` names at step 1; if the model omits a tool it
will in fact call later, that tool is simply absent from `plan_ids`, and
once the one entry the model *did* self-report is consumed, `cursor`
reaches `len(plan_ids)` immediately — the restriction never engages for
the rest of the turn, and Patterns 2/3 (the exact failures this design
targets) can reappear unconstrained.

This was measured directly on `search_then_read` variant 2 against
`gemma4:e2b-mlx` (Task 1 report, root-cause trace): step 1 returned
`tool=files_search remaining=[files_search]` — self-reporting no future
work even though the task needs a subsequent `files_read`. `plan_ids`
therefore collapsed to `["files_search"]`, and the instant that step
succeeded `cursor` reached `len(plan_ids)`, so step 2 onward saw the full,
unrestricted catalog again — the model went on to call `run_command` as
an improvised substitute (Pattern 3) and repeat `files_search` after a
later `files_read` error (Pattern 2). By contrast, variant 1's step 1
returned `remaining=[files_search, files_read]`, `plan_ids` correctly
carried both tools, and the restriction held through a failure exactly as
designed (matches `test_multi_step_restriction_persists_after_step_failure`).

This isn't a new failure mode — it's the same unreliable signal that
motivated `PlanStep.remaining` itself (commit 97eb1da) surfacing at a
different layer: that change already measured the field's self-report as
an incomplete predictor of the model's actual future behavior at 2B
(overall TGC/SGC improved, but genuine multi-step tasks stayed flat). This
design's restriction mechanism inherits that same unreliability as its
own upper bound — it can only be as complete as the self-report it is
built from. No fix is proposed here; a dedicated upfront planning call
(rejected above as approach 1) or a different plan-derivation signal
entirely would be needed to close this, and either is future work.

## Non-goals

- Fixing Pattern 1 (zero engagement) — tracked separately.
- A dedicated upfront planning call — rejected above.
- Any change to `Tier1Reasoner.plan_next`'s signature, `PlanStep`'s shape,
  or `_build_planner_schema` — all reused as-is.
- Persisting the plan across turns, or exposing it outside
  `_run_multi_step` — it is turn-local, matching the existing stateless
  design.
