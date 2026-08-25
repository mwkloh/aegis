# Classification Fallback to the Multi-Step Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When intent classification finds no matching skill, fall through to the multi-step planner with the full skill catalog instead of silently returning `PASS`, so compound "do X then Y" requests stop being completely unhandled.

**Architecture:** A single new branch in `HarnessDispatcher.dispatch()` gated on `descriptor is None`: construct a sentinel `SkillDescriptor`, skip the two checks that only apply to a genuinely classified skill (`has_tool`, confidence threshold), and reuse the existing multi-step chain-handling code unchanged. A new `guard_min_step` parameter on `_run_multi_step` tightens the destructive-tool guard to step 1 for this path only, since there's no classification signal to trust.

**Tech Stack:** Python 3.11, pytest, existing `_StubClassifier`/`_StubPlanRunner`/`_make_loop_dispatcher`/`_two_skill_setup`/`_destructive_setup` test fixtures in `tests/test_harness_dispatcher.py`.

**Spec:** `docs/superpowers/specs/2026-08-21-classification-fallback-design.md`

## Global Constraints

- No changes to the classifier's own prompt/schema (`runtime/intent/classifier.py`) — rejected approach, out of scope.
- The `CLARIFY` path and its confidence threshold stay untouched.
- The single-shot (`multi_step=False`) dispatch path stays untouched — `descriptor is None` still returns `PASS` when multi-step is disabled.
- `_UNCLASSIFIED_DESCRIPTOR` is a private, call-site-local labeling construct — not a registered skill, not exported/imported elsewhere.
- Every existing call site of `_run_multi_step` (there is exactly one, in `dispatch()`'s classified `if self._multi_step:` branch) must be unaffected by the new `guard_min_step` parameter's default.

---

### Task 1: Fallback branch, guard_min_step parameter, and tests

**Files:**
- Modify: `runtime/chat/telegram/harness_dispatcher.py:42-84` (module constants — add sentinel descriptor), `:414-490` (`dispatch()`), `:647-654` (`_run_multi_step` signature), `:760` (destructive-guard check)
- Test: `tests/test_harness_dispatcher.py` (new tests, placed after `test_pass_on_unknown_intent`, which currently ends at line 228)

**Interfaces:**
- Consumes: `SkillDescriptor` (`runtime/skills/registry.py`, already imported in `harness_dispatcher.py`), existing `_run_multi_step`/`_ChainResult`/`DESTRUCTIVE_TOOLS`/`_GUARD_MIN_STEP`.
- Produces: no new public names beyond the module-private `_UNCLASSIFIED_DESCRIPTOR` constant and `_run_multi_step`'s new `guard_min_step` keyword parameter (defaulted, so existing callers are source-compatible).

- [ ] **Step 1: Read the current `dispatch()` and `_run_multi_step` methods in full**

Run: `sed -n '414,495p;647,770p' runtime/chat/telegram/harness_dispatcher.py`

Confirm it matches what this plan assumes: `dispatch()` resolves `descriptor = self._registry.for_intent(intent)` at line 462, returns `PASS` immediately at line 465 when `descriptor is None`; the `has_tool` check is at line 467, the confidence-threshold `CLARIFY` check at line 473; the classified `if self._multi_step:` block starts at line 484. `_run_multi_step`'s signature is at lines 647-654; the destructive-guard check `if plan.tool in DESTRUCTIVE_TOOLS and step_no >= _GUARD_MIN_STEP:` is at line 760. If the file has diverged from this shape, stop and report — this plan's steps assume this exact structure.

- [ ] **Step 2: Write the three new failing tests**

Add these three test functions to `tests/test_harness_dispatcher.py`, directly after `test_pass_on_unknown_intent` (which currently ends at line 228, just before the blank line before `test_pass_when_tool_not_in_harness` at line 231):

```python
async def test_classification_fallback_reaches_planner_with_unclassified_skill_id() -> None:
    """Unknown intent + multi_step=True must reach the planner (not PASS
    immediately), and the resulting tool call must carry the sentinel
    skill_id so the fallback path is distinguishable in recorded calls."""
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        classifier=_StubClassifier("unknown", 0.0),
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="do the compound thing", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert len(runner.plan_next_calls) >= 1
    assert harness.calls[0].tool == "files_search"
    assert harness.calls[0].skill_id == "unclassified_fallback"


async def test_classification_fallback_guards_destructive_tool_at_step_1() -> None:
    """The fallback path has no classification signal at all, so unlike
    normal classified dispatch, a destructive tool is guarded even at
    step 1 -- the operator must confirm before it runs."""
    registry, harness = _destructive_setup("files_delete")
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_delete", args={"path": "/tmp/x"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        classifier=_StubClassifier("unknown", 0.0),
    )
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="delete something", message=message
    )

    assert outcome == DispatchOutcome.FIRED
    assert harness.calls == []  # destructive tool never actually executed
    assert "⚠️ I'd like to run `files_delete`" in message.replies[0]


async def test_multi_step_false_unknown_intent_ignores_fallback() -> None:
    """multi_step=False must keep today's PASS behavior on descriptor is
    None -- the fallback only ever applies to the multi-step path, since
    the single-shot reasoner needs a real descriptor to reason about."""
    dispatcher = _make_dispatcher(classifier=_StubClassifier("unknown", 0.0))
    message = _FakeMessage()

    outcome = await dispatcher.dispatch(chat_id=1, user_text="hmm", message=message)

    assert outcome == DispatchOutcome.PASS
    assert message.replies == []
```

Note: `test_pass_on_unknown_intent` (already in the file, unmodified) and `test_destructive_at_step_1_is_allowed` (already in the file, unmodified) already cover two of the spec's five required test cases as regressions — `test_pass_on_unknown_intent` is the same scenario as the new `test_multi_step_false_unknown_intent_ignores_fallback` above (both are included for clarity of intent, not redundant duplication — this task adds the new one to make the connection to this feature explicit in a searchable test name), and `test_destructive_at_step_1_is_allowed` already proves normal classified dispatch does NOT guard step 1, which must stay true after this change since `_run_multi_step`'s new `guard_min_step` parameter defaults to `_GUARD_MIN_STEP` at every existing call site.

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -k "classification_fallback or multi_step_false_unknown_intent" -v`

Expected: all three FAIL. The two `classification_fallback` tests fail because `dispatch()` still returns `PASS` immediately on `descriptor is None` (no fallback exists yet) — `runner.plan_next_calls` stays empty and `outcome` is `DispatchOutcome.PASS`, not `FIRED`. `test_multi_step_false_unknown_intent_ignores_fallback` should already PASS as written (it exercises no new behavior) — if it doesn't, stop and report before continuing; that would mean an assumption above is wrong.

- [ ] **Step 4: Add the sentinel descriptor and `guard_min_step` parameter**

In `runtime/chat/telegram/harness_dispatcher.py`, replace:

```python
_MAX_CHAIN_RESULT_CHARS = 1024
_GUARD_MIN_STEP = 2  # destructive-tool guard applies from step 2 onward

__all__ = ["DESTRUCTIVE_TOOLS", "DispatchOutcome", "HarnessDispatcher"]
```

with:

```python
_MAX_CHAIN_RESULT_CHARS = 1024
_GUARD_MIN_STEP = 2  # destructive-tool guard applies from step 2 onward

# Classification-fallback sentinel (spec: docs/superpowers/specs/2026-08-21-
# classification-fallback-design.md). Used only as a label when intent
# classification finds no matching skill and the multi-step planner is given
# the full catalog instead. `tool` is a non-empty sentinel string -- never
# checked against harness.has_tool() or passed to harness.execute(); the
# planner's own plan.tool is what actually runs. Never registered as a real
# skill, never imported elsewhere.
_UNCLASSIFIED_DESCRIPTOR = SkillDescriptor(
    id="unclassified_fallback",
    description="Classification miss — multi-step planner chose from the full catalog.",
    intents=[],
    tool="_unclassified",
    args_schema={},
    requires_tier1=True,
)

__all__ = ["DESTRUCTIVE_TOOLS", "DispatchOutcome", "HarnessDispatcher"]
```

Then, still in `harness_dispatcher.py`, replace the `_run_multi_step` signature:

```python
    async def _run_multi_step(
        self,
        *,
        descriptor: SkillDescriptor,
        user_text: str,
        recent: tuple[tuple[str, str], ...],
        turn_id: str,
    ) -> _ChainResult:
```

with:

```python
    async def _run_multi_step(
        self,
        *,
        descriptor: SkillDescriptor,
        user_text: str,
        recent: tuple[tuple[str, str], ...],
        turn_id: str,
        guard_min_step: int = _GUARD_MIN_STEP,
    ) -> _ChainResult:
```

Then replace the destructive-guard check:

```python
            if plan.tool in DESTRUCTIVE_TOOLS and step_no >= _GUARD_MIN_STEP:
```

with:

```python
            if plan.tool in DESTRUCTIVE_TOOLS and step_no >= guard_min_step:
```

- [ ] **Step 5: Add the fallback branch in `dispatch()`**

Still in `harness_dispatcher.py`, replace:

```python
        descriptor = self._registry.for_intent(intent)
        if descriptor is None:
            logger.info("harness_dispatcher.no_descriptor", extra={"intent": intent})
            return DispatchOutcome.PASS

        if not self._harness.has_tool(descriptor.tool):
            logger.info(
                "harness_dispatcher.no_tool", extra={"tool": descriptor.tool}
            )
            return DispatchOutcome.PASS

        if confidence < HARNESS_CONFIDENCE_THRESHOLD:
            question = _clarify_question(descriptor)
            await _send(question)
            self._tier3.append(str(chat_id), "user", user_text)
            self._tier3.append(str(chat_id), "bot", question)
            return DispatchOutcome.CLARIFY

        logger.info(
            "harness_dispatcher.recent_turns_start", extra={"chat_id": chat_id}
        )
        recent = self._recent_turns(chat_id)
        if self._multi_step:
            chain = await self._run_multi_step(
                descriptor=descriptor,
                user_text=user_text,
                recent=recent,
                turn_id=turn_id,
            )
```

with:

```python
        descriptor = self._registry.for_intent(intent)
        is_fallback = False
        if descriptor is None:
            logger.info("harness_dispatcher.no_descriptor", extra={"intent": intent})
            if not self._multi_step:
                return DispatchOutcome.PASS
            descriptor = _UNCLASSIFIED_DESCRIPTOR
            is_fallback = True
            logger.info(
                "harness_dispatcher.classification_fallback_start",
                extra={"chat_id": chat_id, "available_skills": len(self._registry.all())},
            )

        if not is_fallback:
            if not self._harness.has_tool(descriptor.tool):
                logger.info(
                    "harness_dispatcher.no_tool", extra={"tool": descriptor.tool}
                )
                return DispatchOutcome.PASS

            if confidence < HARNESS_CONFIDENCE_THRESHOLD:
                question = _clarify_question(descriptor)
                await _send(question)
                self._tier3.append(str(chat_id), "user", user_text)
                self._tier3.append(str(chat_id), "bot", question)
                return DispatchOutcome.CLARIFY

        logger.info(
            "harness_dispatcher.recent_turns_start", extra={"chat_id": chat_id}
        )
        recent = self._recent_turns(chat_id)
        if self._multi_step:
            chain = await self._run_multi_step(
                descriptor=descriptor,
                user_text=user_text,
                recent=recent,
                turn_id=turn_id,
                guard_min_step=1 if is_fallback else _GUARD_MIN_STEP,
            )
```

This is the complete change to `dispatch()` — everything from `if chain.guarded_intent is not None:` onward (the existing chain-handling block) is unchanged and untouched; it runs identically whether `chain` came from a classified descriptor or the fallback sentinel, since it only reads `chain` and `descriptor.id`, both of which are valid in either case.

- [ ] **Step 6: Run the three new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -k "classification_fallback or multi_step_false_unknown_intent" -v`

Expected: all three PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `.venv/bin/python -m pytest -m unit -q`

Expected: PASS, zero failures. This confirms `test_pass_on_unknown_intent` and `test_destructive_at_step_1_is_allowed` (both unmodified) still pass — the two regression guards this change must not break.

- [ ] **Step 8: Lint, type-check**

Run: `.venv/bin/python -m ruff check runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py`
Run: `.venv/bin/python -m mypy runtime/chat/telegram/harness_dispatcher.py`

Expected: both clean. If ruff or mypy report a pre-existing, unrelated finding elsewhere in either file, confirm it predates this change (e.g. `git stash && <command> && git stash pop`) before treating it as this task's problem.

- [ ] **Step 9: Commit**

```bash
git add runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(harness): fall through to the multi-step planner on classification miss

dispatch() previously returned PASS immediately whenever intent
classification found no matching skill -- zero tool calls, no reply,
the multi-step planner never invoked. A synthetic sentinel descriptor
now lets the fallback reuse the existing multi-step chain-handling
code unchanged, skipping only the two checks (has_tool, confidence
threshold) that require a genuinely classified skill. The destructive
guard tightens to step 1 for this path specifically via a new
guard_min_step parameter on _run_multi_step (default preserves every
existing call site's behavior) -- there's no classification signal to
trust here, so a destructive first action always needs confirmation.

Root-caused via live diagnostic investigation: list_then_read's
cross-model failure this session was a classification gate miss, not
a planner limitation -- a direct plan_next probe with the same
phrasing, bypassing classification, produced the correct plan.
EOF
)"
```

- [ ] **Step 10: Live validation against a real model**

This step is manual, not part of the automated test suite — it requires a reachable Ollama instance per the eval harness's own design.

Confirm `MODEL_SMART_LOCAL=gemma4:e2b-mlx` in `~/.aegis/.env` (it should already be at this value from prior work this session; if not, set it and note you changed it).

Run: `.venv/bin/python -m runtime.eval.cli --yes` from the repo root.

Open the written `eval/results/*.json` and inspect `list_then_read`'s `actual_calls` for each variant. Success: a real `files_list` → `files_read` call sequence now appears (regardless of whether the task's strict grading passes — args mismatches, wrong paths, etc. are a separate, already-documented problem this fix does not claim to solve; reaching the planner at all is this fix's success criterion). Also confirm `search_then_read` and every task that currently passes shows no regression in its pass count compared to the last recorded baseline this session (`read_file`, `time_check` at minimum should stay fully passing).

Report the `list_then_read` `actual_calls` detail and the full TGC/SGC summary either way, including if the result is partial or inconclusive — this step's purpose is measurement, not a pass/fail gate on the task itself.

`MODEL_SMART_LOCAL` should already be at `gemma4:e2b-mlx` after this step; if you changed it above, revert it now.
