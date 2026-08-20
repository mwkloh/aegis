# Multi-Step Plan Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain AEGIS's multi-step tool-use planner so a model cannot regress to an already-completed step or substitute an unplanned tool once execution of a turn's plan has started.

**Architecture:** Entirely inside `HarnessDispatcher._run_multi_step`'s existing loop. After the first `tool_call` step of a turn, derive an ordered `plan_ids: list[str]` from that step's own `tool` plus its `remaining` field, and a `cursor` into it. From the second `plan_next` call onward, while the cursor is inside the plan, narrow the `available_skills` argument to just the one `SkillDescriptor` at the cursor. The cursor advances only on a verified (non-error) execution of that exact tool; a failure keeps the same single-tool restriction on the next call instead of reverting or widening. Once the cursor reaches the end of the plan, the next call reverts to the full, unrestricted catalog — matching today's existing behavior after a plan completes.

**Tech Stack:** Python 3.11, pytest, existing `_StubPlanRunner`/`_MultiSkillRegistry`/`_RecordingHarness` test fixtures in `tests/test_harness_dispatcher.py`.

**Spec:** `docs/superpowers/specs/2026-08-21-multi-step-plan-lock-design.md`

## Global Constraints

- No changes to `Tier1Reasoner.plan_next`'s signature, `PlanStep`'s shape, or `_build_planner_schema` (`runtime/reasoning/tier1_reasoner.py`) — all reused as-is.
- The plan/cursor state (`plan_ids`, `cursor`) must stay local to `_run_multi_step` — not persisted across turns, not exposed on `HarnessDispatcher` or any other class.
- No new retry-count cap. The existing `self._max_steps` loop bound is the only escape valve for a step that keeps failing.
- `respond` and `task_complete` must remain reachable in the planner's schema at every step regardless of restriction — never force a `tool_call`-only choice.
- Pattern 1 (zero-engagement, `list_then_read`-shaped) is out of scope. Do not attempt to make it pass; a flat result there is not a regression.

---

### Task 1: Restrict available_skills to the plan cursor, with tests

**Files:**
- Modify: `runtime/chat/telegram/harness_dispatcher.py:682-764` (`_run_multi_step`)
- Modify: `tests/test_harness_dispatcher.py:788-812` (`_two_skill_setup`) — add an optional `read_fails` parameter
- Test: `tests/test_harness_dispatcher.py` (new tests, placed after `test_multi_step_single_tool_then_respond`, currently ending at line 897)

**Interfaces:**
- Consumes: `PlanStep.remaining: list[str]` (already exists, defaults to `[]`), `PlanStep.tool: str | None`, `SkillDescriptor.tool: str` (already exist). `verdict_for_result(result: ToolResult) -> str` (already imported in `harness_dispatcher.py` at line 27).
- Produces: no new public names. `_run_multi_step`'s external behavior (return type `_ChainResult`, its own signature) is unchanged — only what it passes as `available_skills` to `self._runner.plan_next(...)` on steps after the first tool call changes.

- [ ] **Step 1: Read the current `_run_multi_step` method in full**

Run: `sed -n '647,764p' runtime/chat/telegram/harness_dispatcher.py`

Confirm it matches what this plan assumes: `available = list(self._registry.all())` computed once before the loop; the loop calls `self._runner.plan_next(user_text=..., available_skills=available, history=tuple(history), recent=recent)` every iteration, unchanged; a `tool_call` step builds `tool_intent`, checks the destructive guard (`plan.tool in DESTRUCTIVE_TOOLS and step_no >= _GUARD_MIN_STEP`), executes via `self._harness.execute(tool_intent)`, records via `self._record_tool_call(...)`, and appends `(tool_intent, result)` to `history`. If the file has diverged from this shape, stop and report — this plan's steps assume this exact structure.

- [ ] **Step 2: Add `read_fails` to `_two_skill_setup`, so a later test can make the second planned tool fail**

In `tests/test_harness_dispatcher.py`, replace:

```python
def _two_skill_setup() -> tuple[_MultiSkillRegistry, _RecordingHarness]:
    search = SkillDescriptor(
        id="search_files",
        description="Search for files matching a glob.",
        intents=["search_files"],
        tool="files_search",
        args_schema={"type": "object", "properties": {"glob": {"type": "string"}}},
        requires_tier1=True,
    )
    read = SkillDescriptor(
        id="read_file",
        description="Read a file's contents.",
        intents=["read_file"],
        tool="files_read",
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )
    registry = _MultiSkillRegistry([search, read], primary_intent="search_files")
    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": ["/tmp/a.md", "/tmp/b.md"]},
            "files_read": lambda args: {"content": "hello"},
        }
    )
    return registry, harness
```

with:

```python
def _two_skill_setup(
    *, read_fails: bool = False
) -> tuple[_MultiSkillRegistry, _RecordingHarness]:
    search = SkillDescriptor(
        id="search_files",
        description="Search for files matching a glob.",
        intents=["search_files"],
        tool="files_search",
        args_schema={"type": "object", "properties": {"glob": {"type": "string"}}},
        requires_tier1=True,
    )
    read = SkillDescriptor(
        id="read_file",
        description="Read a file's contents.",
        intents=["read_file"],
        tool="files_read",
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )
    registry = _MultiSkillRegistry([search, read], primary_intent="search_files")

    def _read(args: dict[str, Any]) -> dict[str, Any]:
        if read_fails:
            raise RuntimeError("boom")
        return {"content": "hello"}

    harness = _RecordingHarness(
        tools={
            "files_search": lambda args: {"matches": ["/tmp/a.md", "/tmp/b.md"]},
            "files_read": _read,
        }
    )
    return registry, harness
```

This is backward compatible — every existing call site calls `_two_skill_setup()` with no arguments, unaffected by the new default-`False` parameter.

- [ ] **Step 3: Write the three new failing tests**

Add these three test functions to `tests/test_harness_dispatcher.py`, directly after `test_multi_step_single_tool_then_respond` (which currently ends at line 897, just before the blank line at 898-899):

```python
async def test_multi_step_restricts_tool_choices_to_plan_cursor() -> None:
    """Step 1 sees the full catalog. After it succeeds, step 2 is narrowed
    to just the next planned tool. After that succeeds too (plan
    exhausted), step 3 sees the full catalog again."""
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(
                kind="tool_call",
                tool="files_search",
                args={"glob": "*.md"},
                remaining=["files_search", "files_read"],
            ),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, registry=registry, harness=harness)

    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())

    assert len(runner.plan_next_calls) == 3
    assert {d.tool for d in runner.plan_next_calls[0]["available_skills"]} == {
        "files_search",
        "files_read",
    }
    assert [d.tool for d in runner.plan_next_calls[1]["available_skills"]] == ["files_read"]
    assert {d.tool for d in runner.plan_next_calls[2]["available_skills"]} == {
        "files_search",
        "files_read",
    }


async def test_multi_step_restriction_persists_after_step_failure() -> None:
    """A failing planned step keeps the same single-tool restriction on the
    next call -- the cursor does not advance, and the model does not regain
    access to files_search, an earlier already-succeeded tool."""
    registry, harness = _two_skill_setup(read_fails=True)
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(
                kind="tool_call",
                tool="files_search",
                args={"glob": "*.md"},
                remaining=["files_search", "files_read"],
            ),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, registry=registry, harness=harness)

    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())

    # files_read fails every time it's called (read_fails=True), so the
    # runner keeps returning the last queued step (files_read) and the loop
    # runs to the default max_steps=5. Steps 2 through 5 (call indices 1-4)
    # must all stay restricted to files_read only.
    assert len(runner.plan_next_calls) == 5
    for call in runner.plan_next_calls[1:]:
        assert [d.tool for d in call["available_skills"]] == ["files_read"]


async def test_multi_step_plan_leads_with_actual_tool_called() -> None:
    """If the model's first-step `remaining` omits its own tool, the
    derived plan must still lead with the tool actually called, not just
    whatever the model self-reported -- otherwise the plan would appear
    exhausted after one step and step 2 would wrongly see the full catalog
    instead of being restricted to files_read."""
    registry, harness = _two_skill_setup()
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(
                kind="tool_call",
                tool="files_search",
                args={"glob": "*.md"},
                remaining=["files_read"],  # omits its own tool, "files_search"
            ),
            PlanStep(kind="tool_call", tool="files_read", args={"path": "/tmp/a.md"}),
            PlanStep(kind="respond"),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, registry=registry, harness=harness)

    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())

    assert [d.tool for d in runner.plan_next_calls[1]["available_skills"]] == ["files_read"]
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -k "restricts_tool_choices or restriction_persists or plan_leads_with_actual_tool" -v`

Expected: all three FAIL. `test_multi_step_restricts_tool_choices_to_plan_cursor` and `test_multi_step_plan_leads_with_actual_tool_called` fail because `plan_next_calls[1]["available_skills"]` currently contains both tools, not just `files_read` (no restriction exists yet). `test_multi_step_restriction_persists_after_step_failure` fails for the same reason on every call after the first.

- [ ] **Step 5: Implement the plan-cursor restriction in `_run_multi_step`**

In `runtime/chat/telegram/harness_dispatcher.py`, inside `_run_multi_step`, replace:

```python
        available = list(self._registry.all())
        history: list[tuple[ToolIntent, ToolResult]] = []
        logger.info(
            "harness_dispatcher.multi_step_start",
            extra={
                "skill_id": descriptor.id,
                "available_skills": len(available),
                "max_steps": self._max_steps,
            },
        )

        for step_no in range(1, self._max_steps + 1):
            logger.info(
                "harness_dispatcher.plan_next_start",
                extra={"step": step_no, "history_len": len(history)},
            )
            plan = await self._runner.plan_next(
                user_text=user_text,
                available_skills=available,
                history=tuple(history),
                recent=recent,
            )
            logger.info(
                "harness_dispatcher.plan_next_done step=%d kind=%s tool=%s",
                step_no,
                plan.kind,
                plan.tool,
            )
            if plan.kind == "task_complete":
```

with:

```python
        available = list(self._registry.all())
        history: list[tuple[ToolIntent, ToolResult]] = []
        # Turn-local plan lock (spec: docs/superpowers/specs/2026-08-21-
        # multi-step-plan-lock-design.md). Derived from the first tool_call
        # step's own `tool` + `remaining`; empty until then. While the
        # cursor is inside plan_ids, available_skills is narrowed to just
        # that one step's tool -- the model can retry a failed step but
        # cannot regress to an earlier one or substitute an unplanned tool.
        plan_ids: list[str] = []
        cursor = 0
        logger.info(
            "harness_dispatcher.multi_step_start",
            extra={
                "skill_id": descriptor.id,
                "available_skills": len(available),
                "max_steps": self._max_steps,
            },
        )

        for step_no in range(1, self._max_steps + 1):
            if plan_ids and cursor < len(plan_ids):
                step_available = [d for d in available if d.tool == plan_ids[cursor]]
            else:
                step_available = available
            logger.info(
                "harness_dispatcher.plan_next_start",
                extra={"step": step_no, "history_len": len(history)},
            )
            plan = await self._runner.plan_next(
                user_text=user_text,
                available_skills=step_available,
                history=tuple(history),
                recent=recent,
            )
            logger.info(
                "harness_dispatcher.plan_next_done step=%d kind=%s tool=%s",
                step_no,
                plan.kind,
                plan.tool,
            )
            if plan.kind == "task_complete":
```

Then, still in `_run_multi_step`, replace:

```python
            if plan.kind != "tool_call" or plan.tool is None:
                break

            tool_intent = ToolIntent(
                tool=plan.tool,
                args=dict(plan.args or {}),
                skill_id=descriptor.id,
                rationale=f"multi-step planner: step {step_no}",
            )
```

with:

```python
            if plan.kind != "tool_call" or plan.tool is None:
                break

            if not plan_ids:
                plan_ids = [plan.tool] + [t for t in plan.remaining if t != plan.tool]
                cursor = 0

            tool_intent = ToolIntent(
                tool=plan.tool,
                args=dict(plan.args or {}),
                skill_id=descriptor.id,
                rationale=f"multi-step planner: step {step_no}",
            )
```

Then, still in `_run_multi_step`, replace:

```python
            result = self._harness.execute(tool_intent)
            logger.info(
                "harness_dispatcher.harness_execute_done",
                extra={"step": step_no, "status": result.status},
            )
            self._record_tool_call(
                turn_id=turn_id,
                skill_id=descriptor.id,
                tool_intent=tool_intent,
                result=result,
            )
            history.append((tool_intent, result))

        logger.info(
```

with:

```python
            result = self._harness.execute(tool_intent)
            logger.info(
                "harness_dispatcher.harness_execute_done",
                extra={"step": step_no, "status": result.status},
            )
            self._record_tool_call(
                turn_id=turn_id,
                skill_id=descriptor.id,
                tool_intent=tool_intent,
                result=result,
            )
            history.append((tool_intent, result))

            if (
                plan_ids
                and cursor < len(plan_ids)
                and plan.tool == plan_ids[cursor]
                and verdict_for_result(result) == "verified"
            ):
                cursor += 1

        logger.info(
```

- [ ] **Step 6: Run the three new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -k "restricts_tool_choices or restriction_persists or plan_leads_with_actual_tool" -v`

Expected: all three PASS.

- [ ] **Step 7: Run the full unit suite**

Run: `.venv/bin/python -m pytest -m unit -q`

Expected: PASS, zero failures. This confirms the restriction change doesn't break any existing single-skill or already-passing multi-step tests (which only ever see one skill in their registries, so the restriction narrows a one-element list to itself — a no-op for them).

- [ ] **Step 8: Lint, type-check**

Run: `.venv/bin/python -m ruff check runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py`
Run: `.venv/bin/python -m mypy runtime/chat/telegram/harness_dispatcher.py`

Expected: both clean. If ruff reports a pre-existing, unrelated finding elsewhere in either file, confirm it predates this change (e.g. `git stash` and re-run) before treating it as this task's problem.

- [ ] **Step 9: Commit**

```bash
git add runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(harness): lock multi-step tool choices to the derived plan

_run_multi_step now derives a turn-local plan (plan_ids, cursor) from
the first tool_call step's own tool + remaining, and narrows
available_skills to just the current cursor's tool on every later
plan_next call. The cursor advances only on a verified execution of
that exact tool; a failure keeps the same restriction rather than
reverting or widening. respond/task_complete stay reachable at every
step. Targets two measured failure patterns: regressing to an
already-succeeded step after a later one fails, and substituting an
unplanned tool (e.g. run_command) under difficulty. A third pattern
(zero-engagement) is out of scope -- see the spec's Non-goals.
EOF
)"
```

- [ ] **Step 10: Live validation against real models**

This step is manual, not part of the automated test suite — it requires a reachable Ollama instance per the eval harness's own design.

Run: `.venv/bin/python -m runtime.eval.cli --yes` twice, once with `MODEL_SMART_LOCAL=gemma4:e2b-mlx` and once with `MODEL_SMART_LOCAL=gemma4:e4b-mlx` set in `~/.aegis/.env` (edit the file between runs; `smart_provider` should already be `ollama`).

For the `gemma4:e2b-mlx` run, open the written `eval/results/*.json` and inspect `search_then_read`'s `actual_calls` for each variant. Success: no repeated already-succeeded tool call (the Pattern 2 shape — `files_search` appearing again after `files_read` already ran or errored) and no unplanned tool such as `run_command` (the Pattern 3 shape) anywhere in the sequence. `list_then_read` is expected to stay flat — out of scope, not a regression to worry about.

For the `gemma4:e4b-mlx` run, confirm no task that currently passes (per the session's last recorded 4B run: `list_downloads`, `list_then_read`, `read_file`, `search_files`, `time_check` all full-pass, `search_then_read` 1/2) drops below its prior pass count.

Report the before/after TGC/SGC numbers and the `search_then_read` `actual_calls` detail either way, including if the result is inconclusive or unchanged — this step's purpose is measurement, not a pass/fail gate on the task itself.

Revert `MODEL_SMART_LOCAL` back to `gemma4:e2b-mlx` in `~/.aegis/.env` when done, matching the value it had before this task started.
