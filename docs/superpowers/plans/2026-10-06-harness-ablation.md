# Harness Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AEGIS harness's mechanisms separately switchable, add a production native-tools mode with the deterministic guardrails, and expose six eval arms so an overnight ablation can pick a per-model harness configuration.

**Architecture:** Deterministic guardrails (path autofill, evidence block, verified-tool set, destructive-tool list) move into a pure module, `runtime/harness/guardrails.py`, with a `Guardrails` switch set. The native tool-calling loop moves from `runtime/eval/bare.py` into `runtime/harness/native_loop.py`; a new `runtime/chat/telegram/native_turn.py` runs one native-mode turn. `HarnessDispatcher` gains `harness_mode` (`planner` | `native`) and honours `Guardrails` in both modes. Config gains per-model `harness_mode` / `guardrails` (profile + env override), and the eval CLI maps arm names to presets of that config.

**Tech Stack:** Python 3.12, pydantic v2, httpx, pytest (`asyncio_mode = "auto"`, unit tests marked `pytest.mark.unit`), ruff, Ollama 0.35 native `/api/chat` tools.

**Spec:** `docs/superpowers/specs/2026-10-06-harness-ablation-design.md` (read it first; this plan argues from it).

## Global Constraints

- Every task is test-first: write the test, run it and watch it fail for the stated reason, implement, run it green.
- Every task ends with `.venv/bin/python3 -m pytest tests -q -m unit` passing and `.venv/bin/ruff check runtime tests` clean, then `graphify update .` (project CLAUDE.md).
- An unset profile must change nothing: `harness_mode` defaults to `planner`, guardrails default to today's planner behaviour (autofill, claim_gate, evidence, plan_lock on; synthesis off). Existing tests must pass without edits except where a task says otherwise.
- The `bare` eval arm (`runtime/eval/bare.py`) must keep its exact behaviour; `tests/test_eval_bare.py` must pass untouched.
- Native mode is Ollama-only. A native profile on a non-Ollama provider falls back to `planner` with a logged warning.
- Arm names on the CLI use hyphens (the spec's `native+guard` is `native-guard`, `native+guard+synth` is `native-guard-synth`, `full-lock` is `full-nolock`), so they are shell- and filename-safe.
- Result files must not collide across arms: the filename suffix is `-<arm>` for every arm except `full`.
- Never commit `uv.lock`, `.venv*`, `eval/results/**`, or files whose only change is the file mode (this checkout shows nearly every file as mode-modified; use `git -c core.fileMode=false status`).
- GateGuard hook: the first Bash call and the first Edit/Write of each file need the facts it asks for; state them and retry the same call. Parallel edits to one file can be partly blocked; re-check after a gated batch.

## Deviations from the spec, and what is not in this plan

- **Claim gate stays a dispatcher method.** The spec sketched a pure `gate_claims(reply, history)` in `guardrails.py`. The real gate (`HarnessDispatcher._gate_completion`) reads the evidence ledger and emits `HARNESS_COMPLETION_GATED`, so it cannot be pure. Both modes call that same method (Task 6), which keeps the spec's intent: one gate implementation for both modes.
- **The step-cap note** reads "Stopped after {max_steps} steps without finishing." (spec: "Stopped after N steps.").
- **Not in this plan:** the post-night decision analysis (correct-within-budget scoring, the hallucination veto with hand-reading, the per-model decision rule in spec section 7). It runs on the night results with `summarize_results.py` and `runtime.eval.hallucination`, and gets its own short plan once the data exists.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `runtime/harness/guardrails.py` | create | `Guardrails` switch set; `DESTRUCTIVE_TOOLS`; pure helpers `format_args`, `render_chain`, `with_evidence`, `history_verified_tools`, `autofill_path` |
| `runtime/harness/native_loop.py` | create (moved) | `NATIVE_SYSTEM_PROMPT`, `NATIVE_MAX_TOKENS`, `build_tool_specs`, `parse_tool_calls`, `NativeLoopResult`, `run_native_loop`, `native_chat` |
| `runtime/chat/telegram/native_turn.py` | create | `NativeTurnResult`, `run_native_turn` (loop + autofill + destructive guard + unknown-tool handling + clipping) |
| `runtime/chat/telegram/harness_dispatcher.py` | modify | import guardrails (aliases keep old private names), honour `Guardrails`, `harness_mode` + native dispatch path, `_hold_for_confirmation` extraction |
| `runtime/chat/telegram/bot.py` | modify | build `guardrails`, `harness_mode`, `native_chat` for the dispatcher |
| `runtime/eval/bare.py` | modify | import the moved loop instead of defining it |
| `runtime/config.py` | modify | `ModelProfile` / `ModelConfig` fields, env parsing, `harness_mode_for`, `guardrails_for` |
| `runtime/eval/cli.py` | modify | arm presets, `_apply_arm`, choices |
| `runtime/eval/report.py` | modify | `arm` Literal, filename suffix |
| `tests/test_harness_guardrails.py` | create | Task 1 |
| `tests/test_harness_dispatcher.py` | modify | Tasks 2 and 6 (append tests) |
| `tests/test_native_loop_move.py` | create | Task 3 |
| `tests/test_native_turn.py` | create | Task 4 |
| `tests/test_model_profiles.py` | modify | Task 5 (append tests) |
| `tests/test_eval_arms.py` | create | Task 7 |
| `~/aegis-eval-run-scripts/run_ablation.sh` | create (outside repo) | Task 8 night runs |

---

### Task 0: Baseline commit of the week's benchmark work (needs the user's go-ahead)

The branch `feat/harness-ablation` carries ~45 uncommitted files from the 2026-10-02..05 benchmark work, several of which this plan modifies again (`harness_dispatcher.py`, `config.py`, `bot.py`, `cli.py`, `report.py`, `bare.py`). Without a baseline commit, every later task commit would mix that work with the ablation change.

- [ ] **Step 1: Ask the user** whether to commit the existing work as one baseline commit on this branch. If they decline, skip every "Commit" step in this plan and leave the changes uncommitted.

- [ ] **Step 2: Review what would be staged**

Run: `git -c core.fileMode=false status --short | grep -v "^??" | head -80` and `git -c core.fileMode=false status --short | grep "^??"`
Expected: modified files under `runtime/`, `tests/`, `eval/tasks*/`, `docs/`; untracked new modules, tests and task suites. Confirm none of `uv.lock`, `.venv*`, `eval/results/` is about to be staged.

- [ ] **Step 3: Stage by path (content changes only) and commit**

```bash
git -c core.fileMode=false diff --name-only -- runtime tests eval/tasks docs | xargs git add --
git add runtime/eval/bare.py runtime/eval/hallucination.py runtime/eval/locality.py runtime/harness/limits.py \
  tests/test_eval_bare.py tests/test_eval_hallucination.py tests/test_eval_locality.py tests/test_harness_limits.py \
  tests/test_search_files_skill.py eval/tasks_hard eval/tasks_honesty eval/tasks_large \
  docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md \
  docs/superpowers/plans/2026-10-06-harness-ablation.md
git -c core.fileMode=false diff --cached --stat | tail -3
git commit -m "feat(eval): Mac Mini benchmark work: bare arm, hallucination grader, locality, per-model limits, reply fixes

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

Expected: one commit; `git -c core.fileMode=false status --short | grep -v "^??"` now shows only mode-noise or nothing in `runtime/`, `tests/`.

---

### Task 1: Extract the deterministic guardrails into `runtime/harness/guardrails.py`

Pure refactor plus one small new branch (`with_evidence` with an empty summary, needed by native mode). Planner behaviour must not change: the existing dispatcher tests are the proof.

**Files:**
- Create: `runtime/harness/guardrails.py`
- Modify: `runtime/chat/telegram/harness_dispatcher.py` (remove the moved definitions: `DESTRUCTIVE_TOOLS` lines 49-53, `_PATH_SOURCE_TOOL`/`_PATH_CONSUMER_TOOLS` lines 63-64, `_render_chain` 171-197, `_format_args` 200-205, `_history_verified_tools` 220-237, `_autofill_path_from_prior_search` 240-270; `_with_evidence` method body)
- Test: `tests/test_harness_guardrails.py`

**Interfaces:**
- Produces:
  - `Guardrails` (frozen dataclass): `autofill: bool = True`, `claim_gate: bool = True`, `evidence: bool = True`, `synthesis: bool = False`, `plan_lock: bool = True`
  - `DESTRUCTIVE_TOOLS: frozenset[str]` = `{"files_delete", "files_move", "files_write"}`
  - `format_args(args: dict[str, Any]) -> str`
  - `render_chain(history: Sequence[tuple[ToolIntent, ToolResult]], *, per_result_chars: int = DEFAULT_TOOL_RESULT_CHARS, turn_budget_chars: int = DEFAULT_TURN_RESULT_BUDGET_CHARS) -> str`
  - `with_evidence(summary: str, history: Sequence[tuple[ToolIntent, ToolResult]], *, per_result_chars: int, turn_budget_chars: int) -> str`
  - `history_verified_tools(history: Sequence[tuple[ToolIntent, ToolResult]]) -> set[str]`
  - `autofill_path(tool_intent: ToolIntent, history: Sequence[tuple[ToolIntent, ToolResult]]) -> ToolIntent`
  - The dispatcher keeps the old private names as import aliases: `_render_chain`, `_format_args`, `_history_verified_tools`, `_autofill_path_from_prior_search`, and re-exports `DESTRUCTIVE_TOOLS` (it is in `__all__`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness_guardrails.py`:

```python
"""Shared deterministic guardrails (spec: 2026-10-06-harness-ablation-design.md, M3-M5)."""
from __future__ import annotations

import pytest

from runtime.harness.contract import ToolIntent, ToolResult

pytestmark = pytest.mark.unit


def _search(result: str) -> tuple[ToolIntent, ToolResult]:
    return (
        ToolIntent(tool="files_search", args={"pattern": "*CT-001*"}, skill_id="s"),
        ToolResult(status="ok", payload={"result": result}),
    )


def test_default_guardrails_are_todays_planner_behaviour() -> None:
    from runtime.harness.guardrails import Guardrails

    g = Guardrails()
    assert (g.autofill, g.claim_gate, g.evidence, g.plan_lock) == (True, True, True, True)
    assert g.synthesis is False


def test_with_evidence_without_a_summary_is_just_the_results() -> None:
    """Native mode: a model that spent its budget thinking leaves no text of its own."""
    from runtime.harness.guardrails import with_evidence

    out = with_evidence(
        "", [_search("/tmp/n/CT-001-notes.md")], per_result_chars=3500, turn_budget_chars=8000
    )
    assert out.startswith("Results:\n1. files_search(")
    assert "/tmp/n/CT-001-notes.md" in out


def test_with_evidence_appends_after_the_summary() -> None:
    from runtime.harness.guardrails import with_evidence

    out = with_evidence(
        "Found it.", [_search("/tmp/a.md")], per_result_chars=3500, turn_budget_chars=8000
    )
    assert out.startswith("Found it.\n\nResults:\n")


def test_autofill_replaces_a_wrong_guess_with_the_search_match() -> None:
    from runtime.harness.guardrails import autofill_path

    guess = ToolIntent(tool="files_read", args={"path": "/tmp/n/ct001.md"}, skill_id="s")
    fixed = autofill_path(guess, [_search("/tmp/n/CT-001-notes.md")])
    assert fixed.args["path"] == "/tmp/n/CT-001-notes.md"


def test_the_dispatcher_uses_the_shared_functions() -> None:
    """One implementation for both harness modes, not two copies that can drift."""
    from runtime.chat.telegram import harness_dispatcher as hd
    from runtime.harness import guardrails as g

    assert hd._render_chain is g.render_chain
    assert hd._history_verified_tools is g.history_verified_tools
    assert hd._autofill_path_from_prior_search is g.autofill_path
    assert hd.DESTRUCTIVE_TOOLS is g.DESTRUCTIVE_TOOLS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_harness_guardrails.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'runtime.harness.guardrails'`.

- [ ] **Step 3: Create `runtime/harness/guardrails.py`**

Copy the bodies of the dispatcher's `_render_chain`, `_format_args`, `_history_verified_tools` and `_autofill_path_from_prior_search` verbatim (including docstrings and the `harness_dispatcher.path_autofilled` log event name), renamed:

```python
"""Deterministic guardrails shared by the planner and native harness modes.

Pure functions over a turn's (call, result) history: no model calls, so none of them
costs latency. Extracted from `HarnessDispatcher` so the native tool-calling mode
(spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md) runs the same code
as the planner, and `Guardrails` lets an ablation switch each one off.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.harness.limits import (
    DEFAULT_TOOL_RESULT_CHARS,
    DEFAULT_TURN_RESULT_BUDGET_CHARS,
    allocate_result_limits,
    clip_text,
    result_text,
)
from runtime.tools.record import verdict_for_result

# Keep the dispatcher's logger name so existing log searches still match.
logger = logging.getLogger("runtime.chat.telegram.harness_dispatcher")

History = Sequence[tuple[ToolIntent, ToolResult]]

# Tools that mutate filesystem state. Both harness modes intercept one proposed
# beyond the first tool call and route the operator through a deterministic
# confirmation (docs/PLAN_MULTI_STEP_AGENT_LOOP.md §5 decision 5).
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"files_delete", "files_move", "files_write"})

# Path auto-fill (2026-08-21 handoff, open thread #2): models retype a plausible but
# wrong path for a `files_read` right after `files_search`; trust the search's own
# first match whenever the guess is not one of the actual matches.
_PATH_SOURCE_TOOL = "files_search"
_PATH_CONSUMER_TOOLS: frozenset[str] = frozenset({"files_read"})


@dataclass(frozen=True)
class Guardrails:
    """Which deterministic mechanisms a turn applies. Defaults = today's planner.

    `synthesis` is honoured only in native mode; the planner always synthesises on
    `respond` and never on `task_complete`. `plan_lock` is honoured only by the
    planner (it needs the planner's `remaining` list).
    """

    autofill: bool = True
    claim_gate: bool = True
    evidence: bool = True
    synthesis: bool = False
    plan_lock: bool = True


def format_args(args: dict[str, Any]) -> str:
    # body of the dispatcher's `_format_args`, unchanged
    ...


def render_chain(
    history: History,
    *,
    per_result_chars: int = DEFAULT_TOOL_RESULT_CHARS,
    turn_budget_chars: int = DEFAULT_TURN_RESULT_BUDGET_CHARS,
) -> str:
    # body of the dispatcher's `_render_chain`, unchanged, calling `format_args`
    ...


def with_evidence(
    summary: str, history: History, *, per_result_chars: int, turn_budget_chars: int
) -> str:
    """The summary followed by the tool output it summarises (no model call).

    An empty summary (native mode: a reasoning model that spent its budget thinking)
    yields just the results, never a reply that starts with blank lines.
    """
    if not history:
        return summary
    evidence = render_chain(
        history, per_result_chars=per_result_chars, turn_budget_chars=turn_budget_chars
    )
    if not summary.strip():
        return f"Results:\n{evidence}"
    return f"{summary}\n\nResults:\n{evidence}"


def history_verified_tools(history: History) -> set[str]:
    # body of the dispatcher's `_history_verified_tools`, unchanged
    ...


def autofill_path(tool_intent: ToolIntent, history: History) -> ToolIntent:
    # body of the dispatcher's `_autofill_path_from_prior_search`, unchanged,
    # using `_PATH_SOURCE_TOOL` / `_PATH_CONSUMER_TOOLS` above
    ...


__all__ = [
    "DESTRUCTIVE_TOOLS",
    "Guardrails",
    "autofill_path",
    "format_args",
    "history_verified_tools",
    "render_chain",
    "with_evidence",
]
```

The four `...` bodies are moves, not rewrites: cut each function body from `harness_dispatcher.py` and paste it here unchanged except for the renamed helpers it calls (`_format_args` -> `format_args`). Do not leave `...` in the file.

- [ ] **Step 4: Point the dispatcher at the shared module**

In `runtime/chat/telegram/harness_dispatcher.py`: delete the moved definitions listed under **Files**, then add next to the other `runtime.harness` imports:

```python
from runtime.harness.guardrails import DESTRUCTIVE_TOOLS, with_evidence
from runtime.harness.guardrails import autofill_path as _autofill_path_from_prior_search
from runtime.harness.guardrails import format_args as _format_args
from runtime.harness.guardrails import history_verified_tools as _history_verified_tools
from runtime.harness.guardrails import render_chain as _render_chain
```

Replace the body of `HarnessDispatcher._with_evidence` (keep its docstring) with:

```python
        return with_evidence(
            summary,
            history,
            per_result_chars=self._tool_result_chars,
            turn_budget_chars=self._turn_result_budget_chars,
        )
```

Remove imports the dispatcher no longer uses (`allocate_result_limits`, `clip_text`, `result_text`, possibly `verdict_for_result`); ruff F401 lists them.

- [ ] **Step 5: Verify imports, then run the tests**

Run: `.venv/bin/python3 -c "import runtime.harness.guardrails, runtime.chat.telegram.harness_dispatcher"`
Expected: no output (no import cycle).

Run: `.venv/bin/python3 -m pytest tests/test_harness_guardrails.py tests/test_harness_dispatcher.py -q`
Expected: all PASS, with no edit to `tests/test_harness_dispatcher.py`. That is the proof the extraction is behaviour-preserving (it decides whether the `full` easy results from 2026-10-05 can be reused, spec section 6).

- [ ] **Step 6: Full suite, lint, graph**

Run: `.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .`
Expected: PASS, "All checks passed!".

- [ ] **Step 7: Commit**

```bash
git add runtime/harness/guardrails.py runtime/chat/telegram/harness_dispatcher.py tests/test_harness_guardrails.py
git commit -m "refactor(harness): extract deterministic guardrails into a shared module

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 2: The planner honours `Guardrails` (plan-lock, autofill, claim gate, evidence)

**Files:**
- Modify: `runtime/chat/telegram/harness_dispatcher.py` (`__init__`, `dispatch` completion branch and respond-path verdict gate, `_run_multi_step`)
- Test: `tests/test_harness_dispatcher.py` (extend `_make_loop_dispatcher`, append tests)

**Interfaces:**
- Consumes: `Guardrails` (Task 1).
- Produces: `HarnessDispatcher.__init__(..., guardrails: Guardrails | None = None)`; `None` means `Guardrails()`.

- [ ] **Step 1: Let the test helper pass guardrails**

In `tests/test_harness_dispatcher.py`, add a keyword `guardrails: Any = None` to `_make_loop_dispatcher` and pass `guardrails=guardrails` into `HarnessDispatcher(...)`. Add `from runtime.harness.guardrails import Guardrails` to the imports.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_harness_dispatcher.py`)

```python
# ---------------------------------------------------------------------------
# Guardrail switches, planner mode (spec 2026-10-06-harness-ablation-design.md)
# ---------------------------------------------------------------------------


def _search_then_read_runner(read_path: str = "/tmp/a.md") -> _StubPlanRunner:
    return _StubPlanRunner(
        plan_steps=[
            PlanStep(
                kind="tool_call",
                tool="files_search",
                args={"glob": "*.md"},
                remaining=["files_search", "files_read"],
            ),
            PlanStep(kind="tool_call", tool="files_read", args={"path": read_path}),
            PlanStep(kind="task_complete", summary="Found and read it."),
        ],
    )


async def test_plan_lock_narrows_step_two_to_the_planned_tool() -> None:
    runner = _search_then_read_runner()
    dispatcher = _make_loop_dispatcher(runner=runner)
    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())
    second = runner.plan_next_calls[1]
    assert {d.tool for d in second["available_skills"]} == {"files_read"}


async def test_plan_lock_off_offers_the_full_catalogue_at_step_two() -> None:
    runner = _search_then_read_runner()
    dispatcher = _make_loop_dispatcher(runner=runner, guardrails=Guardrails(plan_lock=False))
    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())
    second = runner.plan_next_calls[1]
    assert {d.tool for d in second["available_skills"]} == {"files_search", "files_read"}


def _search_result_harness() -> _RecordingHarness:
    return _RecordingHarness(
        tools={
            "files_search": lambda args: {"result": "/tmp/real.md"},
            "files_read": lambda args: {"content": "hello"},
        }
    )


async def test_autofill_off_keeps_the_models_guessed_path() -> None:
    registry, _ = _two_skill_setup()
    harness = _search_result_harness()
    dispatcher = _make_loop_dispatcher(
        runner=_search_then_read_runner(read_path="/tmp/guess.md"),
        registry=registry,
        harness=harness,
        guardrails=Guardrails(autofill=False),
    )
    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())
    assert harness.calls[1].args["path"] == "/tmp/guess.md"


async def test_autofill_on_corrects_the_guessed_path() -> None:
    registry, _ = _two_skill_setup()
    harness = _search_result_harness()
    dispatcher = _make_loop_dispatcher(
        runner=_search_then_read_runner(read_path="/tmp/guess.md"),
        registry=registry,
        harness=harness,
    )
    await dispatcher.dispatch(chat_id=1, user_text="find and read", message=_FakeMessage())
    assert harness.calls[1].args["path"] == "/tmp/real.md"


async def test_evidence_off_replies_with_the_summary_alone() -> None:
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*.md"}),
            PlanStep(kind="task_complete", summary="Found the markdown files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(runner=runner, guardrails=Guardrails(evidence=False))
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert message.replies == ["Found the markdown files.\n\n[ollama · stub-model]"]


async def test_claim_gate_off_ships_the_summary_unannotated() -> None:
    registry, _ = _two_skill_setup()

    def _boom(args: dict) -> dict:
        raise PermissionError("denied")

    harness = _RecordingHarness(tools={"files_search": _boom, "files_read": lambda a: {}})
    runner = _StubPlanRunner(
        plan_steps=[
            PlanStep(kind="tool_call", tool="files_search", args={"glob": "*"}),
            PlanStep(kind="task_complete", summary="Searched for the files."),
        ],
    )
    dispatcher = _make_loop_dispatcher(
        runner=runner,
        registry=registry,
        harness=harness,
        guardrails=Guardrails(claim_gate=False),
    )
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find x", message=message)
    assert "⚠️" not in message.replies[0]
```

Check `PlanStep`'s field name for the remaining list before running (`grep -n "class PlanStep" -A12 runtime/reasoning/*.py`); the dispatcher reads it as `plan.remaining`.

- [ ] **Step 3: Run them to verify the right ones fail**

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q -k "plan_lock or autofill_o or evidence_off or claim_gate_off"`
Expected: `test_plan_lock_narrows_step_two_to_the_planned_tool` and `test_autofill_on_corrects_the_guessed_path` PASS (today's behaviour); the four "off" tests FAIL with `TypeError: ... unexpected keyword argument 'guardrails'`.

- [ ] **Step 4: Implement the switches**

In `HarnessDispatcher.__init__` add the parameter `guardrails: Guardrails | None = None` (after `turn_result_budget_chars`), store `self._guardrails = guardrails or Guardrails()`, and import `Guardrails` from `runtime.harness.guardrails`.

In `_run_multi_step`:

```python
            if self._guardrails.plan_lock and plan_ids and cursor < len(plan_ids):
```

(replacing `if plan_ids and cursor < len(plan_ids):`), and

```python
            if self._guardrails.autofill:
                tool_intent = _autofill_path_from_prior_search(tool_intent, history)
```

(replacing the unconditional autofill line), and in the `task_complete` branch

```python
                gated = (
                    self._gate_completion(summary, history, turn_id)
                    if self._guardrails.claim_gate
                    else summary
                )
```

In `dispatch`, the completion branch becomes

```python
                reply_text = (
                    self._with_evidence(chain.completion_summary, chain.history)
                    if self._guardrails.evidence
                    else chain.completion_summary
                )
```

and the respond-path verdict gate after `_synthesize_chain` runs only `if self._guardrails.claim_gate:` (otherwise `reply_text` is the synthesised text unchanged).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q`
Expected: all PASS.

- [ ] **Step 6: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py
git commit -m "feat(harness): planner honours per-mechanism guardrail switches

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 3: Move the native tool-calling loop into `runtime/harness/native_loop.py`

**Files:**
- Create: `runtime/harness/native_loop.py`
- Modify: `runtime/eval/bare.py` (delete the moved definitions, import them)
- Test: `tests/test_native_loop_move.py`

**Interfaces:**
- Produces (in `runtime.harness.native_loop`): `NATIVE_SYSTEM_PROMPT: str`, `NATIVE_MAX_TOKENS = 2048`, `build_tool_specs(descriptors, has_tool) -> list[dict]`, `parse_tool_calls(message) -> list[tuple[str, dict]]`, `NativeLoopResult`, `run_native_loop(...)`, `native_chat(client, cfg, *, model, messages, tools, max_tokens) -> dict` (signatures unchanged from `bare.py`).
- `runtime.eval.bare` keeps exporting `BARE_SYSTEM_PROMPT` (= `NATIVE_SYSTEM_PROMPT`), `NativeLoopResult`, `build_tool_specs`, `native_chat`, `parse_tool_calls`, `run_native_loop`, `run_variant_bare`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_native_loop_move.py`:

```python
"""The native loop is production code; the bare eval arm imports it, not a copy."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_bare_arm_uses_the_production_native_loop() -> None:
    from runtime.eval import bare
    from runtime.harness import native_loop

    assert bare.run_native_loop is native_loop.run_native_loop
    assert bare.native_chat is native_loop.native_chat
    assert bare.build_tool_specs is native_loop.build_tool_specs
    assert bare.parse_tool_calls is native_loop.parse_tool_calls
    assert bare.BARE_SYSTEM_PROMPT == native_loop.NATIVE_SYSTEM_PROMPT
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/test_native_loop_move.py -q`
Expected: FAIL with `ImportError: cannot import name 'native_loop' from 'runtime.harness'`.

- [ ] **Step 3: Create `runtime/harness/native_loop.py`**

Move from `runtime/eval/bare.py`, verbatim: `_EMPTY_OBJECT_SCHEMA`, `build_tool_specs`, `parse_tool_calls`, `NativeLoopResult`, `run_native_loop`, `native_chat`. Rename `BARE_SYSTEM_PROMPT` to `NATIVE_SYSTEM_PROMPT` and `_BARE_MAX_TOKENS` to `NATIVE_MAX_TOKENS` (keep their comments). Module docstring:

```python
"""Ollama-native tool-calling loop: hand the model its trained `tools`, run what it
asks for, feed results back, stop when it stops asking.

Production code since 2026-10-06 (spec: docs/superpowers/specs/
2026-10-06-harness-ablation-design.md): the native harness mode uses it, and the
`bare` eval arm imports it unchanged as its control.
"""
```

Imports it needs (only these): `json`, `time`, `collections.abc` (`Awaitable`, `Callable`, `Sequence`), `dataclasses.dataclass`, `typing.Any`, `httpx`, `runtime.config.AegisConfig`, `runtime.llm.clients.OllamaClient`, the four `runtime.llm.clients.ollama_client` private helpers, `runtime.llm.telemetry` (`CallTelemetry`, `record_call`), `runtime.skills.registry.SkillDescriptor`. End with an `__all__` listing the produced names.

- [ ] **Step 4: Make `bare.py` import them**

In `runtime/eval/bare.py`, delete the moved definitions and add:

```python
from runtime.harness.native_loop import (
    NATIVE_MAX_TOKENS as _BARE_MAX_TOKENS,
    NATIVE_SYSTEM_PROMPT as BARE_SYSTEM_PROMPT,
    NativeLoopResult,
    build_tool_specs,
    native_chat,
    parse_tool_calls,
    run_native_loop,
)
```

Keep `_MAX_TOOL_RESULT_CHARS = 8000` in `bare.py` (the control's flat clip). Remove imports `bare.py` no longer uses (ruff F401). `tests/test_eval_bare.py` monkeypatches `bare_mod.native_chat`; that still works because `run_variant_bare` looks the name up in `bare`'s own namespace.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python3 -m pytest tests/test_native_loop_move.py tests/test_eval_bare.py -q`
Expected: all PASS, `tests/test_eval_bare.py` unedited.

- [ ] **Step 6: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/harness/native_loop.py runtime/eval/bare.py tests/test_native_loop_move.py
git commit -m "refactor(harness): move the native tool-calling loop into production code

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 4: `run_native_turn`: one native-mode turn with the loop-level guardrails

**Files:**
- Create: `runtime/chat/telegram/native_turn.py`
- Test: `tests/test_native_turn.py`

**Interfaces:**
- Consumes: `build_tool_specs`, `parse_tool_calls`, `NATIVE_SYSTEM_PROMPT` (Task 3); `autofill_path`, `DESTRUCTIVE_TOOLS` (Task 1); `result_text`, `clip_text` (`runtime.harness.limits`).
- Produces:
  - `NativeChat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]` — `(messages, tool_specs) -> assistant message dict`.
  - `NativeTurnResult` (frozen dataclass): `history: list[tuple[ToolIntent, ToolResult]]`, `final_text: str | None`, `hit_cap: bool`, `guarded_intent: ToolIntent | None = None`.
  - `async def run_native_turn(*, chat: NativeChat, harness: Any, descriptors: Sequence[SkillDescriptor], user_text: str, skill_id: str, max_steps: int, autofill: bool, tool_result_chars: int, guard_min_call: int = 2, on_executed: Callable[[ToolIntent, ToolResult], None] | None = None) -> NativeTurnResult`. `harness` needs `has_tool(name) -> bool` and `execute(intent) -> ToolResult`. `max_steps` counts rounds of tool calls (as `run_native_loop`); `guard_min_call` counts individual tool calls from 1.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_native_turn.py`:

```python
"""One native-mode turn (spec: 2026-10-06-harness-ablation-design.md, section 4)."""
from __future__ import annotations

from typing import Any

import pytest

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.skills.registry import SkillDescriptor

pytestmark = pytest.mark.unit


def _desc(tool: str) -> SkillDescriptor:
    return SkillDescriptor(
        id=tool,
        description=f"{tool} tool",
        intents=[tool],
        tool=tool,
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        requires_tier1=True,
    )


DESCRIPTORS = [_desc("files_list"), _desc("files_search"), _desc("files_read"), _desc("files_delete")]


class _Harness:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = tools
        self.calls: list[ToolIntent] = []

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, intent: ToolIntent) -> ToolResult:
        self.calls.append(intent)
        return ToolResult(status="ok", payload=self._tools[intent.tool](intent.args))


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
    }


def _answer(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


class _Script:
    """Fake native chat: returns the scripted messages in order, records what it saw.
    Once the script runs out it keeps asking for `files_list` (to exercise the cap)."""

    def __init__(self, *replies: dict[str, Any]) -> None:
        self._replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        return self._replies.pop(0) if self._replies else _call("files_list", path="/tmp")


async def _turn(chat: _Script, harness: _Harness, **kw: Any):
    from runtime.chat.telegram.native_turn import run_native_turn

    args: dict[str, Any] = {
        "descriptors": DESCRIPTORS,
        "user_text": "do it",
        "skill_id": "native",
        "max_steps": 5,
        "autofill": True,
        "tool_result_chars": 3500,
    }
    args.update(kw)
    return await run_native_turn(chat=chat, harness=harness, **args)


async def test_runs_requested_tools_until_the_model_answers() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "a.md\nb.md"}})
    turn = await _turn(_Script(_call("files_list", path="/tmp"), _answer("Two files.")), harness)
    assert turn.final_text == "Two files."
    assert [i.tool for i, _ in turn.history] == ["files_list"]
    assert turn.hit_cap is False


def _search_read_harness() -> _Harness:
    return _Harness(
        {
            "files_search": lambda a: {"result": "/tmp/n/CT-001-notes.md"},
            "files_read": lambda a: {"result": "notes"},
        }
    )


def _search_then_guess() -> _Script:
    return _Script(
        _call("files_search", path="/tmp/n"),
        _call("files_read", path="/tmp/n/ct001.md"),
        _answer("ok"),
    )


async def test_autofill_corrects_a_guessed_read_path() -> None:
    harness = _search_read_harness()
    await _turn(_search_then_guess(), harness)
    assert harness.calls[1].args["path"] == "/tmp/n/CT-001-notes.md"


async def test_autofill_off_keeps_the_guess() -> None:
    harness = _search_read_harness()
    await _turn(_search_then_guess(), harness, autofill=False)
    assert harness.calls[1].args["path"] == "/tmp/n/ct001.md"


async def test_step_cap_ends_with_no_text_and_keeps_the_history() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    turn = await _turn(_Script(), harness, max_steps=2)
    assert turn.hit_cap is True
    assert turn.final_text is None
    assert len(turn.history) == 2


async def test_a_destructive_tool_after_the_first_call_is_intercepted() -> None:
    harness = _Harness(
        {"files_list": lambda a: {"result": "x"}, "files_delete": lambda a: {"result": "gone"}}
    )
    chat = _Script(_call("files_list", path="/tmp"), _call("files_delete", path="/tmp/x"))
    turn = await _turn(chat, harness)
    assert turn.guarded_intent is not None
    assert turn.guarded_intent.tool == "files_delete"
    assert [c.tool for c in harness.calls] == ["files_list"]  # delete never ran


async def test_an_unknown_tool_returns_an_error_to_the_model() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    chat = _Script(_call("files_teleport", path="/tmp"), _answer("sorry"))
    turn = await _turn(chat, harness)
    assert turn.history[0][1].status == "error"
    tool_msgs = [m for m in chat.seen[1] if m.get("role") == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]


async def test_tool_results_are_clipped_to_the_per_model_limit() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "Z" * 20000}})
    chat = _Script(_call("files_list", path="/tmp"), _answer("done"))
    await _turn(chat, harness, tool_result_chars=1000)
    tool_msg = next(m for m in chat.seen[1] if m.get("role") == "tool")
    assert len(tool_msg["content"]) < 1200
    assert "truncated" in tool_msg["content"]


async def test_every_executed_call_is_reported_for_the_ledger() -> None:
    harness = _Harness({"files_list": lambda a: {"result": "x"}})
    seen: list[str] = []
    await _turn(
        _Script(_call("files_list", path="/tmp"), _answer("ok")),
        harness,
        on_executed=lambda intent, result: seen.append(intent.tool),
    )
    assert seen == ["files_list"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_native_turn.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'runtime.chat.telegram.native_turn'`.

- [ ] **Step 3: Implement `runtime/chat/telegram/native_turn.py`**

```python
"""One native-mode turn: Ollama-native tool calls plus the loop-level guardrails.

Native harness mode (spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md)
lets the model drive its trained tool-calling format instead of the AEGIS JSON planner.
This module owns the loop: path autofill before each call, the always-on
destructive-tool guard, unknown-tool errors fed back to the model, and per-model
clipping of what the model sees. Reply construction (synthesis, claim gate, evidence)
belongs to the dispatcher, which runs it the same way for both modes.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from runtime.harness.contract import ToolIntent, ToolResult
from runtime.harness.guardrails import DESTRUCTIVE_TOOLS, autofill_path
from runtime.harness.limits import clip_text, result_text
from runtime.harness.native_loop import NATIVE_SYSTEM_PROMPT, build_tool_specs, parse_tool_calls
from runtime.skills.registry import SkillDescriptor

NativeChat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class NativeTurnResult:
    history: list[tuple[ToolIntent, ToolResult]]
    final_text: str | None
    hit_cap: bool
    guarded_intent: ToolIntent | None = None


async def run_native_turn(
    *,
    chat: NativeChat,
    harness: Any,
    descriptors: Sequence[SkillDescriptor],
    user_text: str,
    skill_id: str,
    max_steps: int,
    autofill: bool,
    tool_result_chars: int,
    guard_min_call: int = 2,
    on_executed: Callable[[ToolIntent, ToolResult], None] | None = None,
) -> NativeTurnResult:
    """Call the model, run the tools it asks for, repeat until it answers.

    `max_steps` caps rounds of tool calls, as `run_native_loop` does. A destructive
    tool at call number `guard_min_call` or later is returned unexecuted in
    `guarded_intent` so the dispatcher can ask the operator to confirm it.
    """
    specs = build_tool_specs(descriptors, harness.has_tool)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    history: list[tuple[ToolIntent, ToolResult]] = []
    rounds = 0
    call_no = 0
    while True:
        message = await chat(messages, specs)
        calls = parse_tool_calls(message)
        if not calls:
            text = str(message.get("content") or "")
            return NativeTurnResult(history=history, final_text=text or None, hit_cap=False)
        if rounds >= max_steps:
            return NativeTurnResult(history=history, final_text=None, hit_cap=True)
        rounds += 1
        messages.append({**message, "role": "assistant"})
        for name, args in calls:
            call_no += 1
            try:
                intent = ToolIntent(
                    tool=name, args=args, skill_id=skill_id, rationale=f"native: call {call_no}"
                )
            except ValidationError as exc:
                error = json.dumps({"error": f"invalid tool call: {exc.errors()[0]['msg']}"})
                messages.append({"role": "tool", "tool_name": name, "content": error})
                continue
            if autofill:
                intent = autofill_path(intent, history)
            if intent.tool in DESTRUCTIVE_TOOLS and call_no >= guard_min_call:
                return NativeTurnResult(
                    history=history, final_text=None, hit_cap=False, guarded_intent=intent
                )
            if harness.has_tool(intent.tool):
                result = harness.execute(intent)
            else:
                result = ToolResult(status="error", error=f"unknown tool: {intent.tool}")
            if on_executed is not None:
                on_executed(intent, result)
            history.append((intent, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": intent.tool,
                    "content": clip_text(result_text(result), tool_result_chars),
                }
            )


__all__ = ["NativeChat", "NativeTurnResult", "run_native_turn"]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python3 -m pytest tests/test_native_turn.py -q`
Expected: all 9 PASS. If `test_an_unknown_tool_returns_an_error_to_the_model` fails on the message text, check what `result_text` renders for an error `ToolResult` (it prefixes errors) and keep the assertion on the substring `unknown tool`.

- [ ] **Step 5: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/chat/telegram/native_turn.py tests/test_native_turn.py
git commit -m "feat(harness): native-mode turn with autofill, destructive guard and clipping

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 5: Per-model `harness_mode` and `guardrails` in config

**Files:**
- Modify: `runtime/config.py` (`ModelConfig`, `ModelProfile`, `AegisConfig`, `_coerce`, two new parsers)
- Test: `tests/test_model_profiles.py` (append)

**Interfaces:**
- Consumes: `Guardrails` (Task 1), imported lazily inside the accessor (the `runtime.harness` package init imports config; see the existing `tool_result_chars_for`).
- Produces:
  - `GuardrailName = Literal["autofill", "claim_gate", "evidence", "synthesis", "plan_lock"]`
  - `HarnessMode = Literal["planner", "native"]`
  - `ModelProfile.harness_mode: HarnessMode | None`, `ModelProfile.guardrails: frozenset[GuardrailName] | None` (a profile with `harness_mode="native"` and `plan_lock` in `guardrails` is invalid)
  - `ModelConfig.harness_mode`, `ModelConfig.guardrails` (env `MODEL_HARNESS_MODE`, `MODEL_GUARDRAILS`: comma list, `none` = empty set)
  - `AegisConfig.harness_mode_for(model: str) -> HarnessMode`, `AegisConfig.guardrails_for(model: str) -> Guardrails`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model_profiles.py`; `_config(env_think=..., profiles=...)` is the file's existing helper; add `_coerce` to its `runtime.config` import if absent)

```python
# --- harness mode + guardrails (spec 2026-10-06-harness-ablation-design.md) ---


def test_unset_harness_profile_is_todays_planner() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(env_think=None, profiles={})
    assert cfg.harness_mode_for("x:1b") == "planner"
    assert cfg.guardrails_for("x:1b") == Guardrails()


def test_native_profile_defaults_to_native_plus_guard() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(env_think=None, profiles={"g:4b": ModelProfile(harness_mode="native")})
    assert cfg.harness_mode_for("g:4b") == "native"
    assert cfg.guardrails_for("g:4b") == Guardrails(plan_lock=False)


def test_an_explicit_guardrail_list_is_honoured() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(
        env_think=None,
        profiles={"q:9b": ModelProfile(harness_mode="native", guardrails=frozenset({"evidence"}))},
    )
    assert cfg.guardrails_for("q:9b") == Guardrails(
        autofill=False, claim_gate=False, evidence=True, synthesis=False, plan_lock=False
    )


def test_the_env_override_beats_a_profile() -> None:
    from runtime.harness.guardrails import Guardrails

    base = _config(env_think=None, profiles={"g:4b": ModelProfile(harness_mode="native")})
    cfg = base.model_copy(
        update={"models": ModelConfig(harness_mode="planner", guardrails=frozenset())}
    )
    assert cfg.harness_mode_for("g:4b") == "planner"
    assert cfg.guardrails_for("g:4b") == Guardrails(
        autofill=False, claim_gate=False, evidence=False, synthesis=False, plan_lock=False
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"harness_mode": "native", "guardrails": ["plan_lock"]},  # plan-lock needs the planner
        {"harness_mode": "turbo"},
        {"guardrails": ["telepathy"]},
    ],
)
def test_invalid_harness_profiles_are_rejected_at_load(raw: dict) -> None:
    assert _coerce_model_profiles({"a:1b": raw}) == {}


def test_env_parsing_for_the_harness_overrides() -> None:
    models = _coerce(
        {"MODEL_HARNESS_MODE": "native", "MODEL_GUARDRAILS": "autofill, evidence"}, {}
    )["models"]
    assert models.harness_mode == "native"
    assert models.guardrails == frozenset({"autofill", "evidence"})
    none = _coerce({"MODEL_GUARDRAILS": "none"}, {})["models"]
    assert none.guardrails == frozenset()
    bad = _coerce(
        {"MODEL_HARNESS_MODE": "turbo", "MODEL_GUARDRAILS": "autofill,bogus"}, {}
    )["models"]
    assert bad.harness_mode is None
    assert bad.guardrails is None
```

Check `_coerce`'s signature first (`grep -n "^def _coerce(" runtime/config.py`): it takes `(env, cfg)` and returns the `AegisConfig` kwargs dict, so `["models"]` is the `ModelConfig`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_model_profiles.py -q -k "harness or guardrail"`
Expected: FAIL (`AttributeError: 'AegisConfig' object has no attribute 'harness_mode_for'`, and pydantic `extra_forbidden` for the new profile fields).

- [ ] **Step 3: Implement**

Near the top of `runtime/config.py` (after the imports):

```python
HarnessMode = Literal["planner", "native"]
GuardrailName = Literal["autofill", "claim_gate", "evidence", "synthesis", "plan_lock"]
_GUARDRAIL_NAMES: frozenset[str] = frozenset(
    {"autofill", "claim_gate", "evidence", "synthesis", "plan_lock"}
)
```

In `ModelConfig`, after `planner_max_tokens`:

```python
    harness_mode: HarnessMode | None = Field(
        default=None,
        description=(
            "Override for the harness mode, beating any profile (`MODEL_HARNESS_MODE`) "
            "-- the eval arms' switch. None defers to the profile, then to `planner`."
        ),
    )
    guardrails: frozenset[GuardrailName] | None = Field(
        default=None,
        description=(
            "Override for which deterministic guardrails run, beating any profile "
            "(`MODEL_GUARDRAILS`, comma list; `none` = no guardrails). None defers to "
            "the profile, then to the mode's default."
        ),
    )
```

In `ModelProfile`, after `planner_max_tokens` (add `model_validator` to the pydantic import if absent; extend the class docstring's "Only fields with a measured per-model effect" paragraph with one line: "`harness_mode` / `guardrails`: chosen per model by the 2026-10 ablation (spec 2026-10-06-harness-ablation-design.md)."):

```python
    harness_mode: HarnessMode | None = Field(
        default=None,
        description="`planner` (AEGIS JSON planner) or `native` (Ollama tool calls). None = planner.",
    )
    guardrails: frozenset[GuardrailName] | None = Field(
        default=None,
        description=(
            "Deterministic guardrails to run. None = the mode's default (planner: all "
            "but synthesis; native: autofill, claim_gate, evidence)."
        ),
    )

    @model_validator(mode="after")
    def _plan_lock_needs_the_planner(self) -> ModelProfile:
        if self.harness_mode == "native" and self.guardrails and "plan_lock" in self.guardrails:
            raise ValueError("plan_lock needs the planner's `remaining` list; not in native mode")
        return self
```

In `AegisConfig`, after `planner_max_tokens_for`:

```python
    def harness_mode_for(self, model: str) -> HarnessMode:
        """`MODEL_HARNESS_MODE` beats a profile; a profile beats `planner`."""
        if self.models.harness_mode is not None:
            return self.models.harness_mode
        profile = self.model_profiles.get(model)
        if profile is not None and profile.harness_mode is not None:
            return profile.harness_mode
        return "planner"

    def guardrails_for(self, model: str) -> Guardrails:
        """Which guardrails run for one model, resolved like `harness_mode_for`.

        Unset means the mode's default: today's planner behaviour, or the
        `native+guard` preset in native mode. `plan_lock` never applies in native
        mode (an env override cannot raise at run time, so it is dropped here).
        """
        from runtime.harness.guardrails import Guardrails  # noqa: PLC0415

        mode = self.harness_mode_for(model)
        names = self.models.guardrails
        if names is None:
            profile = self.model_profiles.get(model)
            names = profile.guardrails if profile is not None else None
        if names is None:
            return Guardrails() if mode == "planner" else Guardrails(plan_lock=False)
        return Guardrails(
            autofill="autofill" in names,
            claim_gate="claim_gate" in names,
            evidence="evidence" in names,
            synthesis="synthesis" in names,
            plan_lock="plan_lock" in names and mode == "planner",
        )
```

Add `from typing import TYPE_CHECKING` and `if TYPE_CHECKING: from runtime.harness.guardrails import Guardrails` at module top so the return annotation resolves for type checkers without a runtime import (the module already uses `from __future__ import annotations`; confirm with `head -12 runtime/config.py`).

Parsers, next to `_parse_positive_int`:

```python
def _parse_harness_mode(raw: str | None) -> HarnessMode | None:
    """Unset or unrecognised -> None (defer to the profile), never fatal."""
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in ("planner", "native"):
        return value  # type: ignore[return-value]
    logger.warning("config.harness_mode.unrecognized_value", extra={"raw": raw})
    return None


def _parse_guardrails(raw: str | None) -> frozenset[str] | None:
    """Comma list of guardrail names; `none` = empty set; any unknown name -> None."""
    if raw is None or not raw.strip():
        return None
    if raw.strip().lower() == "none":
        return frozenset()
    names = frozenset(p.strip().lower() for p in raw.split(",") if p.strip())
    if not names <= _GUARDRAIL_NAMES:
        logger.warning("config.guardrails.unrecognized_value", extra={"raw": raw})
        return None
    return names
```

In `_coerce`, inside `ModelConfig(...)`:

```python
        harness_mode=_parse_harness_mode(env.get("MODEL_HARNESS_MODE")),
        guardrails=_parse_guardrails(env.get("MODEL_GUARDRAILS")),
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python3 -m pytest tests/test_model_profiles.py -q`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/config.py tests/test_model_profiles.py
git commit -m "feat(config): per-model harness mode and guardrail set, with env overrides

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 6: Native mode in `HarnessDispatcher`, wired from `bot.py`

**Files:**
- Modify: `runtime/chat/telegram/harness_dispatcher.py` (`__init__`, `dispatch`, new `_hold_for_confirmation`, `_dispatch_native`, `_native_reply`)
- Modify: `runtime/chat/telegram/bot.py` (`build_harness_dispatcher`, new `_native_mode`)
- Test: `tests/test_harness_dispatcher.py` (append)

**Interfaces:**
- Consumes: `run_native_turn`, `NativeChat`, `NativeTurnResult` (Task 4); `with_evidence`, `Guardrails` (Task 1); `harness_mode_for`, `guardrails_for` (Task 5); `native_chat`, `NATIVE_MAX_TOKENS` (Task 3).
- Produces: `HarnessDispatcher.__init__(..., harness_mode: str = "planner", native_chat: NativeChat | None = None)`; `bot._native_mode(cfg, target, ollama_client) -> tuple[str, NativeChat | None]`.

Behaviour (spec section 4): classification still routes first (an `unknown` intent in native mode runs the native turn over the full catalogue, like the planner's fallback); low confidence still CLARIFYs. Reply order: synthesis (if on, replaces the model's text) -> claim gate (`_gate_completion`, also on an empty history so a claim with no tool behind it is flagged) -> evidence block. Step cap: `"Stopped after {max_steps} steps without finishing."` plus evidence when evidence is on, else no reply. An empty reply returns PASS. A native-turn exception returns PASS. A guarded destructive call uses the same confirmation flow as the planner.

- [ ] **Step 1: Let the test helper build a native dispatcher**

In `tests/test_harness_dispatcher.py`, add keywords `harness_mode: str = "planner"` and `native_chat: Any = None` to `_make_loop_dispatcher` and pass them to `HarnessDispatcher(...)`.

- [ ] **Step 2: Write the failing tests** (append)

```python
# ---------------------------------------------------------------------------
# Native harness mode (spec 2026-10-06-harness-ablation-design.md, section 4)
# ---------------------------------------------------------------------------


def _tool_call_msg(name: str, **args: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": args}}],
    }


class _NativeScript:
    def __init__(self, *replies: dict[str, Any]) -> None:
        self._replies = list(replies)
        self.calls = 0

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls += 1
        return self._replies.pop(0)


def _native(script: Any, *, guardrails: Any = None, **kw: Any) -> HarnessDispatcher:
    return _make_loop_dispatcher(
        runner=_StubPlanRunner(plan_steps=[]),
        harness_mode="native",
        native_chat=script,
        guardrails=guardrails,
        **kw,
    )


async def test_native_mode_answers_from_the_model_and_never_plans() -> None:
    script = _NativeScript(
        _tool_call_msg("files_search", glob="*.md"),
        {"role": "assistant", "content": "Found a.md and b.md."},
    )
    runner = _StubPlanRunner(plan_steps=[])
    dispatcher = _make_loop_dispatcher(runner=runner, harness_mode="native", native_chat=script)
    message = _FakeMessage()
    outcome = await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert outcome == DispatchOutcome.FIRED
    assert runner.plan_next_calls == []
    assert message.replies[0].startswith("Found a.md and b.md.\n\nResults:\n")


async def test_native_mode_with_no_guardrails_sends_the_model_text_as_is() -> None:
    """Parity with the bare arm: guardrails off means nothing is added."""
    script = _NativeScript(
        _tool_call_msg("files_search", glob="*.md"),
        {"role": "assistant", "content": "Found a.md and b.md."},
    )
    dispatcher = _native(
        script,
        guardrails=Guardrails(autofill=False, claim_gate=False, evidence=False, plan_lock=False),
    )
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert message.replies == ["Found a.md and b.md.\n\n[ollama · stub-model]"]


async def test_native_claim_gate_flags_a_claim_with_no_tool_behind_it() -> None:
    script = _NativeScript({"role": "assistant", "content": "I deleted the file for you."})
    dispatcher = _native(script)
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert "unverified tool claim" in message.replies[0]


async def test_native_empty_final_message_becomes_the_evidence() -> None:
    script = _NativeScript(
        _tool_call_msg("files_search", glob="*.md"), {"role": "assistant", "content": ""}
    )
    dispatcher = _native(script)
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert message.replies[0].startswith("Results:\n1. files_search(")


async def test_native_step_cap_says_so_and_shows_the_evidence() -> None:
    script = _NativeScript(*[_tool_call_msg("files_search", glob="*.md") for _ in range(4)])
    dispatcher = _native(script, max_steps=2)
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert message.replies[0].startswith(
        "Stopped after 2 steps without finishing.\n\nResults:\n"
    )


async def test_native_synthesis_makes_one_extra_call_and_replaces_the_text() -> None:
    script = _NativeScript(
        _tool_call_msg("files_search", glob="*.md"),
        {"role": "assistant", "content": "model text"},
    )
    recorder = _RecordingSynthesizer(reply="synthesised text")
    dispatcher = _native(
        script, guardrails=Guardrails(synthesis=True, plan_lock=False), synthesizer=recorder
    )
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find md files", message=message)
    assert recorder.calls == 1
    assert message.replies[0].startswith("synthesised text\n\nResults:\n")


async def test_native_destructive_call_after_the_first_asks_for_confirmation() -> None:
    registry, _ = _two_skill_setup()
    harness = _RecordingHarness(
        tools={
            "files_search": lambda a: {"matches": ["/tmp/a.md"]},
            "files_read": lambda a: {"content": "x"},
            "files_delete": lambda a: {"deleted": True},
        }
    )
    script = _NativeScript(
        _tool_call_msg("files_search", glob="*.md"),
        _tool_call_msg("files_delete", path="/tmp/a.md"),
    )
    dispatcher = _native(script, registry=registry, harness=harness)
    message = _FakeMessage()
    await dispatcher.dispatch(chat_id=1, user_text="find and delete", message=message)
    assert [c.tool for c in harness.calls] == ["files_search"]
    assert 1 in dispatcher._pending
    assert "files_delete" in message.replies[0]


async def test_native_turn_failure_degrades_to_pass() -> None:
    async def _down(messages: Any, tools: Any) -> dict[str, Any]:
        raise RuntimeError("ollama down")

    dispatcher = _native(_down)
    outcome = await dispatcher.dispatch(
        chat_id=1, user_text="find md files", message=_FakeMessage()
    )
    assert outcome == DispatchOutcome.PASS
```

The registry from `_two_skill_setup()` has no `files_delete` descriptor, so the native catalogue does not offer it; the model asks for it anyway (a hallucinated tool), which is exactly what the guard must catch. `run_native_turn` checks the guard before `has_tool`, so this works. The `_RecordingSynthesizer` stub counts `chat` calls; the native turn's own model calls go through `_NativeScript`, so `recorder.calls == 1` counts only the synthesis call.

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q -k native`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'harness_mode'`.

- [ ] **Step 4: Extract the confirmation flow (pure refactor)**

In `dispatch`, the block that handles `chain.guarded_intent` (format the confirmation, `await _send`, store `_PendingConfirmation`, append the event, write tier3, log, return FIRED) becomes a method, and the old place calls `return await self._hold_for_confirmation(chat_id=chat_id, intent=chain.guarded_intent, skill_id=descriptor.id, user_text=user_text, send=_send)`:

```python
    async def _hold_for_confirmation(
        self,
        *,
        chat_id: int,
        intent: ToolIntent,
        skill_id: str,
        user_text: str,
        send: Callable[[str], Awaitable[None]],
    ) -> DispatchOutcome:
        """Ship the deterministic confirmation prompt and hold the intent one turn.

        Arm the pending confirmation ONLY after `send` succeeds: if the send raises,
        the operator never saw the prompt, and a later unrelated "yes" must not run a
        destructive tool they were never shown. (Phase 11 Track D, D1.)
        """
        reply_text = _format_destructive_confirmation(intent)
        await send(reply_text)
        self._pending[chat_id] = _PendingConfirmation(
            intent=intent, skill_id=skill_id, user_text=user_text, created_at=self._now()
        )
        self._append_event(
            EventType.HARNESS_CONFIRMATION_REQUESTED, {"tool": intent.tool, "skill_id": skill_id}
        )
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
        logger.info("harness_dispatcher.dispatch_complete_guarded")
        return DispatchOutcome.FIRED
```

Run `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q -k "confirm or guard"` and expect PASS before going on.

- [ ] **Step 5: Add the native path**

`__init__`: parameters `harness_mode: str = "planner"` and `native_chat: NativeChat | None = None`; store `self._harness_mode = harness_mode if native_chat is not None else "planner"` (no chat function means native mode cannot run) and `self._native_chat = native_chat`. Import `NativeChat`, `NativeTurnResult`, `run_native_turn` from `runtime.chat.telegram.native_turn`.

`dispatch`: the classification-fallback condition becomes

```python
            native = self._harness_mode == "native"
            if not (self._multi_step or native) or intent != "unknown":
                return DispatchOutcome.PASS
```

and immediately after `recent = self._recent_turns(chat_id)`:

```python
        if self._harness_mode == "native":
            return await self._dispatch_native(
                chat_id=chat_id,
                user_text=user_text,
                skill_id=descriptor.id,
                turn_id=turn_id,
                guard_min_call=1 if is_fallback else _GUARD_MIN_STEP,
                send=_send,
            )
```

New methods:

```python
    async def _dispatch_native(
        self,
        *,
        chat_id: int,
        user_text: str,
        skill_id: str,
        turn_id: str,
        guard_min_call: int,
        send: Callable[[str], Awaitable[None]],
    ) -> DispatchOutcome:
        """Native harness mode: the model tool-calls in its own format (spec section 4)."""
        assert self._native_chat is not None  # guaranteed by __init__

        def _ledger(intent: ToolIntent, result: ToolResult) -> None:
            self._record_tool_call(
                turn_id=turn_id, skill_id=skill_id, tool_intent=intent, result=result
            )

        try:
            turn = await run_native_turn(
                chat=self._native_chat,
                harness=self._harness,
                descriptors=list(self._registry.all()),
                user_text=user_text,
                skill_id=skill_id,
                max_steps=self._max_steps,
                autofill=self._guardrails.autofill,
                tool_result_chars=self._tool_result_chars,
                guard_min_call=guard_min_call,
                on_executed=_ledger,
            )
        except Exception:
            logger.exception("harness_dispatcher.native_turn_failed")
            return DispatchOutcome.PASS
        if turn.guarded_intent is not None:
            return await self._hold_for_confirmation(
                chat_id=chat_id,
                intent=turn.guarded_intent,
                skill_id=skill_id,
                user_text=user_text,
                send=send,
            )
        reply_text = await self._native_reply(user_text, turn, chat_id=chat_id, turn_id=turn_id)
        if not reply_text.strip():
            return DispatchOutcome.PASS
        await send(reply_text + self._footer())
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
        logger.info("harness_dispatcher.dispatch_complete_native")
        return DispatchOutcome.FIRED

    async def _native_reply(
        self, user_text: str, turn: NativeTurnResult, *, chat_id: int, turn_id: str
    ) -> str:
        """Synthesis (optional) -> claim gate -> evidence, the planner's order.

        The gate reads only the model's own words, never the appended evidence.
        """
        guard = self._guardrails
        history = turn.history
        if turn.hit_cap:
            text = (
                f"Stopped after {self._max_steps} steps without finishing."
                if guard.evidence
                else ""
            )
        else:
            text = (turn.final_text or "").strip()
        if guard.synthesis and history:
            text = await self._synthesize_chain(user_text, history, chat_id=chat_id)
        if text and guard.claim_gate:
            text = self._gate_completion(text, history, turn_id)
        if guard.evidence and history:
            text = with_evidence(
                text,
                history,
                per_result_chars=self._tool_result_chars,
                turn_budget_chars=self._turn_result_budget_chars,
            )
        return text
```

- [ ] **Step 6: Run the dispatcher tests**

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q`
Expected: all PASS.

- [ ] **Step 7: Wire it in `bot.py` (test first)**

First find what `ModelRouter.route` returns: `grep -n "class .*\|def route" runtime/llm/router.py`. Then append to `tests/test_harness_dispatcher.py` (using that class and its constructor; shown here as `RouteTarget(provider=..., model=...)`):

```python
def test_native_mode_falls_back_to_the_planner_off_ollama() -> None:
    from runtime.chat.telegram.bot import _native_mode
    from runtime.config import AegisConfig, ModelConfig
    from runtime.llm.router import RouteTarget

    cfg = AegisConfig().model_copy(update={"models": ModelConfig(harness_mode="native")})
    mode, chat = _native_mode(cfg, RouteTarget(provider="openrouter", model="x/y"), None)
    assert (mode, chat) == ("planner", None)
    mode, chat = _native_mode(cfg, RouteTarget(provider="ollama", model="g:4b"), object())
    assert mode == "native"
    assert callable(chat)
```

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q -k off_ollama`
Expected: FAIL with `ImportError: cannot import name '_native_mode'`.

Then add to `runtime/chat/telegram/bot.py` (module level, near `build_harness_dispatcher`):

```python
def _native_mode(cfg: AegisConfig, target: Any, ollama_client: Any) -> tuple[str, Any]:
    """(harness_mode, native_chat) for the routed model. Native mode is Ollama-only."""
    from runtime.harness.native_loop import NATIVE_MAX_TOKENS, native_chat  # noqa: PLC0415

    mode = cfg.harness_mode_for(target.model)
    if mode != "native":
        return "planner", None
    if target.provider != "ollama":
        logger.warning(
            "harness_dispatcher.native_mode_unsupported", extra={"provider": target.provider}
        )
        return "planner", None

    async def _chat(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await native_chat(
            ollama_client,
            cfg,
            model=target.model,
            messages=messages,
            tools=tools,
            max_tokens=NATIVE_MAX_TOKENS,
        )

    return "native", _chat
```

and in `build_harness_dispatcher`, before `return HarnessDispatcher(`:

```python
    harness_mode, native = _native_mode(cfg, target, ollama_client)
```

with three new keyword arguments in the `HarnessDispatcher(...)` call:

```python
        guardrails=cfg.guardrails_for(target.model),
        harness_mode=harness_mode,
        native_chat=native,
```

Run: `.venv/bin/python3 -m pytest tests/test_harness_dispatcher.py -q -k off_ollama`
Expected: PASS.

- [ ] **Step 8: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/chat/telegram/harness_dispatcher.py runtime/chat/telegram/bot.py tests/test_harness_dispatcher.py
git commit -m "feat(harness): native harness mode in the dispatcher, wired per model

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 7: Eval arms as presets of the runtime config

**Files:**
- Modify: `runtime/eval/cli.py` (`--arm` choices, `_ARM_PRESETS`, `_apply_arm`, `_arm_unsupported_reason`, `_run`)
- Modify: `runtime/eval/report.py` (`EvalReport.arm` Literal, `write_json` suffix)
- Test: `tests/test_eval_arms.py`

**Interfaces:**
- Consumes: `ModelConfig.harness_mode` / `guardrails`, `harness_mode_for`, `guardrails_for` (Task 5).
- Produces: `ARMS = ("full", "full-nolock", "native", "native-guard", "native-guard-synth", "bare")`; `_apply_arm(cfg: AegisConfig, arm: str) -> AegisConfig`.

- [ ] **Step 1: Write the failing tests**

First check `EvalReport`'s required fields: `grep -n "class EvalReport" -A30 runtime/eval/report.py`. If it requires more than `provider`, `model`, `started_at`, `tasks`, `arm`, build reports the way `tests/test_eval_report.py::_arm_report` does.

Create `tests/test_eval_arms.py`:

```python
"""Eval arms are presets of the production config (spec 2026-10-06, section 5)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.config import AegisConfig, ModelProfile
from runtime.eval.cli import ARMS, _apply_arm, _arm_unsupported_reason, _variant_runner
from runtime.eval.report import EvalReport, write_json
from runtime.harness.guardrails import Guardrails

pytestmark = pytest.mark.unit

M = "gemma4:e4b-mlx"


def _cfg() -> AegisConfig:
    # A native profile must not leak into the `full` arm.
    return AegisConfig(model_profiles={M: ModelProfile(harness_mode="native")})


@pytest.mark.parametrize(
    ("arm", "mode", "guard"),
    [
        ("full", "planner", Guardrails()),
        ("full-nolock", "planner", Guardrails(plan_lock=False)),
        ("native", "native", Guardrails(False, False, False, False, False)),
        ("native-guard", "native", Guardrails(plan_lock=False)),
        ("native-guard-synth", "native", Guardrails(synthesis=True, plan_lock=False)),
    ],
)
def test_each_arm_is_a_config_preset(arm: str, mode: str, guard: Guardrails) -> None:
    cfg = _apply_arm(_cfg(), arm)
    assert cfg.harness_mode_for(M) == mode
    assert cfg.guardrails_for(M) == guard


def test_bare_stays_the_eval_only_control() -> None:
    from runtime.eval import bare

    assert "bare" in ARMS
    assert _apply_arm(_cfg(), "bare") == _cfg()
    assert _variant_runner("bare") is bare.run_variant_bare


@pytest.mark.parametrize("arm", ["native", "native-guard", "native-guard-synth", "bare"])
def test_native_arms_need_ollama(arm: str) -> None:
    assert _arm_unsupported_reason(arm, "openrouter") is not None
    assert _arm_unsupported_reason(arm, "ollama") is None


def test_result_files_do_not_collide_across_arms(tmp_path: Path) -> None:
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    names = {
        write_json(
            EvalReport(provider="ollama", model=M, started_at=stamp, tasks=(), arm=arm), tmp_path
        ).name
        for arm in ARMS
    }
    assert len(names) == len(ARMS)
    assert "2026-10-06T00-00-00Z-ollama-gemma4-e4b-mlx.json" in names  # full: unchanged
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_eval_arms.py -q`
Expected: FAIL with `ImportError: cannot import name 'ARMS' from 'runtime.eval.cli'`.

- [ ] **Step 3: Implement**

`runtime/eval/report.py` — the `arm` field's type becomes (keep its default `"full"` and description, adding one sentence: "Non-bare arms are presets of the runtime harness config (spec 2026-10-06)."):

```python
    arm: Literal[
        "full", "full-nolock", "native", "native-guard", "native-guard-synth", "bare"
    ] = Field(
```

and in `write_json`:

```python
    arm_suffix = "" if report.arm == "full" else f"-{report.arm}"
```

`runtime/eval/cli.py` (change the config import to `from runtime.config import AegisConfig, get_config`):

```python
# Every arm but `bare` is a preset of the production config, so whatever wins can ship
# as a ModelProfile (spec: docs/superpowers/specs/2026-10-06-harness-ablation-design.md).
_GUARDED = frozenset({"autofill", "claim_gate", "evidence"})
_ARM_PRESETS: dict[str, tuple[str, frozenset[str]]] = {
    "full": ("planner", _GUARDED | {"plan_lock"}),
    "full-nolock": ("planner", _GUARDED),
    "native": ("native", frozenset()),
    "native-guard": ("native", _GUARDED),
    "native-guard-synth": ("native", _GUARDED | {"synthesis"}),
}
ARMS: tuple[str, ...] = (*_ARM_PRESETS, "bare")
_NATIVE_ARMS = frozenset({"native", "native-guard", "native-guard-synth", "bare"})


def _apply_arm(cfg: AegisConfig, arm: str) -> AegisConfig:
    """Pin the harness mode and guardrails for one arm, overriding any profile."""
    preset = _ARM_PRESETS.get(arm)
    if preset is None:  # `bare` runs its own eval-only loop
        return cfg
    mode, guardrails = preset
    return cfg.model_copy(
        update={
            "models": cfg.models.model_copy(
                update={"harness_mode": mode, "guardrails": guardrails}
            )
        }
    )
```

`--arm`: `choices=ARMS`, and its help gains one line per arm (full = AEGIS planner; full-nolock = planner without plan-lock; native = native tool calls, no guardrails; native-guard = plus autofill, claim gate and evidence; native-guard-synth = plus a synthesis call; bare = the eval-only control loop).

`_arm_unsupported_reason`:

```python
    if arm in _NATIVE_ARMS and provider != "ollama":
        return (
            f"the {arm} arm drives Ollama's native tools API; {provider!r} is not "
            "supported yet"
        )
    return None
```

`_run`: change `cfg = get_config()` to `cfg = _apply_arm(get_config(), args.arm)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python3 -m pytest tests/test_eval_arms.py tests/test_eval_report.py tests/test_eval_bare.py -q`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, graph, commit**

```bash
.venv/bin/python3 -m pytest tests -q -m unit && .venv/bin/ruff check runtime tests && graphify update .
git add runtime/eval/cli.py runtime/eval/report.py tests/test_eval_arms.py
git commit -m "feat(eval): six ablation arms as presets of the runtime harness config

Claude-Session: https://claude.ai/code/session_01R1z7fLKWPMTHPhw5rU41nq"
```

---

### Task 8: Live checks and the night-run script (operations; needs the Mac Mini idle)

Live model runs load the machine: run them only when the user says the Mac Mini is free, and record results in the plan doc before the nights.

**Files:**
- Create: `~/aegis-eval-run-scripts/run_ablation.sh` (outside the repo)
- Modify: `docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md` (new section 19)

- [ ] **Step 1: Smoke-test each new arm on one model** (search tasks only, ~2-10 min each; `preflight.py` and `tasks_search/` already exist in `~/aegis-eval-run-scripts/`)

```bash
cd /Volumes/DATA_SSD/Development/aegis
S=~/aegis-eval-run-scripts
for ARM in native native-guard native-guard-synth full-nolock; do
  .venv/bin/python3 $S/preflight.py gemma4:e4b-mlx && \
  MODEL_SMART_PROVIDER=ollama MODEL_SMART_LOCAL=gemma4:e4b-mlx .venv/bin/python3 -u -m runtime.eval.cli \
    --arm $ARM --yes --repeat 1 --require-local --tasks-dir $S/tasks_search --out-dir eval/results/ablation/smoke 2>&1 | tail -8
done
/Volumes/DATA_SSD/Applications/Ollama.app/Contents/Resources/ollama stop gemma4:e4b-mlx
```

Expected: four result files named `...-native.json`, `...-native-guard.json`, `...-native-guard-synth.json`, `...-full-nolock.json`; no `raised:` reasons in the output. Read every reply in them and confirm the shape matches the arm (no `Results:` block for `native`; one for the guarded arms).

- [ ] **Step 2: Parity check** (`native` vs `bare`, easy suite, `gemma4:e4b-mlx`, ~20 min)

```bash
for ARM in bare native; do
  MODEL_SMART_PROVIDER=ollama MODEL_SMART_LOCAL=gemma4:e4b-mlx .venv/bin/python3 -u -m runtime.eval.cli \
    --arm $ARM --yes --repeat 3 --require-local --out-dir eval/results/ablation/parity 2>&1 | tail -6
done
.venv/bin/python3 -m runtime.eval.hallucination eval/results/ablation/parity
```

Pass: `answer_ok` within 10 points between the two rows. Then count classifier diversions in the `native` file (variants with no tool calls and no reply, where the intent classifier returned PASS, or a reply that is a clarify question) and report them separately: the `bare` arm skips the classifier, native mode does not (it is production-faithful). If the difference beyond those diversions exceeds 10 points, stop and investigate before any night run.

- [ ] **Step 3: Write `~/aegis-eval-run-scripts/run_ablation.sh`**

Copy `~/aegis-eval-run-scripts/run_evening.sh` (same `wait_until`, `DEADLINE`, `unload_all`, residency sampler, `preflight.py` guard and `perl alarm` cap in `run_cfg`), set the deadline to the next morning 07:55, write results under `eval/results/ablation/<suite>/`, and replace its stages with:

```zsh
MODELS=(llama3.2:3b qwen3.5:4b-mlx gemma4:e4b-mlx lfm2.5:8b qwen3.5:9b)
O=eval/results/ablation
mkdir -p $O/easy $O/hard $O/honesty

if [ "$NIGHT" = 1 ]; then
  for SUITE in easy hard; do
    TASKS=eval/tasks; REP=3; [ $SUITE = hard ] && { TASKS=eval/tasks_hard; REP=2; }
    for ARM in bare native-guard native-guard-synth; do
      for M in $MODELS; do run_cfg $ARM $M $TASKS $O/$SUITE $SUITE local $REP; done
    done
  done
  for ARM in bare native-guard; do
    for M in $MODELS; do run_cfg $ARM $M eval/tasks_honesty $O/honesty honesty local 2; done
  done
fi

if [ "$NIGHT" = 2 ]; then
  for M in $MODELS; do run_cfg full $M eval/tasks_hard $O/hard hard local 2; done
  for SUITE in easy hard; do
    TASKS=eval/tasks; REP=3; [ $SUITE = hard ] && { TASKS=eval/tasks_hard; REP=2; }
    for M in $MODELS; do run_cfg full-nolock $M $TASKS $O/$SUITE $SUITE local $REP; done
  done
fi
```

Run it as `NIGHT=1 nohup caffeinate -i ~/aegis-eval-run-scripts/run_ablation.sh > <log> 2>&1 &` on the agreed night, and `NIGHT=2` the next. `full` on easy is reused from `eval/results/fixed/easy` only if Task 1 Step 5 passed with no test edits; otherwise add `for M in $MODELS; do run_cfg full $M eval/tasks $O/easy easy local 3; done` to night 2. Syntax-check with `zsh -n` before arming.

- [ ] **Step 4: Record and hand off**

Add section 19 to `docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md`: what was built (commit hashes), the smoke and parity results with numbers, classifier diversions, whether `full` easy is reused, and the armed night schedule. Update the memory handoff (`session_handoff_active.md`) with the same, plus the verify commands for the nights.
