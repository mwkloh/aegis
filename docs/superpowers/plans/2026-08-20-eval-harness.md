# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live-model benchmark that measures whether `HarnessDispatcher`'s multi-step planner actually completes tasks reliably against Aegis's real skill catalog and whichever model is currently pinned via `smart_provider`, reporting TGC (per-run) and SGC (strict, per-task) rates.

**Architecture:** New `runtime/eval/` package (task models + YAML loader, pure grading logic, report models + rendering, the runner that builds a real `HarnessDispatcher` via `build_harness_dispatcher` and wraps its `HarnessAdapter` in an observer to capture real tool-call args, and a CLI entrypoint) plus a new top-level `eval/tasks/*.yaml` directory of hand-authored task definitions. Grading and reporting are pure, fully unit-tested. The runner's dispatcher-building/dispatching path requires a live model and is not unit-tested, matching the spec.

**Tech Stack:** Python 3.11, pydantic v2, PyYAML (`safe_load`), pytest (`pytest.mark.unit` for the testable pieces).

**Spec:** `docs/superpowers/specs/2026-08-20-eval-harness-design.md`

## Global Constraints

- No LLM-judge grading — structural only, via observed real tool calls (spec's Grading section, corrected).
- Fixture isolation is non-negotiable: every run's `FilesClient.allowed_roots` points *only* at a fresh temp sandbox, never the real filesystem.
- No production code is modified. This plan only adds new files under `runtime/eval/`, `eval/`, and small additions to `Makefile`/`.gitignore`.
- `runtime/eval/cli.py`'s end-to-end path is never added to `pytest -m unit` and never runs in CI — it costs real tokens/time against a live model.
- Each variant runs once per invocation (no repeated sampling in v1).

---

### Task 1: Task definition models + YAML loader

**Files:**
- Create: `runtime/eval/__init__.py` (empty)
- Create: `runtime/eval/tasks.py`
- Test: `tests/test_eval_tasks.py`

**Interfaces:**
- Produces: `FixtureFile`, `TaskFixture`, `ExpectedCall`, `EvalTask` (all frozen Pydantic models), `load_tasks(tasks_dir: Path) -> list[EvalTask]`, `substitute_sandbox(value: Any, sandbox: Path) -> Any` — consumed by Task 3 (runner) and Task 2 (grading, for `ExpectedCall`'s type only).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_tasks.py`:
```python
"""Eval task YAML loader + `{sandbox}` substitution."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.eval.tasks import EvalTask, load_tasks, substitute_sandbox

pytestmark = pytest.mark.unit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_tasks_parses_full_task(tmp_path: Path) -> None:
    _write(
        tmp_path / "search_then_read.yaml",
        """
id: search_then_read
description: "Search for files matching a pattern, then read the first result."
fixture:
  files:
    - path: "notes/CT-001-notes.md"
      content: "Some notes about CT-001."
variants:
  - "find files about CT-001 in {sandbox}/notes and read the first one"
  - "search {sandbox}/notes for CT-001 and open the top match"
expected_calls:
  - tool: files_search
    args_match: {glob: "*CT-001*"}
  - tool: files_read
    args_match: {}
""",
    )
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "search_then_read"
    assert len(task.fixture.files) == 1
    assert task.fixture.files[0].path == "notes/CT-001-notes.md"
    assert task.fixture.files[0].content == "Some notes about CT-001."
    assert len(task.variants) == 2
    assert len(task.expected_calls) == 2
    assert task.expected_calls[0].tool == "files_search"
    assert task.expected_calls[0].args_match == {"glob": "*CT-001*"}
    assert task.expected_calls[1].args_match == {}


def test_load_tasks_defaults_empty_fixture(tmp_path: Path) -> None:
    _write(
        tmp_path / "time_check.yaml",
        """
id: time_check
description: "Ask what time it is."
variants:
  - "what time is it?"
expected_calls:
  - tool: time
    args_match: {}
""",
    )
    tasks = load_tasks(tmp_path)
    assert tasks[0].fixture.files == ()


def test_load_tasks_sorted_by_filename(tmp_path: Path) -> None:
    _write(tmp_path / "b_task.yaml", "id: b\ndescription: b\nvariants: ['x']\nexpected_calls: [{tool: echo, args_match: {}}]\n")
    _write(tmp_path / "a_task.yaml", "id: a\ndescription: a\nvariants: ['x']\nexpected_calls: [{tool: echo, args_match: {}}]\n")
    tasks = load_tasks(tmp_path)
    assert [t.id for t in tasks] == ["a", "b"]


def test_load_tasks_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_tasks(tmp_path) == []


def test_load_tasks_ignores_non_yaml_files(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", "not a task")
    assert load_tasks(tmp_path) == []


def test_substitute_sandbox_in_string() -> None:
    result = substitute_sandbox("list files in {sandbox}/notes", Path("/tmp/eval-abc"))
    assert result == "list files in /tmp/eval-abc/notes"


def test_substitute_sandbox_in_dict_values() -> None:
    result = substitute_sandbox({"path": "{sandbox}/notes", "glob": "*.md"}, Path("/tmp/eval-abc"))
    assert result == {"path": "/tmp/eval-abc/notes", "glob": "*.md"}


def test_substitute_sandbox_passthrough_non_string() -> None:
    assert substitute_sandbox(42, Path("/tmp/eval-abc")) == 42
    assert substitute_sandbox(None, Path("/tmp/eval-abc")) is None


def test_evaltask_requires_at_least_one_variant() -> None:
    with pytest.raises(Exception):
        EvalTask(id="x", description="x", variants=(), expected_calls=({"tool": "echo", "args_match": {}},))


def test_evaltask_requires_at_least_one_expected_call() -> None:
    with pytest.raises(Exception):
        EvalTask(id="x", description="x", variants=("hi",), expected_calls=())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.eval'` (the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `runtime/eval/__init__.py` (empty file).

Create `runtime/eval/tasks.py`:
```python
"""Eval task definitions — YAML-loaded, matching SkillDescriptor's `safe_load`-only convention.

A task is one behavioral scenario against Aegis's real skill catalog: a set
of natural-language phrasings ("variants") that should all produce the same
underlying tool-call sequence ("expected_calls"), optionally seeded with
fixture files into a per-run sandbox directory. See
docs/superpowers/specs/2026-08-20-eval-harness-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FixtureFile(BaseModel):
    """One file seeded into the per-run sandbox before dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, description="Relative to the sandbox root.")
    content: str = Field(default="")


class TaskFixture(BaseModel):
    """Files a task needs present in the sandbox. Empty for fixture-free tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[FixtureFile, ...] = Field(default_factory=tuple)


class ExpectedCall(BaseModel):
    """One tool call a passing run must make, in order, among the actual calls made.

    `args_match` is partial — only listed keys are checked. String values
    compare via substring containment against the real argument value after
    `{sandbox}` substitution; non-string values compare by equality.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1)
    args_match: dict[str, Any] = Field(default_factory=dict)


class EvalTask(BaseModel):
    """One task template: several phrasings, one expected tool-call sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    fixture: TaskFixture = Field(default_factory=TaskFixture)
    variants: tuple[str, ...] = Field(min_length=1)
    expected_calls: tuple[ExpectedCall, ...] = Field(min_length=1)


def load_tasks(tasks_dir: Path) -> list[EvalTask]:
    """Load every `*.yaml` file under `tasks_dir`, sorted by filename.

    Non-dict YAML content (e.g. a stray non-task file) is skipped rather
    than raising, matching this codebase's degrade-don't-crash convention
    for declarative loaders (see `SkillRegistry.from_directory`).
    """
    tasks: list[EvalTask] = []
    for path in sorted(Path(tasks_dir).glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        tasks.append(EvalTask.model_validate(raw))
    return tasks


def substitute_sandbox(value: Any, sandbox: Path) -> Any:
    """Replace `{sandbox}` with the real sandbox path in strings, recursively through dicts.

    Non-string, non-dict values pass through unchanged.
    """
    if isinstance(value, str):
        return value.replace("{sandbox}", str(sandbox))
    if isinstance(value, dict):
        return {k: substitute_sandbox(v, sandbox) for k, v in value.items()}
    return value


__all__ = [
    "EvalTask",
    "ExpectedCall",
    "FixtureFile",
    "TaskFixture",
    "load_tasks",
    "substitute_sandbox",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_tasks.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/eval/__init__.py runtime/eval/tasks.py tests/test_eval_tasks.py
git commit -m "feat(eval): task definition models + YAML loader"
```

---

### Task 2: Grading (pure, no dispatcher involved)

**Files:**
- Create: `runtime/eval/grading.py`
- Test: `tests/test_eval_grading.py`

**Interfaces:**
- Consumes: `ExpectedCall` (Task 1).
- Produces: `CallRecord = tuple[str, dict[str, Any], str]` (tool, args, status), `grade_calls(expected_calls: tuple[ExpectedCall, ...], actual_calls: list[CallRecord]) -> GradeResult` — consumed by Task 3 (runner).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_grading.py`:
```python
"""Ordered subsequence grading of observed tool calls against expected_calls."""
from __future__ import annotations

import pytest

from runtime.eval.grading import grade_calls
from runtime.eval.tasks import ExpectedCall

pytestmark = pytest.mark.unit


def test_exact_match_passes() -> None:
    expected = (ExpectedCall(tool="files_search", args_match={"glob": "*CT-001*"}),)
    actual = [("files_search", {"glob": "*CT-001*", "root": "."}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is True


def test_missing_call_fails() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_search", {"glob": "*.md"}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is False
    assert "files_read" in result.reason


def test_wrong_args_fails() -> None:
    expected = (ExpectedCall(tool="files_search", args_match={"glob": "*CT-001*"}),)
    actual = [("files_search", {"glob": "*other*"}, "ok")]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_error_status_does_not_count() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_read", {"path": "/tmp/x"}, "error")]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_order_matters() -> None:
    expected = (
        ExpectedCall(tool="files_search", args_match={}),
        ExpectedCall(tool="files_read", args_match={}),
    )
    # files_read happens BEFORE files_search -- wrong order.
    actual = [
        ("files_read", {"path": "/tmp/x"}, "ok"),
        ("files_search", {"glob": "*.md"}, "ok"),
    ]
    result = grade_calls(expected, actual)
    assert result.passed is False


def test_incidental_extra_calls_are_tolerated() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [
        ("time", {}, "ok"),  # incidental, unrelated call
        ("files_read", {"path": "/tmp/x"}, "ok"),
    ]
    result = grade_calls(expected, actual)
    assert result.passed is True


def test_empty_args_match_matches_any_args() -> None:
    expected = (ExpectedCall(tool="files_read", args_match={}),)
    actual = [("files_read", {"path": "/tmp/anything"}, "ok")]
    assert grade_calls(expected, actual).passed is True


def test_non_string_args_match_by_equality() -> None:
    expected = (ExpectedCall(tool="tier2_compress", args_match={"limit": 5}),)
    assert grade_calls(expected, [("tier2_compress", {"limit": 5}, "ok")]).passed is True
    assert grade_calls(expected, [("tier2_compress", {"limit": 6}, "ok")]).passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_grading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.eval.grading'`.

- [ ] **Step 3: Write the implementation**

Create `runtime/eval/grading.py`:
```python
"""Ordered subsequence grading — did the run make the expected tool calls, in order?

No LLM-judge, no reply-text parsing. Matches this codebase's existing
philosophy of trusting structural evidence over self-reported text (see
`HarnessDispatcher._gate_completion`).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from runtime.eval.tasks import ExpectedCall

CallRecord = tuple[str, dict[str, Any], str]  # (tool, args, status)


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reason: str = ""


def _value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected in actual
    return bool(expected == actual)


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(
        key in actual and _value_matches(value, actual[key])
        for key, value in expected.items()
    )


def grade_calls(
    expected_calls: tuple[ExpectedCall, ...], actual_calls: list[CallRecord]
) -> GradeResult:
    """Ordered subsequence match: each expected call must match some actual
    call at or after the position of the previous match. Incidental extra
    calls between matches are tolerated. A call with status "error" never
    counts as satisfying an expected call.
    """
    cursor = 0
    for expected in expected_calls:
        found = False
        while cursor < len(actual_calls):
            tool, args, status = actual_calls[cursor]
            cursor += 1
            if tool == expected.tool and status != "error" and _args_match(
                expected.args_match, args
            ):
                found = True
                break
        if not found:
            return GradeResult(
                passed=False,
                reason=(
                    f"expected call to {expected.tool!r} with args matching "
                    f"{expected.args_match!r} never found"
                ),
            )
    return GradeResult(passed=True)


__all__ = ["CallRecord", "GradeResult", "grade_calls"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_grading.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/eval/grading.py tests/test_eval_grading.py
git commit -m "feat(eval): ordered subsequence grading against observed tool calls"
```

---

### Task 3: Report models + rendering

**Files:**
- Create: `runtime/eval/report.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Produces: `VariantResult`, `TaskResult` (with `.all_passed` property), `EvalReport` (with `.tgc`/`.sgc` properties), `render_console(report: EvalReport) -> str`, `write_json(report: EvalReport, out_dir: Path) -> Path` — consumed by Task 4 (runner) and Task 5 (CLI).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_report.py`:
```python
"""EvalReport TGC/SGC computation and rendering."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.eval.report import EvalReport, TaskResult, VariantResult, render_console, write_json

pytestmark = pytest.mark.unit


def _report() -> EvalReport:
    return EvalReport(
        provider="ollama",
        model="gemma4:e2b-mlx",
        started_at=datetime(2026, 8, 20, 14, 30, tzinfo=UTC),
        tasks=(
            TaskResult(
                task_id="list_downloads",
                description="List files.",
                variants=(
                    VariantResult(task_id="list_downloads", variant_text="v1", passed=True, duration_s=1.0),
                    VariantResult(task_id="list_downloads", variant_text="v2", passed=True, duration_s=1.0),
                    VariantResult(task_id="list_downloads", variant_text="v3", passed=True, duration_s=1.0),
                ),
            ),
            TaskResult(
                task_id="search_then_read",
                description="Search then read.",
                variants=(
                    VariantResult(task_id="search_then_read", variant_text="v1", passed=True, duration_s=1.0),
                    VariantResult(
                        task_id="search_then_read",
                        variant_text="v2",
                        passed=False,
                        reason="files_read never called",
                        duration_s=1.0,
                    ),
                ),
            ),
        ),
    )


def test_tgc_is_per_run_pass_fraction() -> None:
    report = _report()
    # 4 of 5 total variant runs passed.
    assert report.tgc == pytest.approx(4 / 5)


def test_sgc_is_per_task_all_pass_fraction() -> None:
    report = _report()
    # Only list_downloads has every variant passing; search_then_read does not.
    assert report.sgc == pytest.approx(1 / 2)


def test_task_result_all_passed_true_when_every_variant_passes() -> None:
    task = _report().tasks[0]
    assert task.all_passed is True


def test_task_result_all_passed_false_when_any_variant_fails() -> None:
    task = _report().tasks[1]
    assert task.all_passed is False


def test_empty_report_metrics_do_not_divide_by_zero() -> None:
    report = EvalReport(
        provider="ollama", model="x", started_at=datetime(2026, 1, 1, tzinfo=UTC), tasks=()
    )
    assert report.tgc == 0.0
    assert report.sgc == 0.0


def test_render_console_includes_metrics_and_task_lines() -> None:
    text = render_console(_report())
    assert "TGC" in text
    assert "SGC" in text
    assert "list_downloads" in text
    assert "search_then_read" in text
    assert "3/3" in text
    assert "1/2" in text


def test_write_json_round_trips(tmp_path: Path) -> None:
    report = _report()
    path = write_json(report, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["provider"] == "ollama"
    assert len(data["tasks"]) == 2


def test_write_json_filename_is_safe_and_stamped(tmp_path: Path) -> None:
    report = EvalReport(
        provider="openrouter",
        model="x-ai/grok-4.1-fast",
        started_at=datetime(2026, 8, 20, 14, 30, 0, tzinfo=UTC),
        tasks=(),
    )
    path = write_json(report, tmp_path)
    assert "/" not in path.name.replace(str(tmp_path), "")
    assert "2026-08-20" in path.name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runtime.eval.report'`.

- [ ] **Step 3: Write the implementation**

Create `runtime/eval/report.py`:
```python
"""Eval run results: per-variant, per-task, and the whole-report TGC/SGC rollup."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class VariantResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    variant_text: str
    passed: bool
    reason: str = ""
    duration_s: float = Field(ge=0.0)


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    description: str
    variants: tuple[VariantResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(v.passed for v in self.variants)


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    started_at: datetime
    tasks: tuple[TaskResult, ...]

    @property
    def tgc(self) -> float:
        total = sum(len(t.variants) for t in self.tasks)
        if total == 0:
            return 0.0
        passed = sum(1 for t in self.tasks for v in t.variants if v.passed)
        return passed / total

    @property
    def sgc(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.all_passed) / len(self.tasks)


def render_console(report: EvalReport) -> str:
    total_runs = sum(len(t.variants) for t in report.tasks)
    lines = [
        f"Pinned: {report.provider} / {report.model}  |  "
        f"{len(report.tasks)} tasks, {total_runs} runs",
        "",
        f"TGC (per-run):          {report.tgc:.1%}",
        f"SGC (per-task, strict): {report.sgc:.1%}",
        "",
    ]
    for t in report.tasks:
        n_pass = sum(1 for v in t.variants if v.passed)
        n_total = len(t.variants)
        if n_pass == n_total:
            status = "PASS"
        elif n_pass == 0:
            status = "FAIL"
        else:
            status = "PARTIAL"
        lines.append(f"{t.task_id:<24} {n_pass}/{n_total} variants  {status}")
    return "\n".join(lines)


def write_json(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_model = report.model.replace("/", "-").replace(":", "-")
    safe_provider = report.provider.replace("/", "-").replace(":", "-")
    path = out_dir / f"{stamp}-{safe_provider}-{safe_model}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


__all__ = ["EvalReport", "TaskResult", "VariantResult", "render_console", "write_json"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_report.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/eval/report.py tests/test_eval_report.py
git commit -m "feat(eval): report models with TGC/SGC computation and rendering"
```

---

### Task 4: Runner — sandbox setup, observing harness, dispatch-and-grade

**Files:**
- Create: `runtime/eval/runner.py`
- Test: none (this task's logic requires a live model to exercise meaningfully — matches the spec's stated testing boundary: only `cli.py`'s end-to-end path is live-only, but the dispatcher-building/dispatching glue in `runner.py` has no fake-model seam by design, since it deliberately reuses `build_harness_dispatcher`'s real production wiring). Verify by import and a manual smoke run in Task 6, not by a unit test suite here.

**Interfaces:**
- Consumes: `EvalTask`, `substitute_sandbox` (Task 1); `grade_calls`, `CallRecord` (Task 2); `VariantResult`, `TaskResult` (Task 3); `build_harness_dispatcher` (`runtime/chat/telegram/bot.py`, unchanged); `SkillRegistry`, `Tier1Loader`, `Tier3Store`, `FilesClient`, `EventStream`, `AegisConfig` (all pre-existing, unchanged).
- Produces: `run_task(cfg: AegisConfig, registry: SkillRegistry, tier1_loader: Tier1Loader, task: EvalTask) -> TaskResult` — consumed by Task 5 (CLI).

- [ ] **Step 1: Write the implementation directly** (no TDD — see Files note above)

Create `runtime/eval/runner.py`:
```python
"""Builds a real HarnessDispatcher per run, dispatches each task variant, grades it.

Reuses `build_harness_dispatcher` directly -- the real classifier, real
Tier1Reasoner, real synthesizer, whichever `smart_provider`/model the
operator's live config pins. This module requires a reachable model to do
anything meaningful and is not exercised by `pytest -m unit` (see
docs/superpowers/specs/2026-08-20-eval-harness-design.md, Testing section).
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.telegram.bot import build_harness_dispatcher
from runtime.config import AegisConfig
from runtime.events import EventStream
from runtime.eval.grading import CallRecord, grade_calls
from runtime.eval.report import TaskResult, VariantResult
from runtime.eval.tasks import EvalTask, ExpectedCall, substitute_sandbox
from runtime.files.client import FilesClient
from runtime.harness.adapter import HarnessAdapter
from runtime.harness.contract import ToolIntent, ToolResult
from runtime.skills.registry import SkillRegistry


class _ObservingHarness:
    """Wraps a real HarnessAdapter; records (tool, args, status) before delegating.

    Eval-only test-seam code. Production code and the real evidence ledger
    are untouched -- see the spec's Grading section for why this exists
    instead of reading tool-call arguments back from the ledger.
    """

    def __init__(self, inner: HarnessAdapter) -> None:
        self._inner = inner
        self.calls: list[CallRecord] = []

    def has_tool(self, name: str) -> bool:
        return self._inner.has_tool(name)

    def execute(self, intent: ToolIntent) -> ToolResult:
        result = self._inner.execute(intent)
        self.calls.append((intent.tool, dict(intent.args), result.status))
        return result


def _seed_fixture(task: EvalTask, sandbox: Path) -> None:
    for f in task.fixture.files:
        target = sandbox / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")


async def run_variant(
    cfg: AegisConfig,
    registry: SkillRegistry,
    tier1_loader: Tier1Loader,
    task: EvalTask,
    variant_text: str,
) -> VariantResult:
    """Run one (task, variant) pair against a fresh sandbox and fresh dispatcher."""
    t0 = time.monotonic()
    sandbox = Path(tempfile.mkdtemp(prefix="aegis-eval-"))
    try:
        _seed_fixture(task, sandbox)
        resolved_text = substitute_sandbox(variant_text, sandbox)

        tier3 = Tier3Store()
        events = EventStream(sandbox / "sessions")
        files_client = FilesClient(allowed_roots=[sandbox])

        dispatcher = build_harness_dispatcher(
            cfg,
            skill_registry=registry,
            tier3=tier3,
            tier1_loader=tier1_loader,
            files_client=files_client,
            events=events,
        )
        if dispatcher is None:
            return VariantResult(
                task_id=task.id,
                variant_text=variant_text,
                passed=False,
                reason="build_harness_dispatcher returned None (a hard dependency is unavailable)",
                duration_s=time.monotonic() - t0,
            )

        observing = _ObservingHarness(dispatcher._harness)  # noqa: SLF001 -- eval-only test seam
        dispatcher._harness = observing  # type: ignore[assignment]  # noqa: SLF001

        collected: list[str] = []

        async def _capture(text: str) -> None:
            collected.append(text)

        try:
            await dispatcher.dispatch(
                chat_id=1, user_text=resolved_text, message=None, reply=_capture
            )
        except Exception as exc:  # eval harness must never crash the batch
            return VariantResult(
                task_id=task.id,
                variant_text=variant_text,
                passed=False,
                reason=f"dispatch raised: {exc!r}",
                duration_s=time.monotonic() - t0,
            )

        resolved_expected = tuple(
            ExpectedCall(
                tool=ec.tool,
                args_match=substitute_sandbox(ec.args_match, sandbox),
            )
            for ec in task.expected_calls
        )
        grade = grade_calls(resolved_expected, observing.calls)
        return VariantResult(
            task_id=task.id,
            variant_text=variant_text,
            passed=grade.passed,
            reason=grade.reason,
            duration_s=time.monotonic() - t0,
        )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


async def run_task(
    cfg: AegisConfig,
    registry: SkillRegistry,
    tier1_loader: Tier1Loader,
    task: EvalTask,
) -> TaskResult:
    """Run every variant of one task, sequentially (no concurrency -- keeps
    live-model call ordering predictable and easy to read in the console
    report as it streams)."""
    results: list[VariantResult] = []
    for variant_text in task.variants:
        results.append(await run_variant(cfg, registry, tier1_loader, task, variant_text))
    return TaskResult(task_id=task.id, description=task.description, variants=tuple(results))


__all__ = ["run_task", "run_variant"]
```

- [ ] **Step 2: Verify by import**

Run: `.venv/bin/python -c "from runtime.eval.runner import run_task, run_variant; print('ok')"`
Expected: `ok` — confirms no import errors, no syntax errors, all referenced names exist (`build_harness_dispatcher`, `FilesClient`, etc.).

- [ ] **Step 3: Commit**

```bash
git add runtime/eval/runner.py
git commit -m "feat(eval): runner builds real HarnessDispatcher, grades via observed calls"
```

---

### Task 5: CLI entrypoint

**Files:**
- Create: `runtime/eval/cli.py`
- Modify: `Makefile` (add `eval` target)

**Interfaces:**
- Consumes: `load_tasks` (Task 1); `run_task` (Task 4); `EvalReport`, `render_console`, `write_json` (Task 3); `get_config`, `ModelRouter`, `ModelTier` (pre-existing); `SkillRegistry.from_directory`, `Tier1Loader` (pre-existing).
- Produces: `main() -> int`, the `python -m runtime.eval.cli` entrypoint.

- [ ] **Step 1: Write the implementation**

Create `runtime/eval/cli.py`:
```python
"""Live-model benchmark entrypoint. Slow, costs real tokens against a live model.

Usage:
    python -m runtime.eval.cli [--tasks-dir DIR] [--out-dir DIR] [--yes]

Never runs in CI, never part of `pytest -m unit`. See
docs/superpowers/specs/2026-08-20-eval-harness-design.md.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.config import get_config
from runtime.eval.report import EvalReport, TaskResult, render_console, write_json
from runtime.eval.runner import run_task
from runtime.eval.tasks import load_tasks
from runtime.llm.router import ModelRouter, ModelTier
from runtime.skills.registry import SkillRegistry

_DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "tasks"
_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "results"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aegis eval harness (live models).")
    parser.add_argument("--tasks-dir", type=Path, default=_DEFAULT_TASKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument(
        "--yes", action="store_true", help="Skip the cost-confirmation prompt."
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    tasks = load_tasks(args.tasks_dir)
    if not tasks:
        print(f"No tasks found under {args.tasks_dir}", file=sys.stderr)
        return 1

    target = ModelRouter(cfg).route(ModelTier.SMART)
    total_runs = sum(len(t.variants) for t in tasks)
    print(
        f"Pinned: {target.provider} / {target.model}  |  "
        f"{len(tasks)} tasks, {total_runs} live model calls (at least)"
    )
    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 0

    registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
    tier1_loader = Tier1Loader(cfg.storage.workspace)

    task_results: list[TaskResult] = []
    for task in tasks:
        print(f"Running {task.id}...")
        result = await run_task(cfg, registry, tier1_loader, task)
        task_results.append(result)

    report = EvalReport(
        provider=target.provider,
        model=target.model,
        started_at=datetime.now(UTC),
        tasks=tuple(task_results),
    )
    print()
    print(render_console(report))
    path = write_json(report, args.out_dir)
    print(f"\nWritten: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

In `Makefile`, add `eval` to the `.PHONY` line (currently `.PHONY: help setup venv install bootstrap doctor lint type test test-unit test-e2e security run reflect review harness apply clean` — append `eval`), and add the target after `run:`:
```makefile
eval:  ## Live-model benchmark of the multi-step harness (slow, costs tokens)
	$(PY) -m runtime.eval.cli
```

- [ ] **Step 2: Verify by import and `--help`**

Run: `.venv/bin/python -m runtime.eval.cli --help`
Expected: argparse help text listing `--tasks-dir`, `--out-dir`, `--yes`, exit code 0.

Run: `make eval` with no tasks present yet (Task 6 not done) — expected: prints "No tasks found under .../eval/tasks", exits 1. This is expected at this point in the plan; Task 6 adds the tasks.

- [ ] **Step 3: Commit**

```bash
git add runtime/eval/cli.py Makefile
git commit -m "feat(eval): CLI entrypoint with cost-confirmation prompt"
```

---

### Task 6: V1 task content + `.gitignore`

**Files:**
- Create: `eval/tasks/list_downloads.yaml`
- Create: `eval/tasks/read_file.yaml`
- Create: `eval/tasks/search_files.yaml`
- Create: `eval/tasks/time_check.yaml`
- Create: `eval/tasks/search_then_read.yaml`
- Create: `eval/tasks/list_then_read.yaml`
- Modify: `.gitignore` (ignore `eval/results/`)

**Interfaces:** None — these are data files, no code interface. `load_tasks` (Task 1) is the consumer.

- [ ] **Step 1: Write the task files**

`eval/tasks/list_downloads.yaml`:
```yaml
id: list_downloads
description: "List files in a folder."
fixture:
  files:
    - path: "report.txt"
      content: "hello"
    - path: "notes.md"
      content: "notes"
variants:
  - "list files in {sandbox}"
  - "what's in {sandbox}?"
  - "show me the files under {sandbox}"
expected_calls:
  - tool: files_list
    args_match: {path: "{sandbox}"}
```

`eval/tasks/read_file.yaml`:
```yaml
id: read_file
description: "Read the contents of a specific file."
fixture:
  files:
    - path: "report.txt"
      content: "Q3 numbers look good."
variants:
  - "read {sandbox}/report.txt"
  - "what does {sandbox}/report.txt say?"
  - "open {sandbox}/report.txt and tell me what's in it"
expected_calls:
  - tool: files_read
    args_match: {path: "{sandbox}/report.txt"}
```

`eval/tasks/search_files.yaml`:
```yaml
id: search_files
description: "Search a folder for files matching a pattern."
fixture:
  files:
    - path: "notes/CT-001-notes.md"
      content: "notes about CT-001"
    - path: "notes/other.md"
      content: "unrelated"
variants:
  - "search {sandbox}/notes for CT-001"
  - "find files about CT-001 in {sandbox}/notes"
  - "look for CT-001 in {sandbox}/notes"
expected_calls:
  - tool: files_search
    args_match: {glob: "*CT-001*"}
```

`eval/tasks/time_check.yaml`:
```yaml
id: time_check
description: "Ask the current time -- no fixture needed."
variants:
  - "what time is it?"
  - "tell me the current time"
expected_calls:
  - tool: time
    args_match: {}
```

`eval/tasks/search_then_read.yaml`:
```yaml
id: search_then_read
description: "Multi-step: search for a file, then read the result. Exercises HarnessDispatcher._run_multi_step."
fixture:
  files:
    - path: "notes/CT-001-notes.md"
      content: "Some notes about CT-001."
    - path: "notes/other.md"
      content: "Unrelated notes."
variants:
  - "find files about CT-001 in {sandbox}/notes and read the first one"
  - "search {sandbox}/notes for CT-001 and open the top match"
expected_calls:
  - tool: files_search
    args_match: {glob: "*CT-001*"}
  - tool: files_read
    args_match: {}
```

`eval/tasks/list_then_read.yaml`:
```yaml
id: list_then_read
description: "Multi-step: list a folder, then read one of the listed files. Exercises HarnessDispatcher._run_multi_step."
fixture:
  files:
    - path: "report.txt"
      content: "Q3 numbers look good."
variants:
  - "list files in {sandbox} and then read report.txt"
  - "show me what's in {sandbox}, then open report.txt and tell me what it says"
expected_calls:
  - tool: files_list
    args_match: {path: "{sandbox}"}
  - tool: files_read
    args_match: {path: "{sandbox}/report.txt"}
```

In `.gitignore`, add (in the "Generated artifacts" section, alongside the existing `coding_harness/diffs/*.patch` entries):
```
eval/results/
```

- [ ] **Step 2: Verify tasks load and pass schema validation**

Run: `.venv/bin/python -c "from runtime.eval.tasks import load_tasks; from pathlib import Path; ts = load_tasks(Path('eval/tasks')); print(len(ts), [t.id for t in ts])"`
Expected: `6 ['list_downloads', 'list_then_read', 'read_file', 'search_files', 'search_then_read', 'time_check']` (alphabetical by filename).

- [ ] **Step 3: Commit**

```bash
git add eval/tasks/ .gitignore
git commit -m "feat(eval): v1 task set (4 single-tool, 2 multi-step chains)"
```

---

### Task 7: Full verification + live smoke run

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest -m unit -q`
Expected: PASS, 0 failures — includes the new `tests/test_eval_tasks.py`, `tests/test_eval_grading.py`, `tests/test_eval_report.py`.

- [ ] **Step 2: Run lint and type check**

Run: `make lint` and `make type`
Expected: no new findings in `runtime/eval/*.py` or the test files (pre-existing findings in unrelated files, if any, are out of scope).

- [ ] **Step 3: Live smoke run** — flag this back to the coordinating session rather than having a subagent run it unattended; it calls a real model and costs real tokens/time.

Run: `make eval` (or `python -m runtime.eval.cli`), confirm the cost prompt, let it run against whichever `smart_provider`/model is currently pinned. Confirm: the console table renders, TGC/SGC percentages are between 0-100%, a JSON file lands in `eval/results/`, and no run crashes the batch (a task erroring should show as a FAIL with a reason, not an unhandled exception killing the process).

- [ ] **Step 4: No commit** — verification only.
