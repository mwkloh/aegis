"""Eval task YAML loader + `{sandbox}` substitution."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.eval.tasks import (
    EvalTask,
    ExpectedCall,
    FixtureFile,
    load_tasks,
    substitute_sandbox,
)

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
    _write(
        tmp_path / "b_task.yaml",
        "id: b\ndescription: b\nvariants: ['x']\nexpected_calls: "
        "[{tool: echo, args_match: {}}]\n",
    )
    _write(
        tmp_path / "a_task.yaml",
        "id: a\ndescription: a\nvariants: ['x']\nexpected_calls: "
        "[{tool: echo, args_match: {}}]\n",
    )
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
    with pytest.raises(ValidationError):
        EvalTask(
            id="x",
            description="x",
            variants=(),
            expected_calls=({"tool": "echo", "args_match": {}},),
        )


def test_evaltask_requires_at_least_one_expected_call() -> None:
    with pytest.raises(ValidationError):
        EvalTask(id="x", description="x", variants=("hi",), expected_calls=())


def test_fixturefile_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        FixtureFile(path="/etc/evil", content="x")


def test_fixturefile_rejects_home_relative_path() -> None:
    with pytest.raises(ValidationError):
        FixtureFile(path="~/evil", content="x")


def test_fixturefile_rejects_dotdot_segment() -> None:
    with pytest.raises(ValidationError):
        FixtureFile(path="../escape", content="x")


def test_fixturefile_rejects_nested_dotdot_segment() -> None:
    with pytest.raises(ValidationError):
        FixtureFile(path="notes/../../escape", content="x")


def test_fixturefile_accepts_relative_path() -> None:
    f = FixtureFile(path="notes/ok.md", content="x")
    assert f.path == "notes/ok.md"


# --- answer facts: what a correct reply must say ------------------------------
# Tool-call grading says whether the model *did* the work; these say whether what
# it then *told the user* is right. Without them a reply cannot be checked.


def test_evaltask_answer_contains_defaults_to_empty() -> None:
    task = EvalTask(
        id="t",
        description="d",
        variants=("v",),
        expected_calls=({"tool": "time", "args_match": {}},),  # type: ignore[arg-type]
    )
    assert task.answer_contains == ()


def test_load_tasks_reads_answer_contains(tmp_path: Path) -> None:
    (tmp_path / "t.yaml").write_text(
        "id: t\n"
        "description: d\n"
        "variants: ['v']\n"
        "answer_contains: ['7421', 'access code']\n"
        "expected_calls:\n"
        "  - tool: files_read\n"
        "    args_match: {}\n",
        encoding="utf-8",
    )
    (task,) = load_tasks(tmp_path)
    assert task.answer_contains == ("7421", "access code")


def test_load_tasks_reads_unreachable_and_allow_error(tmp_path: Path) -> None:
    (tmp_path / "t.yaml").write_text(
        "id: t\n"
        "description: d\n"
        "variants: ['v']\n"
        "unreachable: ['ghost.txt']\n"
        "expected_calls:\n"
        "  - tool: files_read\n"
        "    args_match: {path: ghost.txt}\n"
        "    allow_error: true\n",
        encoding="utf-8",
    )
    (task,) = load_tasks(tmp_path)
    assert task.unreachable == ("ghost.txt",)
    assert task.expected_calls[0].allow_error is True


def test_evaltask_unreachable_defaults_to_empty_and_rejects_blanks() -> None:
    base = {"id": "t", "description": "d", "variants": ("v",)}
    call = {"tool": "time", "args_match": {}}
    assert EvalTask(**base, expected_calls=(call,)).unreachable == ()  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EvalTask(**base, unreachable=(" ",), expected_calls=(call,))  # type: ignore[arg-type]


# --- large fixtures: a generated file with one needle in it --------------------
# Tool results are capped (files_read at 3,500 chars; the planner prompt at 1,024),
# so *where* the needle sits decides which harness can see it. Writing a 60 KB log
# into the YAML by hand would be unreviewable, so the fixture generates it.


def _filler(**kw: object) -> FixtureFile:
    spec = {"line": "INFO worker-{i} ok", "count": 10, "needle": "ERROR code=E4417", "needle_at": 3}
    spec.update(kw)
    return FixtureFile(path="app.log", filler=spec)  # type: ignore[arg-type]


def test_filler_renders_count_lines_with_the_needle_in_place() -> None:
    lines = _filler().render().split("\n")

    assert len(lines) == 10
    assert lines[3] == "ERROR code=E4417"
    assert lines[0] == "INFO worker-0 ok"
    assert lines[9] == "INFO worker-9 ok"


def test_plain_content_renders_as_itself() -> None:
    assert FixtureFile(path="a.txt", content="hello").render() == "hello"


def test_a_fixture_cannot_have_both_content_and_filler() -> None:
    with pytest.raises(ValidationError):
        FixtureFile(
            path="a.log",
            content="x",
            filler={"line": "l{i}", "count": 2, "needle": "n", "needle_at": 0},  # type: ignore[arg-type]
        )


def test_needle_position_must_fall_inside_the_file() -> None:
    with pytest.raises(ValidationError):
        _filler(needle_at=10)
    with pytest.raises(ValidationError):
        _filler(needle_at=-1)


def test_seeding_writes_the_generated_file(tmp_path: Path) -> None:
    from runtime.eval.runner import _seed_fixture
    from runtime.eval.tasks import TaskFixture

    task = EvalTask(
        id="t",
        description="d",
        variants=("v",),
        fixture=TaskFixture(files=(_filler(),)),
        expected_calls=({"tool": "files_read", "args_match": {}},),  # type: ignore[arg-type]
    )
    _seed_fixture(task, tmp_path)

    assert (tmp_path / "app.log").read_text(encoding="utf-8").split("\n")[3] == "ERROR code=E4417"


def test_resolve_expected_calls_keeps_every_flag(tmp_path: Path) -> None:
    """Both arms resolve `{sandbox}` in the expected calls before grading. Each used
    to rebuild the `ExpectedCall` by hand and silently dropped `allow_error`, so a
    graded attempt at a missing file scored as a failure (found by a smoke test of
    the honesty probes: 7 of 15 variants wrongly `tool_errored`)."""
    from runtime.eval.tasks import resolve_expected_calls

    task = EvalTask(
        id="t",
        description="d",
        variants=("v",),
        expected_calls=(
            ExpectedCall(
                tool="files_read", args_match={"path": "{sandbox}/ghost.txt"}, allow_error=True
            ),
            ExpectedCall(tool="files_list", args_match={"path": "{sandbox}"}),
        ),
    )
    first, second = resolve_expected_calls(task, tmp_path)

    assert first.args_match == {"path": f"{tmp_path}/ghost.txt"}
    assert first.allow_error is True
    assert second.args_match == {"path": str(tmp_path)}
    assert second.allow_error is False


def test_evaltask_answer_contains_rejects_blank_entries() -> None:
    """A blank fact would match every reply and silently mark everything correct."""
    with pytest.raises(ValidationError):
        EvalTask(
            id="t",
            description="d",
            variants=("v",),
            answer_contains=("",),
            expected_calls=({"tool": "time", "args_match": {}},),  # type: ignore[arg-type]
        )
