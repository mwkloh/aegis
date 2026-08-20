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
