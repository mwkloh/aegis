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
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FixtureFile(BaseModel):
    """One file seeded into the per-run sandbox before dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, description="Relative to the sandbox root.")
    content: str = Field(default="")

    @field_validator("path")
    @classmethod
    def _reject_escaping_path(cls, v: str) -> str:
        """Reject anything that could land `sandbox / path` outside the sandbox.

        `_seed_fixture` (runner.py) does `sandbox / f.path` to seed this file.
        `pathlib` discards the left operand of `/` when the right operand is
        absolute, and a `..` segment walks back up past the sandbox root --
        either would let a fixture write outside the sandbox, which this
        codebase treats as non-negotiable. Mirrors the reasoning (though not
        the code) of `_has_dotdot_segment`/`_check_path_arg` in
        `runtime/harness/tools/command_tool.py`.
        """
        if v.startswith("/") or v.startswith("~"):
            raise ValueError(
                f"fixture path must be relative to the sandbox root, got {v!r}"
            )
        if ".." in v.split("/"):
            raise ValueError(f"fixture path contains a '..' segment: {v!r}")
        return v


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
