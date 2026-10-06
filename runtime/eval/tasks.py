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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FillerSpec(BaseModel):
    """A generated file: `count` filler lines with one needle line swapped in.

    Tool results are capped -- `files_read` at 3,500 characters, the planner
    prompt at 1,024 -- so *where* a needle sits decides which harness can see it.
    Writing a 60 KB log into a task YAML by hand would be unreviewable, so the
    file is generated: `line` is a `str.format` template over the line index `i`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    line: str = Field(min_length=1)
    count: int = Field(ge=1, le=200_000)
    needle: str = Field(min_length=1)
    needle_at: int = Field(ge=0, description="0-based index of the line the needle replaces.")

    @model_validator(mode="after")
    def _needle_inside_file(self) -> FillerSpec:
        if self.needle_at >= self.count:
            raise ValueError(f"needle_at {self.needle_at} is outside {self.count} lines")
        return self

    def render(self) -> str:
        lines = [self.line.format(i=i) for i in range(self.count)]
        lines[self.needle_at] = self.needle
        return "\n".join(lines)


class FixtureFile(BaseModel):
    """One file seeded into the per-run sandbox before dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, description="Relative to the sandbox root.")
    content: str = Field(default="")
    filler: FillerSpec | None = Field(
        default=None,
        description="Generate the content instead of writing it out; exclusive with `content`.",
    )

    @model_validator(mode="after")
    def _content_xor_filler(self) -> FixtureFile:
        if self.filler is not None and self.content:
            raise ValueError("a fixture file takes `content` or `filler`, not both")
        return self

    def render(self) -> str:
        """The file's text -- what `_seed_fixture` writes and what a reply is graded against."""
        return self.filler.render() if self.filler is not None else self.content

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
    allow_error: bool = Field(
        default=False,
        description=(
            "Count an attempt that came back with status `error`. For honesty "
            "probes, where the target does not exist: a structural grader can only "
            "ask that the model tried; whether it then told the truth is the reply "
            "grader's job."
        ),
    )


class EvalTask(BaseModel):
    """One task template: several phrasings, one expected tool-call sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    fixture: TaskFixture = Field(default_factory=TaskFixture)
    variants: tuple[str, ...] = Field(min_length=1)
    expected_calls: tuple[ExpectedCall, ...] = Field(min_length=1)
    answer_contains: tuple[str, ...] = Field(
        default=(),
        description=(
            "Substrings a correct final reply must contain (case-insensitive). "
            "Tool-call grading says whether the model did the work; this says "
            "whether what it then told the user is right. Empty means the task "
            "has no checkable answer (e.g. a pure side effect)."
        ),
    )

    unreachable: tuple[str, ...] = Field(
        default=(),
        description=(
            "Things the task asks for that do not exist, so the only honest reply "
            "admits it. Non-empty marks the task as an honesty probe: a reply that "
            "does not admit the absence has invented a result."
        ),
    )

    @field_validator("answer_contains", "unreachable")
    @classmethod
    def _entries_must_not_be_blank(cls, entries: tuple[str, ...]) -> tuple[str, ...]:
        # A blank fact is a substring of every reply and would mark all of them correct.
        if any(not entry.strip() for entry in entries):
            raise ValueError("answer_contains / unreachable entries must not be blank")
        return entries


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


def resolve_expected_calls(task: EvalTask, sandbox: Path) -> tuple[ExpectedCall, ...]:
    """The task's expected calls with `{sandbox}` replaced by the real sandbox path.

    Copies each call and updates only `args_match`, so every other field -- today
    `allow_error`, tomorrow whatever comes next -- survives. Both arms used to
    rebuild the `ExpectedCall` by hand and silently dropped fields they did not
    know about.
    """
    return tuple(
        ec.model_copy(update={"args_match": substitute_sandbox(ec.args_match, sandbox)})
        for ec in task.expected_calls
    )


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
