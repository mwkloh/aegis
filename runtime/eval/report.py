"""Eval run results: per-variant, per-task, and the whole-report TGC/SGC rollup."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from runtime.llm.telemetry import PRODUCTION_READ_TIMEOUT_S, CallTelemetry

# Mirrors `grading.FailureKind.HARNESS_UNAVAILABLE`; held as a literal because
# `grading` already imports this module.
_HARNESS_UNAVAILABLE = "harness_unavailable"


class ObservedCall(BaseModel):
    """One (tool, args, status) tuple as actually executed by `_ObservingHarness`.

    Distinct from `ExpectedCall` (runtime/eval/tasks.py) -- this is what the
    model really did, not what a task author declared it should do. Kept on
    `VariantResult` so a failed variant's JSON shows the real call sequence,
    not just the grader's pass/fail reason -- previously the JSON gave no
    way to tell, e.g., a skipped-a-required-step failure from a
    stopped-after-one-step failure without re-running the harness.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str
    args: dict[str, Any]
    status: str


class VariantTelemetry(BaseModel):
    """Model-call cost for one variant, aggregated from `CallTelemetry`.

    Exists to separate "the model could not do it" from "the harness cut it
    off". Before this, both reached the JSON as the same
    `reason="expected call ... never found"` string.

    `None` (rather than a zeroed instance) means the variant was never
    instrumented -- the same absent-vs-zero distinction that `actual_calls`
    got wrong for results written before `78f84e0`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: int = Field(ge=0)
    load_ms_total: int = Field(ge=0)
    eval_ms_total: int = Field(ge=0)
    max_call_wall_ms: int = Field(ge=0)
    timed_out_calls: int = Field(ge=0)
    max_thinking_token_share: float = Field(ge=0.0, le=1.0)
    truncated_calls: int = Field(default=0, ge=0)
    # On-device performance. Zero means "not recorded" (older files, or a call
    # that reported nothing), never "free".
    prompt_eval_ms_total: int = Field(default=0, ge=0)
    tokens_in_total: int = Field(default=0, ge=0)
    tokens_out_total: int = Field(default=0, ge=0)

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed: output tokens over provider-reported eval time.

        None when nothing was timed (a timeout, a hosted call, or an old file) --
        absent, not zero, so a missing measurement cannot read as a slow model.
        """
        if self.eval_ms_total <= 0 or self.tokens_out_total <= 0:
            return None
        return self.tokens_out_total / (self.eval_ms_total / 1000)

    @property
    def any_timed_out(self) -> bool:
        """At least one call died on a retry-exhausted timeout."""
        return self.timed_out_calls > 0

    @classmethod
    def from_calls(cls, calls: Sequence[CallTelemetry]) -> VariantTelemetry | None:
        if not calls:
            return None
        return cls(
            model_calls=len(calls),
            load_ms_total=sum(c.load_ms for c in calls),
            eval_ms_total=sum(c.eval_ms for c in calls),
            max_call_wall_ms=max(c.wall_ms for c in calls),
            timed_out_calls=sum(1 for c in calls if c.timed_out),
            max_thinking_token_share=max(c.thinking_token_share for c in calls),
            truncated_calls=sum(1 for c in calls if c.truncated_by_budget),
            prompt_eval_ms_total=sum(c.prompt_eval_ms for c in calls),
            tokens_in_total=sum(c.tokens_in for c in calls),
            tokens_out_total=sum(c.tokens_out for c in calls),
        )


class VariantResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    variant_text: str
    passed: bool
    reason: str = ""
    duration_s: float = Field(ge=0.0)
    actual_calls: tuple[ObservedCall, ...] = ()
    telemetry: VariantTelemetry | None = None
    failure_kind: str | None = Field(
        default=None,
        description=(
            "Why this variant failed, from `runtime.eval.grading.FailureKind`. "
            "Held as a plain string rather than the enum so `report` stays free "
            "of a `grading` import -- `grading` already imports this module for "
            "VariantTelemetry, and the cycle would be the only thing the typed "
            "field bought."
        ),
    )
    reply: str | None = Field(
        default=None,
        description=(
            "What the model said back to the user, newline-joined. Pass/fail "
            "grades tool calls, never the words, so a claim the model made "
            "without doing the work is only visible here. None means it said "
            "nothing -- or the file predates this field."
        ),
    )


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    description: str
    variants: tuple[VariantResult, ...]

    @property
    def all_passed(self) -> bool:
        return all(v.passed for v in self.variants)

    @property
    def pass_rate(self) -> float:
        """Fraction of variant runs that passed.

        With `--repeat` this is the number that carries information: a task
        that passes 3 of 4 runs is a different object from one that passes 0
        of 4, and `all_passed` flattens both to False. SGC deliberately keeps
        using `all_passed` -- it is defined as strict, and softening it here
        would silently redefine a published metric.
        """
        if not self.variants:
            return 0.0
        return sum(1 for v in self.variants if v.passed) / len(self.variants)

    @property
    def is_flaky(self) -> bool:
        """Both passed and failed at least once -- the F8 shape.

        `openrouter/qwen3.5-9b` failed `time_check` 0/2 on one run and 2/2 on
        the next, minutes apart. At n=1 that is indistinguishable from a
        regression; only repeats can tell them apart.
        """
        outcomes = {v.passed for v in self.variants}
        return len(outcomes) > 1


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    started_at: datetime
    tasks: tuple[TaskResult, ...]
    locality: Literal["local", "cloud", "hosted", "unknown"] = Field(
        default="unknown",
        description=(
            "Where the model ran. `local` = pulled onto this machine's Ollama; "
            "`cloud` = an Ollama cloud model proxied through the local daemon "
            "(`provider` still says ollama); `hosted` = a third-party API such as "
            "OpenRouter. The benchmark asks how small models run on-device, so a "
            "cloud or hosted result must never be read as one. Files written "
            "before the field existed are `unknown`."
        ),
    )
    arm: Literal[
        "full", "full-nolock", "native", "native-guard", "native-guard-synth", "bare"
    ] = Field(
        default="full",
        description=(
            "Which harness produced these numbers. `full` is the AEGIS planner "
            "with its guardrails; `bare` is a native-tool-calling loop without "
            "them (`runtime.eval.bare`). The tasks and models are the same, so "
            "without this two files for one model cannot be told apart. Files "
            "written before the field existed are all `full`. Non-bare arms are "
            "presets of the runtime harness config (spec 2026-10-06)."
        ),
    )

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

    @property
    def tgc_within_budget(self) -> float:
        """TGC restricted to runs that would also have fit the shipped timeout.

        `tgc` answers "can the model do this at all", measured under the eval's
        own generous budget. This answers "and would it have been usable",
        against `PRODUCTION_READ_TIMEOUT_S`. Reporting only the first flatters
        a slow model; reporting only the second reads a harness cutoff as
        incapability -- which is how `qwen3-vl:4b` came to be published as a
        flat 0%.

        A passing run with no telemetry counts as within budget: absent is not
        the same as too slow, and historical results carry no timings.
        """
        total = sum(len(t.variants) for t in self.tasks)
        if total == 0:
            return 0.0
        budget_ms = PRODUCTION_READ_TIMEOUT_S * 1000
        ok = sum(
            1
            for t in self.tasks
            for v in t.variants
            if v.passed
            and (v.telemetry is None or v.telemetry.max_call_wall_ms <= budget_ms)
        )
        return ok / total


def render_console(report: EvalReport) -> str:
    total_runs = sum(len(t.variants) for t in report.tasks)
    arm_note = " [bare arm]" if report.arm == "bare" else ""
    where_note = "" if report.locality == "unknown" else f" [{report.locality}]"
    lines = [
        f"Pinned: {report.provider} / {report.model}{arm_note}{where_note}  |  "
        f"{len(report.tasks)} tasks, {total_runs} runs",
        "",
        f"TGC (per-run):          {report.tgc:.1%}",
        f"SGC (per-task, strict): {report.sgc:.1%}",
        (
            f"TGC within {PRODUCTION_READ_TIMEOUT_S:.0f}s budget: "
            f"{report.tgc_within_budget:.1%}"
        ),
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
        flag = "  FLAKY" if t.is_flaky and n_pass else ""
        lines.append(f"{t.task_id:<24} {n_pass}/{n_total} variants  {status}{flag}")

    kinds = Counter(
        v.failure_kind
        for t in report.tasks
        for v in t.variants
        if not v.passed and v.failure_kind
    )
    if kinds:
        lines.extend(["", "Failures by kind:"])
        lines.extend(
            f"  {kind:<22} {count}" for kind, count in sorted(kinds.most_common())
        )

    unreached = sum(
        1
        for t in report.tasks
        for v in t.variants
        if v.failure_kind == _HARNESS_UNAVAILABLE
    )
    if unreached:
        lines.append(
            f"\n{unreached} variant(s) never reached the model -- the harness "
            "could not build its dispatcher. These measure the harness, "
            "not model capability; rerun them."
        )

    timed_out = sum(
        1
        for t in report.tasks
        for v in t.variants
        if v.telemetry is not None and v.telemetry.any_timed_out
    )
    if timed_out:
        lines.append(
            f"\n{timed_out} variant(s) hit a retry-exhausted timeout -- "
            "these measure the harness budget, not model capability."
        )
    return "\n".join(lines)


def write_json(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_model = report.model.replace("/", "-").replace(":", "-")
    safe_provider = report.provider.replace("/", "-").replace(":", "-")
    arm_suffix = "" if report.arm == "full" else f"-{report.arm}"
    path = out_dir / f"{stamp}-{safe_provider}-{safe_model}{arm_suffix}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


__all__ = [
    "EvalReport",
    "ObservedCall",
    "TaskResult",
    "VariantResult",
    "VariantTelemetry",
    "render_console",
    "write_json",
]
