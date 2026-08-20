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
