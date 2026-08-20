"""EvalReport TGC/SGC computation and rendering."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.eval.report import (
    EvalReport,
    ObservedCall,
    TaskResult,
    VariantResult,
    render_console,
    write_json,
)

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
                    VariantResult(
                        task_id="list_downloads",
                        variant_text="v1",
                        passed=True,
                        duration_s=1.0,
                    ),
                    VariantResult(
                        task_id="list_downloads",
                        variant_text="v2",
                        passed=True,
                        duration_s=1.0,
                    ),
                    VariantResult(
                        task_id="list_downloads",
                        variant_text="v3",
                        passed=True,
                        duration_s=1.0,
                    ),
                ),
            ),
            TaskResult(
                task_id="search_then_read",
                description="Search then read.",
                variants=(
                    VariantResult(
                        task_id="search_then_read",
                        variant_text="v1",
                        passed=True,
                        duration_s=1.0,
                    ),
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


def test_variant_result_actual_calls_defaults_to_empty() -> None:
    """Existing construction sites (runner.py's early-return paths, and
    every pre-existing test/fixture in this file) don't pass `actual_calls`
    -- it must default rather than become required."""
    variant = VariantResult(
        task_id="t", variant_text="v", passed=True, duration_s=1.0
    )
    assert variant.actual_calls == ()


def test_variant_result_actual_calls_round_trips_through_json(tmp_path: Path) -> None:
    """The raw observed tool-call sequence must survive write_json, so a
    failed variant's JSON shows what the model actually did, not just the
    grader's pass/fail reason."""
    variant = VariantResult(
        task_id="search_then_read",
        variant_text="v2",
        passed=False,
        reason="files_read never called",
        duration_s=1.0,
        actual_calls=(
            ObservedCall(tool="files_search", args={"pattern": "CT-001"}, status="ok"),
        ),
    )
    report = EvalReport(
        provider="ollama",
        model="gemma4:e2b-mlx",
        started_at=datetime(2026, 8, 20, 14, 30, tzinfo=UTC),
        tasks=(
            TaskResult(task_id="search_then_read", description="d", variants=(variant,)),
        ),
    )
    path = write_json(report, tmp_path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    calls = reloaded["tasks"][0]["variants"][0]["actual_calls"]
    assert calls == [{"tool": "files_search", "args": {"pattern": "CT-001"}, "status": "ok"}]


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
