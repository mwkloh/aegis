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
    VariantTelemetry,
    render_console,
    write_json,
)
from runtime.llm.telemetry import CallTelemetry

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


# --- Variant telemetry --------------------------------------------------
# docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md


def test_variant_result_without_telemetry_still_validates() -> None:
    """Historical results predate instrumentation; they must keep parsing."""
    v = VariantResult(
        task_id="time_check", variant_text="what time is it?", passed=True, duration_s=1.0
    )
    assert v.telemetry is None


def test_existing_result_files_still_parse() -> None:
    """Guards the absent-vs-zero trap: `actual_calls` was added in 78f84e0, so
    older files lack fields entirely. A schema change must not orphan them."""
    results = Path(__file__).resolve().parent.parent / "eval" / "results"
    files = sorted(results.glob("*.json"))
    if not files:
        pytest.skip("no historical results checked out")
    for path in files:
        EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_variant_telemetry_aggregates_across_calls() -> None:
    t = VariantTelemetry.from_calls(
        [
            CallTelemetry(
                provider="ollama", model="m", wall_ms=1000, load_ms=800, eval_ms=200
            ),
            CallTelemetry(
                provider="ollama", model="m", wall_ms=3000, load_ms=0, eval_ms=3000
            ),
        ]
    )
    assert t.model_calls == 2
    assert t.load_ms_total == 800
    assert t.eval_ms_total == 3200
    assert t.max_call_wall_ms == 3000
    assert not t.any_timed_out


def test_variant_telemetry_flags_retry_exhaustion() -> None:
    """The qwen3-vl:4b shape: every call dies on a retry-exhausted timeout."""
    t = VariantTelemetry.from_calls(
        [
            CallTelemetry(
                provider="ollama",
                model="qwen3-vl:4b",
                wall_ms=90_600,
                attempts=3,
                timed_out=True,
            )
        ]
    )
    assert t.any_timed_out
    assert t.timed_out_calls == 1


def test_variant_telemetry_reports_worst_thinking_share() -> None:
    """Drives the deferred thinking-mode decision: did any call burn its budget?"""
    t = VariantTelemetry.from_calls(
        [
            CallTelemetry(
                provider="ollama", model="m", wall_ms=10, tokens_out=100, thinking_tokens=0
            ),
            CallTelemetry(
                provider="ollama", model="m", wall_ms=10, tokens_out=512, thinking_tokens=512
            ),
        ]
    )
    assert t.max_thinking_token_share == pytest.approx(1.0)


def test_variant_telemetry_from_no_calls_is_none() -> None:
    """A variant that never reached a model records nothing, not zeros."""
    assert VariantTelemetry.from_calls([]) is None


def test_variant_result_with_telemetry_round_trips_json() -> None:
    v = VariantResult(
        task_id="t",
        variant_text="x",
        passed=False,
        duration_s=90.6,
        telemetry=VariantTelemetry.from_calls(
            [
                CallTelemetry(
                    provider="ollama",
                    model="qwen3-vl:4b",
                    wall_ms=90_600,
                    attempts=3,
                    timed_out=True,
                )
            ]
        ),
    )
    back = VariantResult.model_validate_json(v.model_dump_json())
    assert back.telemetry is not None
    assert back.telemetry.any_timed_out


def test_variant_result_carries_failure_kind() -> None:
    v = VariantResult(
        task_id="t",
        variant_text="x",
        passed=False,
        duration_s=90.6,
        failure_kind="timeout_exhausted",
    )
    assert VariantResult.model_validate_json(v.model_dump_json()).failure_kind == (
        "timeout_exhausted"
    )


def test_render_console_breaks_failures_down_by_kind() -> None:
    """A bare '0/2 FAIL' hides whether the model was cut off or chose wrong."""
    report = EvalReport(
        provider="ollama",
        model="qwen3-vl:4b",
        started_at=datetime(2026, 8, 24, tzinfo=UTC),
        tasks=(
            TaskResult(
                task_id="time_check",
                description="ask the time",
                variants=(
                    VariantResult(
                        task_id="time_check",
                        variant_text="a",
                        passed=False,
                        duration_s=92.0,
                        failure_kind="timeout_exhausted",
                    ),
                    VariantResult(
                        task_id="time_check",
                        variant_text="b",
                        passed=False,
                        duration_s=96.3,
                        failure_kind="timeout_exhausted",
                    ),
                ),
            ),
        ),
    )
    out = render_console(report)
    assert "timeout_exhausted" in out
    assert "2" in out


# --- Capability vs product budget ---------------------------------------
# Decision 2026-08-24: report both from one run rather than choosing.


def _variant(passed: bool, max_call_wall_ms: int | None) -> VariantResult:
    tel = (
        None
        if max_call_wall_ms is None
        else VariantTelemetry(
            model_calls=1,
            load_ms_total=0,
            eval_ms_total=0,
            max_call_wall_ms=max_call_wall_ms,
            timed_out_calls=0,
            max_thinking_token_share=0.0,
        )
    )
    return VariantResult(
        task_id="t", variant_text="x", passed=passed, duration_s=1.0, telemetry=tel
    )


def _report_of(*variants: VariantResult) -> EvalReport:
    return EvalReport(
        provider="ollama",
        model="m",
        started_at=datetime(2026, 8, 24, tzinfo=UTC),
        tasks=(TaskResult(task_id="t", description="d", variants=variants),),
    )


def test_fast_passing_runs_count_under_both_metrics() -> None:
    r = _report_of(_variant(True, 5_000), _variant(True, 9_000))
    assert r.tgc == pytest.approx(1.0)
    assert r.tgc_within_budget == pytest.approx(1.0)


def test_a_slow_pass_counts_for_capability_but_not_for_the_product_budget() -> None:
    """The whole point of the split: it succeeded, but not usably fast."""
    r = _report_of(_variant(True, 45_000))
    assert r.tgc == pytest.approx(1.0)
    assert r.tgc_within_budget == pytest.approx(0.0)


def test_a_failing_run_counts_for_neither() -> None:
    r = _report_of(_variant(False, 1_000))
    assert r.tgc == pytest.approx(0.0)
    assert r.tgc_within_budget == pytest.approx(0.0)


def test_uninstrumented_passing_runs_are_not_penalised() -> None:
    """Historical results carry no telemetry -- absent must not read as 'too slow'."""
    r = _report_of(_variant(True, None))
    assert r.tgc_within_budget == pytest.approx(1.0)


def test_render_console_reports_the_budget_metric() -> None:
    out = render_console(_report_of(_variant(True, 45_000)))
    assert "budget" in out.lower()


# --- Repeat runs (variance) ---------------------------------------------
# F8: qwen3.5-9b flipped which task failed between two consecutive runs.


def test_task_pass_rate_across_repeats() -> None:
    """With repeats, a task reports a rate, not a single verdict."""
    t = TaskResult(
        task_id="time_check",
        description="d",
        variants=(
            _variant(True, 1_000),
            _variant(False, 1_000),
            _variant(True, 1_000),
            _variant(True, 1_000),
        ),
    )
    assert t.pass_rate == pytest.approx(0.75)


def test_all_passed_still_means_every_variant() -> None:
    """SGC must not soften into 'mostly passed' when repeats are enabled."""
    t = TaskResult(
        task_id="t",
        description="d",
        variants=(_variant(True, 1_000), _variant(False, 1_000)),
    )
    assert t.pass_rate == pytest.approx(0.5)
    assert t.all_passed is False


def test_flaky_task_is_flagged_but_a_consistent_one_is_not() -> None:
    flaky = TaskResult(
        task_id="t",
        description="d",
        variants=(_variant(True, 1_000), _variant(False, 1_000)),
    )
    consistent = TaskResult(
        task_id="t",
        description="d",
        variants=(_variant(True, 1_000), _variant(True, 1_000)),
    )
    assert flaky.is_flaky is True
    assert consistent.is_flaky is False
