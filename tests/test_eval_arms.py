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
