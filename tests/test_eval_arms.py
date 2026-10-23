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
        ("full", "planner", Guardrails(plan_lock=True)),
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


@pytest.mark.parametrize("arm", ["native", "native-guard", "native-guard-synth"])
def test_measured_native_arms_keep_the_bare_prompt(arm: str) -> None:
    """Nights 1-2 measured these arms with the bare prompt; a rerun must too."""
    assert _apply_arm(_cfg(), arm).models.native_prompt == "bare"


def test_native_guard_id_is_native_guard_with_the_identity_prompt() -> None:
    cfg = _apply_arm(_cfg(), "native-guard-id")
    assert cfg.harness_mode_for(M) == "native"
    assert cfg.guardrails_for(M) == _apply_arm(_cfg(), "native-guard").guardrails_for(M)
    assert cfg.models.native_prompt == "identity"


def test_every_arm_reads_the_shipped_template_never_the_operator_workspace(
    tmp_path: Path,
) -> None:
    """The eval measures what a fresh install sends: the repo's Aegis template, pinned by
    the commit. It never reads this machine's IDENTITY.md (a private persona, different
    per machine) -- in the native prompt or in reply synthesis, for any arm."""
    from runtime.eval.cli import _TEMPLATE_WORKSPACE, _tier1_root

    cfg = AegisConfig.model_validate({"storage": {"workspace": str(tmp_path / "private")}})
    assert _tier1_root() == _TEMPLATE_WORKSPACE
    assert _tier1_root() != cfg.storage.workspace
    assert "**Name:** Aegis" in (_TEMPLATE_WORKSPACE / "IDENTITY.md").read_text(encoding="utf-8")


def test_production_defaults_to_the_identity_prompt() -> None:
    cfg = AegisConfig()
    assert cfg.models.native_prompt is None  # unset: defer to the profile, then identity
    assert cfg.native_prompt_for(M) == "identity"


def test_an_arm_beats_a_models_profile_prompt() -> None:
    """Eval arms pin the prompt globally, so a production profile cannot leak into a rerun."""
    cfg = AegisConfig(
        model_profiles={M: ModelProfile(harness_mode="native", native_prompt="bare")}
    )
    assert _apply_arm(cfg, "native-guard-id").native_prompt_for(M) == "identity"
    assert _apply_arm(cfg, "native-guard").native_prompt_for(M) == "bare"


def test_bare_stays_the_eval_only_control() -> None:
    from runtime.eval import bare

    assert "bare" in ARMS
    assert _apply_arm(_cfg(), "bare") == _cfg()
    assert _variant_runner("bare") is bare.run_variant_bare


@pytest.mark.parametrize(
    "arm", ["native", "native-guard", "native-guard-synth", "native-guard-id", "bare"]
)
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
