"""Per-model config profiles — ModelProfile + _coerce_model_profiles + think_for.

Motivation, measured 2026-08-25 across the six-task suite: the same setting
helps one model and breaks another, so a tier-wide value cannot be right.

    qwen3-vl:4b     think=False  TGC 40.0% -> 80.0%
    qwen3.5:4b-mlx  think=False  in-budget 73.3% -> 6.7%

Keys match on exact model id. Family prefixes are deliberately NOT supported:
`qwen3-vl:4b` and `qwen3.5:4b-mlx` share a vendor and diverge sharply, so
shared-by-default would encode an assumption the data contradicts.
"""
from __future__ import annotations

import pytest

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ModelProfile,
    _coerce_model_profiles,
)

pytestmark = pytest.mark.unit


# --- _coerce_model_profiles: a half-edited config.json must still boot ---


def test_missing_section_yields_no_profiles() -> None:
    assert _coerce_model_profiles(None) == {}


def test_non_dict_section_is_ignored() -> None:
    assert _coerce_model_profiles(["qwen3-vl:4b"]) == {}
    assert _coerce_model_profiles("qwen3-vl:4b") == {}


def test_profiles_are_parsed_by_exact_model_id() -> None:
    profiles = _coerce_model_profiles(
        {"qwen3-vl:4b": {"think": False}, "qwen3.5:4b-mlx": {"think": True}}
    )
    assert profiles["qwen3-vl:4b"].think is False
    assert profiles["qwen3.5:4b-mlx"].think is True


def test_empty_profile_body_means_no_override() -> None:
    assert _coerce_model_profiles({"gemma4:e4b-mlx": {}})["gemma4:e4b-mlx"].think is None


def test_unparseable_entry_is_dropped_not_raised() -> None:
    """One bad entry must not stop the process booting."""
    profiles = _coerce_model_profiles(
        {"good:1b": {"think": False}, "bad:1b": {"think": "banana"}}
    )
    assert "good:1b" in profiles
    assert "bad:1b" not in profiles


def test_non_dict_profile_body_is_dropped() -> None:
    profiles = _coerce_model_profiles({"good:1b": {"think": True}, "bad:1b": "nope"})
    assert list(profiles) == ["good:1b"]


def test_unknown_profile_fields_are_rejected() -> None:
    """extra=forbid: a typo'd key must not silently do nothing forever."""
    profiles = _coerce_model_profiles({"a:1b": {"thnik": False}})
    assert profiles == {}


# --- think_for: precedence ---------------------------------------------


def _config(*, env_think: bool | None, profiles: dict[str, ModelProfile]) -> AegisConfig:
    base = AegisConfig()
    return base.model_copy(
        update={
            "models": ModelConfig(smart_think=env_think),
            "model_profiles": profiles,
        }
    )


def test_profile_applies_when_env_is_unset() -> None:
    cfg = _config(env_think=None, profiles={"qwen3-vl:4b": ModelProfile(think=False)})
    assert cfg.think_for("qwen3-vl:4b") is False


def test_env_overrides_a_profile() -> None:
    """MODEL_SMART_THINK stays the operator's escape hatch for a debugging
    session, matching how MODEL_SMART_LOCAL already beats config.json."""
    cfg = _config(env_think=True, profiles={"qwen3-vl:4b": ModelProfile(think=False)})
    assert cfg.think_for("qwen3-vl:4b") is True


def test_env_false_overrides_a_profile_too() -> None:
    """False is a real setting, not 'unset' -- it must win like True does."""
    cfg = _config(env_think=False, profiles={"qwen3.5:4b-mlx": ModelProfile(think=True)})
    assert cfg.think_for("qwen3.5:4b-mlx") is False


def test_model_without_a_profile_gets_no_override() -> None:
    cfg = _config(env_think=None, profiles={"qwen3-vl:4b": ModelProfile(think=False)})
    assert cfg.think_for("gemma4:e4b-mlx") is None


def test_no_profiles_and_no_env_is_none() -> None:
    assert _config(env_think=None, profiles={}).think_for("anything:1b") is None


def test_env_applies_to_models_with_no_profile() -> None:
    cfg = _config(env_think=False, profiles={})
    assert cfg.think_for("gemma4:e2b-mlx") is False


def test_matching_is_exact_not_prefix() -> None:
    """`qwen3-vl:4b` and `qwen3.5:4b-mlx` need opposite settings -- a family
    prefix match would hand one of them the other's value."""
    cfg = _config(env_think=None, profiles={"qwen3-vl:4b": ModelProfile(think=False)})
    assert cfg.think_for("qwen3-vl:4b-instruct") is None
    assert cfg.think_for("qwen3-vl") is None


# --- Wiring: the resolved value must reach the reasoner -----------------


def test_construction_sites_resolve_think_per_model() -> None:
    """Both Tier1Reasoner sites must go through `think_for`.

    A source-level check rather than a behavioural one: constructing the real
    dispatcher needs a live client and a skills catalog, and a stubbed one
    would only prove the stub forwards its own argument. The failure this
    guards is specific and silent -- a site reading `cfg.models.smart_think`
    directly leaves every profile inert with nothing to observe at runtime.
    """
    from pathlib import Path

    for rel in ("runtime/chat/telegram/bot.py", "runtime/chat/cli.py"):
        src = Path(rel).read_text(encoding="utf-8")
        assert "think=cfg.models.smart_think" not in src, f"{rel} bypasses think_for"
        assert "think_for(" in src, f"{rel} does not resolve per-model think"
