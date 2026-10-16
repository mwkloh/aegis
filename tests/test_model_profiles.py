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
    _coerce,
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


# --- num_ctx: the context window decides whether a model fits in RAM -----
# Measured 2026-10-03 on a 16 GB Mac Mini: `qwen3-vl:4b` defaults to a
# 262,144-token context, which Ollama sized at 43 GB (80% CPU) and then hung;
# at num_ctx=8192 the same model loaded at 4.2 GB, 100% GPU, and answered.


def test_profile_carries_num_ctx_independently_of_think() -> None:
    profile = _coerce_model_profiles({"qwen3-vl:4b": {"num_ctx": 16384}})["qwen3-vl:4b"]
    assert profile.num_ctx == 16384
    assert profile.think is None


def test_num_ctx_must_be_positive() -> None:
    """0 or negative would reach Ollama as a nonsense window; drop it at load."""
    assert _coerce_model_profiles({"a:1b": {"num_ctx": 0}}) == {}
    assert _coerce_model_profiles({"a:1b": {"num_ctx": -1}}) == {}


def test_num_ctx_for_resolves_by_exact_model_id() -> None:
    cfg = _config(env_think=None, profiles={"qwen3-vl:4b": ModelProfile(num_ctx=16384)})
    assert cfg.num_ctx_for("qwen3-vl:4b") == 16384
    assert cfg.num_ctx_for("qwen3-vl:4b-instruct") is None


def test_num_ctx_for_is_none_without_a_profile() -> None:
    assert _config(env_think=None, profiles={}).num_ctx_for("anything:1b") is None


# --- tool_result_chars: how much of a tool result the planner prompt carries ---
# Measured 2026-10-04 on this Mac Mini: prefill speed differs ~12x between the small
# models (gemma4:e2b-mlx ~2,500 tok/s, qwen3.5:9b ~185), so the same 3,500 characters
# costs +0.5 s on one and +6.4 s per planner call on another. The old fixed 1,024
# had no measurement behind it. See docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md
# section 14.


def test_profile_carries_tool_result_chars_independently() -> None:
    profile = _coerce_model_profiles({"qwen3.5:9b": {"tool_result_chars": 2048}})["qwen3.5:9b"]
    assert profile.tool_result_chars == 2048
    assert profile.think is None
    assert profile.num_ctx is None


def test_tool_result_chars_must_be_positive() -> None:
    assert _coerce_model_profiles({"a:1b": {"tool_result_chars": 0}}) == {}
    assert _coerce_model_profiles({"a:1b": {"tool_result_chars": -5}}) == {}


def test_tool_result_chars_defaults_to_the_shared_limit() -> None:
    from runtime.harness.limits import DEFAULT_TOOL_RESULT_CHARS

    cfg = _config(env_think=None, profiles={})
    assert cfg.tool_result_chars_for("anything:1b") == DEFAULT_TOOL_RESULT_CHARS


def test_tool_result_chars_for_uses_the_models_profile() -> None:
    cfg = _config(
        env_think=None, profiles={"gemma4:e2b-mlx": ModelProfile(tool_result_chars=8000)}
    )
    assert cfg.tool_result_chars_for("gemma4:e2b-mlx") == 8000
    assert cfg.tool_result_chars_for("qwen3.5:9b") != 8000


def test_the_env_override_beats_a_profile_so_a_sweep_can_pin_one_value() -> None:
    """`MODEL_TOOL_RESULT_CHARS` is the operator's escape hatch for a benchmark sweep,
    the same direction MODEL_SMART_THINK already overrides a profile."""
    base = _config(
        env_think=None, profiles={"qwen3.5:9b": ModelProfile(tool_result_chars=8000)}
    )
    cfg = base.model_copy(update={"models": ModelConfig(tool_result_chars=1024)})
    assert cfg.tool_result_chars_for("qwen3.5:9b") == 1024
    assert cfg.tool_result_chars_for("anything:1b") == 1024


@pytest.mark.parametrize("raw", [None, "", "  ", "abc", "0", "-5", "3.5"])
def test_a_bad_env_value_is_ignored_not_fatal(raw: str | None) -> None:
    from runtime.config import _parse_positive_int

    assert _parse_positive_int(raw) is None


def test_a_good_env_value_parses() -> None:
    from runtime.config import _parse_positive_int

    assert _parse_positive_int(" 8000 ") == 8000


# --- planner_max_tokens: the structured-decision token budget ------------------
# The planner was hard-coded to 512 output tokens, thinking included. Run 2 (full arm,
# easy suite): qwen3-vl:4b had 81 calls cut off by it and all 30 of its failures were
# `thinking_budget_exhausted`; qwen3.5:4b-mlx had 13 of 45 variants hit it and paid in
# retries (57.8% inside the 30 s budget). The dispatcher's free-text synthesis already
# allows 2,048, so only the structured decisions were on the tight budget. Same audit
# as the tool-result clip; see docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md.


def test_profile_carries_planner_max_tokens_independently() -> None:
    profile = _coerce_model_profiles({"qwen3-vl:4b": {"planner_max_tokens": 2048}})["qwen3-vl:4b"]
    assert profile.planner_max_tokens == 2048
    assert profile.think is None
    assert profile.tool_result_chars is None


@pytest.mark.parametrize("bad", [0, -1, 8193])
def test_planner_max_tokens_must_be_a_usable_request_budget(bad: int) -> None:
    """ChatRequest allows 1..8192; anything else would fail every planner call."""
    assert _coerce_model_profiles({"a:1b": {"planner_max_tokens": bad}}) == {}


def test_planner_max_tokens_default_is_the_current_behaviour() -> None:
    """The default stays 512 until the sweep decides, so a run without a profile is
    still the clean baseline the sweep is measured against."""
    from runtime.harness.limits import DEFAULT_PLANNER_MAX_TOKENS

    assert DEFAULT_PLANNER_MAX_TOKENS == 512
    assert _config(env_think=None, profiles={}).planner_max_tokens_for("x:1b") == 512


def test_planner_max_tokens_for_uses_the_models_profile() -> None:
    cfg = _config(
        env_think=None, profiles={"qwen3-vl:4b": ModelProfile(planner_max_tokens=2048)}
    )
    assert cfg.planner_max_tokens_for("qwen3-vl:4b") == 2048
    assert cfg.planner_max_tokens_for("gemma4:e4b-mlx") == 512


def test_the_planner_budget_env_override_beats_a_profile() -> None:
    base = _config(
        env_think=None, profiles={"qwen3-vl:4b": ModelProfile(planner_max_tokens=2048)}
    )
    cfg = base.model_copy(update={"models": ModelConfig(planner_max_tokens=1024)})
    assert cfg.planner_max_tokens_for("qwen3-vl:4b") == 1024
    assert cfg.planner_max_tokens_for("anything:1b") == 1024


def test_construction_sites_resolve_the_planner_budget_per_model() -> None:
    from pathlib import Path

    assert "planner_max_tokens_for(" in Path("runtime/chat/telegram/bot.py").read_text(
        encoding="utf-8"
    )
    assert "planner_max_tokens_for(" in Path("runtime/chat/cli.py").read_text(encoding="utf-8")


def test_construction_sites_resolve_the_result_limit_per_model() -> None:
    """Both production wiring sites must go through `tool_result_chars_for`, and the
    dispatcher must receive it too, so the planner and the synthesiser agree on what
    a result said. A site that omits it silently falls back to the default."""
    from pathlib import Path

    bot = Path("runtime/chat/telegram/bot.py").read_text(encoding="utf-8")
    cli = Path("runtime/chat/cli.py").read_text(encoding="utf-8")

    assert bot.count("tool_result_chars_for(") >= 2  # Tier1Reasoner and HarnessDispatcher
    assert "tool_result_chars_for(" in cli


# --- harness mode + guardrails (spec 2026-10-06-harness-ablation-design.md) ---


def test_unset_harness_profile_is_todays_planner() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(env_think=None, profiles={})
    assert cfg.harness_mode_for("x:1b") == "planner"
    assert cfg.guardrails_for("x:1b") == Guardrails()


def test_native_profile_defaults_to_native_plus_guard() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(env_think=None, profiles={"g:4b": ModelProfile(harness_mode="native")})
    assert cfg.harness_mode_for("g:4b") == "native"
    assert cfg.guardrails_for("g:4b") == Guardrails(plan_lock=False)


def test_an_explicit_guardrail_list_is_honoured() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(
        env_think=None,
        profiles={"q:9b": ModelProfile(harness_mode="native", guardrails=frozenset({"evidence"}))},
    )
    assert cfg.guardrails_for("q:9b") == Guardrails(
        autofill=False, claim_gate=False, evidence=True, synthesis=False, plan_lock=False
    )


def test_the_env_override_beats_a_profile() -> None:
    from runtime.harness.guardrails import Guardrails

    base = _config(env_think=None, profiles={"g:4b": ModelProfile(harness_mode="native")})
    cfg = base.model_copy(
        update={"models": ModelConfig(harness_mode="planner", guardrails=frozenset())}
    )
    assert cfg.harness_mode_for("g:4b") == "planner"
    assert cfg.guardrails_for("g:4b") == Guardrails(
        autofill=False, claim_gate=False, evidence=False, synthesis=False, plan_lock=False
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"harness_mode": "native", "guardrails": ["plan_lock"]},  # plan-lock needs the planner
        {"harness_mode": "turbo"},
        {"guardrails": ["telepathy"]},
    ],
)
def test_invalid_harness_profiles_are_rejected_at_load(raw: dict) -> None:
    assert _coerce_model_profiles({"a:1b": raw}) == {}


def test_env_parsing_for_the_harness_overrides() -> None:
    models = _coerce(
        {"MODEL_HARNESS_MODE": "native", "MODEL_GUARDRAILS": "autofill, evidence"}, {}
    )["models"]
    assert models.harness_mode == "native"
    assert models.guardrails == frozenset({"autofill", "evidence"})
    none = _coerce({"MODEL_GUARDRAILS": "none"}, {})["models"]
    assert none.guardrails == frozenset()
    bad = _coerce(
        {"MODEL_HARNESS_MODE": "turbo", "MODEL_GUARDRAILS": "autofill,bogus"}, {}
    )["models"]
    assert bad.harness_mode is None
    assert bad.guardrails is None


def test_plan_lock_is_dropped_in_native_mode_even_when_an_override_asks_for_it() -> None:
    from runtime.harness.guardrails import Guardrails

    base = _config(env_think=None, profiles={})
    cfg = base.model_copy(
        update={
            "models": ModelConfig(
                harness_mode="native", guardrails=frozenset({"plan_lock", "evidence"})
            )
        }
    )
    assert cfg.guardrails_for("x:1b") == Guardrails(
        autofill=False, claim_gate=False, evidence=True, synthesis=False, plan_lock=False
    )


def test_plan_lock_is_kept_in_planner_mode_with_an_explicit_set() -> None:
    from runtime.harness.guardrails import Guardrails

    cfg = _config(
        env_think=None,
        profiles={
            "p:1b": ModelProfile(harness_mode="planner", guardrails=frozenset({"plan_lock"}))
        },
    )
    assert cfg.guardrails_for("p:1b") == Guardrails(
        autofill=False, claim_gate=False, evidence=False, synthesis=False, plan_lock=True
    )


def test_a_valid_native_profile_loads() -> None:
    profiles = _coerce_model_profiles(
        {"g:4b": {"harness_mode": "native", "guardrails": ["evidence"]}}
    )
    assert profiles["g:4b"].harness_mode == "native"
    assert profiles["g:4b"].guardrails == frozenset({"evidence"})


# --- native prompt per model (plan doc section 23: identity prompt regressed llama3.2:3b) ---


def test_native_prompt_defaults_to_identity() -> None:
    cfg = _config(env_think=None, profiles={})
    assert cfg.native_prompt_for("x:1b") == "identity"


def test_a_profile_can_pick_the_bare_native_prompt() -> None:
    cfg = _config(
        env_think=None,
        profiles={"l:3b": ModelProfile(harness_mode="native", native_prompt="bare")},
    )
    assert cfg.native_prompt_for("l:3b") == "bare"
    assert cfg.native_prompt_for("g:4b") == "identity"  # other models are unaffected


def test_the_global_native_prompt_beats_a_profile() -> None:
    """The eval arms pin the prompt globally; a profile must not leak into a rerun."""
    base = _config(
        env_think=None,
        profiles={"l:3b": ModelProfile(harness_mode="native", native_prompt="bare")},
    )
    cfg = base.model_copy(
        update={"models": base.models.model_copy(update={"native_prompt": "identity"})}
    )
    assert cfg.native_prompt_for("l:3b") == "identity"


def test_native_prompt_is_parsed_from_a_profile_dict() -> None:
    profiles = _coerce_model_profiles({"l:3b": {"harness_mode": "native", "native_prompt": "bare"}})
    assert profiles["l:3b"].native_prompt == "bare"


def test_an_unknown_native_prompt_drops_the_profile_not_the_boot() -> None:
    assert _coerce_model_profiles({"l:3b": {"native_prompt": "fancy"}}) == {}
