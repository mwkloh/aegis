"""Phase 8 Track A3 — `ModelRouter` local tiers + fallback logic.

Pins:

* FAST / FAST_LOCAL / REFLECTION always resolve to local (ollama).
* SMART_LOCAL forces local even when OpenRouter is configured.
* SMART with `prefer_local=True` and a reachable probe → local, not degraded.
* SMART with `prefer_local=False` and OpenRouter key → OpenRouter.
* SMART with neither OpenRouter nor reachable local → local, `degraded=True`
  (silent degrade — never raises).
* SMART with `prefer_local=True` but local unreachable → OpenRouter if key set.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
    VaultIndexingConfig,
)
from runtime.model_router.router import ModelRouter, ModelTarget, ModelTier

pytestmark = pytest.mark.unit


def _cfg(
    *,
    prefer_local: bool = False,
    openrouter_key: str | None = None,
    smart: str = "minimax/minimax-m2.7",
    smart_local: str = "llama3.1:8b",
) -> AegisConfig:
    return AegisConfig(
        models=ModelConfig(
            fast="gemma4:e2b",
            smart=smart,
            smart_local=smart_local,
            reflection="gemma4:e4b",
            coding=smart,
            prefer_local=prefer_local,
        ),
        providers=ProviderConfig(
            ollama_base_url="http://127.0.0.1:11434",
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_api_key=openrouter_key,
        ),
        telegram=TelegramConfig(),
        storage=StorageConfig(),
        vault_indexing=VaultIndexingConfig(),
    )


def test_fast_routes_to_local_ollama() -> None:
    r = ModelRouter(_cfg(), local_ready=lambda: True)
    t = r.route(ModelTier.FAST)
    assert isinstance(t, ModelTarget)
    assert t.provider == "ollama"
    assert t.model == "gemma4:e2b"
    assert t.base_url == "http://127.0.0.1:11434"
    assert t.degraded is False


def test_fast_local_always_local() -> None:
    r = ModelRouter(_cfg(), local_ready=lambda: False)
    t = r.route(ModelTier.FAST_LOCAL)
    assert t.provider == "ollama"


def test_reflection_routes_to_local() -> None:
    r = ModelRouter(_cfg(), local_ready=lambda: True)
    t = r.route(ModelTier.REFLECTION)
    assert t.provider == "ollama"
    assert t.model == "gemma4:e4b"


def test_smart_local_forces_local_even_with_openrouter_key() -> None:
    r = ModelRouter(
        _cfg(openrouter_key="sk-real"), local_ready=lambda: True
    )
    t = r.route(ModelTier.SMART_LOCAL)
    assert t.provider == "ollama"
    assert t.model == "llama3.1:8b"


def test_smart_prefers_local_when_flag_on_and_probe_true() -> None:
    r = ModelRouter(
        _cfg(prefer_local=True, openrouter_key="sk-real"),
        local_ready=lambda: True,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"
    assert t.model == "llama3.1:8b"
    assert t.degraded is False


def test_smart_uses_openrouter_when_local_not_preferred() -> None:
    r = ModelRouter(
        _cfg(prefer_local=False, openrouter_key="sk-real"),
        local_ready=lambda: True,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "openrouter"
    assert t.model == "minimax/minimax-m2.7"
    assert t.degraded is False


def test_smart_falls_back_to_openrouter_when_preferred_local_unreachable() -> None:
    r = ModelRouter(
        _cfg(prefer_local=True, openrouter_key="sk-real"),
        local_ready=lambda: False,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "openrouter"
    assert t.degraded is False


def test_smart_silent_degrade_when_no_openrouter_and_no_local() -> None:
    r = ModelRouter(
        _cfg(prefer_local=False, openrouter_key=None),
        local_ready=lambda: False,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"
    assert t.model == "llama3.1:8b"
    assert t.degraded is True


def test_smart_silent_degrade_when_preferred_local_unreachable_no_key() -> None:
    r = ModelRouter(
        _cfg(prefer_local=True, openrouter_key=None),
        local_ready=lambda: False,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"
    assert t.degraded is True


def test_default_local_probe_succeeds_on_200() -> None:
    # No explicit probe → uses `_default_local_probe`.
    cfg = _cfg(prefer_local=True, openrouter_key="sk-real")
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        r = ModelRouter(cfg)
        t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"


def test_default_local_probe_falls_back_on_connect_error() -> None:
    cfg = _cfg(prefer_local=True, openrouter_key="sk-real")
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            side_effect=httpx.ConnectError("refused")
        )
        r = ModelRouter(cfg)
        t = r.route(ModelTier.SMART)
    # Degraded? No — OpenRouter key is present and takes over.
    assert t.provider == "openrouter"
    assert t.degraded is False


def test_default_local_probe_caches() -> None:
    """Second call within 30s window must not re-issue the probe."""
    cfg = _cfg(prefer_local=True, openrouter_key="sk-real")
    with respx.mock() as mock:
        route = mock.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        r = ModelRouter(cfg)
        r.route(ModelTier.SMART)
        r.route(ModelTier.SMART)
    assert route.call_count == 1
