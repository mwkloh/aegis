"""Phase 8 Track A3 — `ModelRouter` local tiers + explicit provider pin.

Pins:

* FAST / FAST_LOCAL / REFLECTION always resolve to local (ollama).
* SMART_LOCAL forces local even when OpenRouter is configured.
* SMART is a direct lookup on `smart_provider` — no liveness probe in the
  routing decision, no fallback between providers.
"""
from __future__ import annotations

import pytest

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
    VaultIndexingConfig,
)
from runtime.llm.router import ModelRouter, ModelTarget, ModelTier

pytestmark = pytest.mark.unit


def _cfg(
    *,
    smart_provider: str = "ollama",
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
            smart_provider=smart_provider,  # type: ignore[arg-type]
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


def test_smart_pinned_to_ollama_ignores_probe() -> None:
    """No liveness probe in the decision — even a down probe doesn't reroute."""
    r = ModelRouter(
        _cfg(smart_provider="ollama", openrouter_key="sk-real"),
        local_ready=lambda: False,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"
    assert t.model == "llama3.1:8b"
    assert t.degraded is False


def test_smart_pinned_to_openrouter_ignores_missing_key() -> None:
    """route() itself never validates key presence — that's the builder's job."""
    r = ModelRouter(
        _cfg(smart_provider="openrouter", openrouter_key=None),
        local_ready=lambda: True,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "openrouter"
    assert t.model == "minimax/minimax-m2.7"
    assert t.degraded is False


def test_smart_pinned_to_ollama_uses_smart_local_model() -> None:
    r = ModelRouter(
        _cfg(smart_provider="ollama", smart_local="deepseek-v4-pro:0813-cloud"),
        local_ready=lambda: True,
    )
    t = r.route(ModelTier.SMART)
    assert t.provider == "ollama"
    assert t.model == "deepseek-v4-pro:0813-cloud"
