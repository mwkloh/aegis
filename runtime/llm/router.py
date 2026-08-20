"""Model router — picks model + provider for a tier.

Phase 8 Track A3 extended the Phase 0 stub with `FAST_LOCAL` / `SMART_LOCAL`
tiers for callers that insist on local.

* `FAST` / `FAST_LOCAL` / `REFLECTION` always resolve to local (ollama).
* `SMART_LOCAL` forces local even when OpenRouter is configured.
* `SMART` is a direct lookup on `smart_provider` — no liveness probe in the
  routing decision, no fallback between providers (2026-08-19 explicit
  model-provider routing design). Callers that need to know whether Ollama
  is actually reachable before constructing a client still use
  `is_local_ready()` separately.

The router is intentionally synchronous. Callers already hold a
`ModelClient` reference for the returned target and make their own
async calls; the router only decides *which* target. Liveness is
probed through an injectable `local_ready` callable (cached for 30s
by default) so tests can pin it without touching the network.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict, Field

from runtime.config import AegisConfig

_LOCAL_PROBE_TIMEOUT = httpx.Timeout(connect=0.5, read=0.5, write=0.5, pool=0.5)
_LOCAL_PROBE_TTL_S = 30.0


class ModelTier(StrEnum):
    FAST = "fast"
    FAST_LOCAL = "fast_local"
    SMART = "smart"
    SMART_LOCAL = "smart_local"
    REFLECTION = "reflection"


class ModelTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: ModelTier
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    base_url: str = Field(min_length=1)


LocalReadyProbe = Callable[[], bool]


def _default_local_probe(base_url: str) -> LocalReadyProbe:
    """Sync ping of `/api/tags` with a short timeout and 30s cache."""
    state: dict[str, float | bool] = {"checked_at": 0.0, "ready": False}

    def _probe() -> bool:
        now = time.monotonic()
        if now - float(state["checked_at"]) < _LOCAL_PROBE_TTL_S:
            return bool(state["ready"])
        ready = False
        try:
            with httpx.Client(
                base_url=base_url, timeout=_LOCAL_PROBE_TIMEOUT, follow_redirects=False
            ) as client:
                resp = client.get("/api/tags")
                ready = resp.status_code == httpx.codes.OK
        except httpx.HTTPError:
            ready = False
        state["checked_at"] = now
        state["ready"] = ready
        return ready

    return _probe


class ModelRouter:
    """Resolves a tier to a concrete (model, provider, base_url)."""

    def __init__(
        self,
        config: AegisConfig,
        *,
        local_ready: LocalReadyProbe | None = None,
    ) -> None:
        self._config = config
        self._local_ready: LocalReadyProbe = local_ready or _default_local_probe(
            config.providers.ollama_base_url
        )

    def is_local_ready(self) -> bool:
        """Probe whether the local Ollama daemon is reachable.

        Callers outside `route()` (e.g. `build_chat_pipeline` picking
        an embedder) need the same decision the router uses. Exposing
        the probe here keeps the 30 s cache shared across consumers so
        we don't hammer the loopback port multiple times per startup.
        """
        return self._local_ready()

    def route(self, tier: ModelTier) -> ModelTarget:
        providers = self._config.providers
        models = self._config.models

        if tier in (ModelTier.FAST, ModelTier.FAST_LOCAL):
            return ModelTarget(
                tier=tier,
                model=models.fast,
                provider="ollama",
                base_url=providers.ollama_base_url,
            )
        if tier is ModelTier.REFLECTION:
            return ModelTarget(
                tier=tier,
                model=models.reflection,
                provider="ollama",
                base_url=providers.ollama_base_url,
            )
        if tier is ModelTier.SMART_LOCAL:
            return ModelTarget(
                tier=tier,
                model=models.smart_local,
                provider="ollama",
                base_url=providers.ollama_base_url,
            )

        if tier is ModelTier.SMART:
            if models.smart_provider == "ollama":
                return ModelTarget(
                    tier=tier,
                    model=models.smart_local,
                    provider="ollama",
                    base_url=providers.ollama_base_url,
                )
            return ModelTarget(
                tier=tier,
                model=models.smart,
                provider="openrouter",
                base_url=providers.openrouter_base_url,
            )

        raise ValueError(f"unhandled tier: {tier!r}")  # pragma: no cover — exhaustive above
