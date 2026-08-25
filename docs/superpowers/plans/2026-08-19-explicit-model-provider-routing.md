# Explicit Model-Provider Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implicit `prefer_local` + liveness-probe fallback for the SMART model tier with an explicit `smart_provider` config pin, applied uniformly to `ChatPipeline` and `HarnessDispatcher` (which today hardcodes OpenRouter), plus a live per-reply `[provider · model]` footer in Telegram so the operator always knows what actually answered.

**Architecture:** `ModelConfig.prefer_local: bool` becomes `ModelConfig.smart_provider: Literal["ollama", "openrouter"]`. `ModelRouter.route(SMART)` becomes a pure lookup (no network probe in the decision). `build_chat_pipeline` and `build_harness_dispatcher` (`runtime/chat/telegram/bot.py`) both consult the same route and construct only the named client, failing closed if it's unreachable. `ChatPipeline` and `HarnessDispatcher` each gain a `provider` field so `route_chat`/the dispatcher can decorate outgoing Telegram text with `_[provider · model]_` — added only to what's *sent*, never to what's stored in `Tier3Store` (which feeds back into future LLM context).

**Tech Stack:** Python 3.11, pytest (`pytest.mark.unit`), pydantic v2, respx/httpx for mocked HTTP, ruff/mypy/bandit via `make lint type security`.

**Spec:** `docs/superpowers/specs/2026-08-19-explicit-model-provider-routing-design.md`

## Global Constraints

- No new provider client classes — only `OllamaClient` and `OpenRouterClient` (spec Non-Goals).
- `FAST` / `FAST_LOCAL` / `REFLECTION` tiers in `ModelRouter.route()` are untouched.
- The three-layer reply fallback order (`HarnessDispatcher` → `ChatPipeline` → static stub) is untouched — only how each layer picks its own provider internally changes.
- No runtime/Telegram-command-driven provider switching — config-file only.
- The footer must never be appended to text stored in `Tier3Store` or used as conversation history — only to what is transmitted to Telegram.
- Every task ends green on `pytest -m unit` for the files it touches before moving to the next task.

---

### Task 1: Config schema — `smart_provider` replaces `prefer_local`

**Files:**
- Modify: `runtime/config.py:14` (imports), `runtime/config.py:34-53` (`ModelConfig`), `runtime/config.py:255-271` (helper functions), `runtime/config.py:274-288` (`_coerce`)
- Test: `tests/test_config_loads.py`

**Interfaces:**
- Produces: `ModelConfig.smart_provider: Literal["ollama", "openrouter"]` (default `"ollama"`), consumed by Task 2 (`ModelRouter.route`).
- Removes: `ModelConfig.prefer_local: bool` (no longer exists — any remaining reference is a bug to fix in a later task).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config_loads.py` (after `test_config_picks_up_env_overrides`, following the existing `aegis_sandbox`/`reset_config`/`get_config` pattern in that file):

```python
def test_config_smart_provider_defaults_to_ollama(aegis_sandbox: Path) -> None:
    cfg = get_config()
    assert cfg.models.smart_provider == "ollama"


def test_config_smart_provider_reads_openrouter_from_env(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_SMART_PROVIDER", "openrouter")
    reset_config()
    cfg = get_config()
    assert cfg.models.smart_provider == "openrouter"


def test_config_smart_provider_invalid_value_falls_back_to_ollama(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_SMART_PROVIDER", "not-a-real-provider")
    reset_config()
    cfg = get_config()
    assert cfg.models.smart_provider == "ollama"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_loads.py -k smart_provider -v`
Expected: FAIL — `AttributeError: 'ModelConfig' object has no attribute 'smart_provider'` (the field doesn't exist yet).

- [ ] **Step 3: Implement**

In `runtime/config.py:14`, change:
```python
from typing import Any
```
to:
```python
from typing import Any, Literal
```

In `runtime/config.py`, replace the `ModelConfig` class (lines 34-53):
```python
class ModelConfig(BaseModel):
    """Pinned model identifiers and routing preferences."""

    fast: str = Field(default="gemma4:e2b", description="Tier 0 intent classifier.")
    smart: str = Field(
        default="minimax/minimax-m2.7", description="Tier 1 reasoning (OpenRouter)."
    )
    smart_local: str = Field(
        default="llama3.1:8b",
        description="Tier 1 local fallback when prefer_local=True or OpenRouter unavailable.",
    )
    reflection: str = Field(default="gemma4:e4b", description="Reflection plane.")
    coding: str = Field(
        default="minimax/minimax-m2.7",
        description="Plane 3 coding harness (OpenRouter by default).",
    )
    prefer_local: bool = Field(
        default=False,
        description="If True AND Ollama is reachable, route SMART to local.",
    )
```
with:
```python
class ModelConfig(BaseModel):
    """Pinned model identifiers and routing preferences."""

    fast: str = Field(default="gemma4:e2b", description="Tier 0 intent classifier.")
    smart: str = Field(
        default="minimax/minimax-m2.7", description="Tier 1 reasoning (OpenRouter)."
    )
    smart_local: str = Field(
        default="llama3.1:8b",
        description="Tier 1 reasoning via Ollama — local weights or an Ollama Cloud tag.",
    )
    reflection: str = Field(default="gemma4:e4b", description="Reflection plane.")
    coding: str = Field(
        default="minimax/minimax-m2.7",
        description="Plane 3 coding harness (OpenRouter by default).",
    )
    smart_provider: Literal["ollama", "openrouter"] = Field(
        default="ollama",
        description="Explicit SMART-tier provider pin — no silent fallback between them.",
    )
```

In `runtime/config.py`, after `_env_bool` (which ends at line 259, right before `_parse_env_allowlist` at line 262), add a new helper:

```python
def _parse_smart_provider(raw: str | None) -> Literal["ollama", "openrouter"]:
    """Explicit SMART-tier provider pin. Unset/invalid -> 'ollama' (local-first default)."""
    if raw is not None and raw.strip().lower() == "openrouter":
        return "openrouter"
    return "ollama"
```

In `runtime/config.py`, inside `_coerce()`, replace:
```python
        prefer_local=_env_bool(env.get("MODELS_PREFER_LOCAL"), default=False),
```
with:
```python
        smart_provider=_parse_smart_provider(env.get("MODEL_SMART_PROVIDER")),
```
(this is the last kwarg of the `models = ModelConfig(...)` call inside `_coerce()`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_loads.py -v`
Expected: PASS (all tests in the file, not just the new ones — confirms no regression).

- [ ] **Step 5: Commit**

```bash
git add runtime/config.py tests/test_config_loads.py
git commit -m "feat(config): replace prefer_local with explicit smart_provider pin"
```

---

### Task 2: `ModelRouter.route()` — explicit lookup, no probe in the decision

**Files:**
- Modify: `runtime/llm/router.py:108-162` (`ModelRouter.route`)
- Test: `tests/test_model_router.py` (rewrite SMART-tier tests)

**Interfaces:**
- Consumes: `ModelConfig.smart_provider` (Task 1).
- Produces: `ModelRouter.route(ModelTier.SMART) -> ModelTarget` with `provider` set directly from `smart_provider`, `degraded` always `False` for SMART. `is_local_ready()` signature/behavior unchanged (still used by Task 4/5's builders).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_model_router.py` in full (the whole file — every SMART-tier test changes shape since the fallback semantics are gone):

```python
"""Phase 8 Track A3 — `ModelRouter` local tiers + explicit provider pin.

Pins:

* FAST / FAST_LOCAL / REFLECTION always resolve to local (ollama).
* SMART_LOCAL forces local even when OpenRouter is configured.
* SMART is a direct lookup on `smart_provider` — no liveness probe in the
  routing decision, no fallback between providers.
"""
from __future__ import annotations

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
    VaultIndexingConfig,
)
from runtime.llm.router import ModelRouter, ModelTarget, ModelTier

import pytest

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
```

Note: `respx`/`httpx` imports and the three `test_default_local_probe_*` tests are deleted entirely — `route(SMART)` no longer calls the probe, so there is nothing to mock there. `is_local_ready()`'s own probe behavior is exercised indirectly by Task 4/5's builder tests (Task 4, 5) via the `local_ready` injection point, which is unchanged.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_model_router.py -v`
Expected: FAIL on the new `test_smart_pinned_to_*` tests — `ModelConfig` accepts `smart_provider` now (Task 1 done), but `route()` still branches on the (deleted) `prefer_local` field, so this will raise `AttributeError: 'ModelConfig' object has no attribute 'prefer_local'` inside `route()`.

- [ ] **Step 3: Implement**

In `runtime/llm/router.py`, replace the `route()` method body's SMART section (currently lines 134-162, everything from the `# SMART — layered preference:` comment to the end of the method) with:

```python
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
```

Also update the module docstring (`runtime/llm/router.py:1-16`) to drop the now-false "Pins" bullets about fallback and replace with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_model_router.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/llm/router.py tests/test_model_router.py
git commit -m "feat(router): SMART tier is a pure smart_provider lookup, no fallback"
```

---

### Task 3: `ChatPipeline` gains `provider`, exposed for the footer (storage stays clean)

**Files:**
- Modify: `runtime/chat/pipeline.py:146-171` (`__init__`), add a `provider` property
- Test: `tests/test_chat_pipeline.py:89-113` (`_make_pipeline`), `tests/test_chat_pipeline.py:364-377` (`test_model_name_required`)

**Interfaces:**
- Produces: `ChatPipeline.provider: str` (read-only property), `ChatPipeline.model_name: str` (new read-only property — `_model_name` already existed as a private attribute; this task also exposes it publicly since Task 4's `route_chat` change needs to read it from outside the class).
- Consumed by: Task 4 (`route_chat`'s footer logic).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_chat_pipeline.py`, after `test_model_name_required`:

```python
def test_provider_and_model_name_are_exposed(tmp_path: Path) -> None:
    pipe, _, _, _ = _make_pipeline(tmp_path)
    assert pipe.model_name == "fake-model"
    assert pipe.provider == "ollama"


def test_provider_required(tmp_path: Path) -> None:
    _write(tmp_path / "IDENTITY.md", "i")
    tier1 = Tier1Loader(tmp_path)
    tier3 = Tier3Store()
    builder = ContextBuilder(tier1, tier3)
    with pytest.raises(ValueError, match="provider"):
        ChatPipeline(
            tier1=tier1,
            tier3=tier3,
            recall=_FakeRecall(),
            builder=builder,
            model=_FakeModel(),
            model_name="fake-model",
            provider="",
        )
```

Update `_make_pipeline` (`tests/test_chat_pipeline.py:89-113`) to pass `provider="ollama"` — change the `ChatPipeline(...)` call inside it from:
```python
    pipe = ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=r,
        builder=builder,
        model=m,
        model_name="fake-model",
        events=events,
    )
```
to:
```python
    pipe = ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=r,
        builder=builder,
        model=m,
        model_name="fake-model",
        provider="ollama",
        events=events,
    )
```

Update `test_model_name_required` (`tests/test_chat_pipeline.py:364-377`) — its direct `ChatPipeline(...)` call needs `provider="ollama"` added too (it's testing the `model_name` validation specifically, so give it a valid `provider` so only the intended check fires):
```python
def test_model_name_required(tmp_path: Path) -> None:
    _write(tmp_path / "IDENTITY.md", "i")
    tier1 = Tier1Loader(tmp_path)
    tier3 = Tier3Store()
    builder = ContextBuilder(tier1, tier3)
    with pytest.raises(ValueError, match="model_name"):
        ChatPipeline(
            tier1=tier1,
            tier3=tier3,
            recall=_FakeRecall(),
            builder=builder,
            model=_FakeModel(),
            model_name="",
            provider="ollama",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_chat_pipeline.py -v`
Expected: FAIL — `TypeError: ChatPipeline.__init__() got an unexpected keyword argument 'provider'` on every test that goes through `_make_pipeline` (i.e. most of the file).

- [ ] **Step 3: Implement**

In `runtime/chat/pipeline.py`, replace `ChatPipeline.__init__` (lines 146-171):

```python
    def __init__(
        self,
        *,
        tier1: Tier1Loader,
        tier3: Tier3Store,
        recall: RecallPort,
        builder: ContextBuilder,
        model: ModelClient,
        model_name: str,
        provider: str,
        events: EventStream | None = None,
        system_prefix: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if not provider:
            raise ValueError("provider must be non-empty")
        self._tier1 = tier1
        self._tier3 = tier3
        self._recall = recall
        self._builder = builder
        self._model = model
        self._model_name = model_name
        self._provider = provider
        self._events = events
        self._system_prefix = system_prefix
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_chat_pipeline.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/chat/pipeline.py tests/test_chat_pipeline.py
git commit -m "feat(chat-pipeline): expose provider + model_name for reply attribution"
```

---

### Task 4: `build_chat_pipeline` passes `provider`; `route_chat` adds the live footer

**Files:**
- Modify: `runtime/chat/telegram/bot.py:52` (import), `runtime/chat/telegram/bot.py:802-810` (`ChatPipeline(...)` construction), `runtime/chat/telegram/bot.py:481-499` (`route_chat`'s pipeline-reply branch)
- Test: `tests/test_telegram_bot.py:1162-1180` (`_cfg_with_allowlist`), `tests/test_telegram_bot.py` (`_build_cfg`, `_FakePipeline`, and the 3 exact-match `route_chat` assertions found below)

**Interfaces:**
- Consumes: `ChatPipeline.__init__(..., provider=...)` (Task 3).
- Produces: every `route_chat` reply routed through `ChatPipeline` now carries a `\n\n_[provider · model]_` footer when the reply is a genuine model response (not the internal `STUB_REPLY` and not empty).

- [ ] **Step 1: Write the failing tests**

In `tests/test_telegram_bot.py`, update `_build_cfg` (around line 915-935) — add `smart_provider="openrouter"` so its existing 8 call sites keep testing "OpenRouter is configured and should be used" (unchanged from today's `prefer_local=False` default behavior):
```python
def _build_cfg(
    tmp_path: Path,
    *,
    api_key: str | None = "sk-test",
    base_url: str = "https://openrouter.ai/api/v1",
) -> AegisConfig:
    """Minimal AegisConfig for factory tests — never touches real disk state."""
    return AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(smart="minimax/minimax-m2.7", smart_provider="openrouter"),
        providers=ProviderConfig(
            openrouter_base_url=base_url,
            openrouter_api_key=api_key,
        ),
        telegram=TelegramConfig(),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
    )
```

Update `test_build_chat_pipeline_local_path_selects_ollama_and_bgem3` (around line 1010-1029) — replace `prefer_local=True` with `smart_provider="ollama"`:
```python
    cfg = AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(
            smart="minimax/minimax-m2.7",
            smart_local="qwen3:8b",
            smart_provider="ollama",
        ),
        providers=ProviderConfig(openrouter_api_key="sk-test"),
        telegram=TelegramConfig(),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
    )
```
(the rest of that test is unchanged — the comment on the line above `prefer_local` was already "Model name follows the local target when prefer_local wins", update its wording to "Model name follows the local target since smart_provider='ollama'".)

Update `_cfg_with_allowlist` (around line 1162-1180) — add `smart_provider="openrouter"` (this helper's `api_key`/`smart` params are already OpenRouter-flavored; every test using it expects OpenRouter routing):
```python
def _cfg_with_allowlist(
    tmp_path: Path,
    allowlist: list[int],
    *,
    api_key: str | None = "sk-test",
    smart: str = "x-ai/grok-4",
) -> AegisConfig:
    return AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(smart=smart, smart_provider="openrouter"),
        providers=ProviderConfig(openrouter_api_key=api_key),
        telegram=TelegramConfig(user_allowlist=allowlist),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
    )
```

Update `_FakePipeline` (around line 279-291) to expose `provider`/`model_name`, matching the new `ChatPipeline` public surface:
```python
@dataclass
class _FakePipeline:
    """Stand-in for `ChatPipeline` — records calls, returns canned replies."""

    canned_reply: str = "acknowledged"
    calls: list[tuple[str, str]] = field(default_factory=list)
    raises: Exception | None = None
    provider: str = "ollama"
    model_name: str = "stub-model"

    async def turn(self, chat_id: str, user_text: str) -> str:
        self.calls.append((chat_id, user_text))
        if self.raises is not None:
            raise self.raises
        return self.canned_reply
```

Fix the 3 exact-match assertions that now need the footer suffix (`\n\n_[ollama · stub-model]_`, matching `_FakePipeline`'s new defaults):

Line ~321, in `test_route_chat_routes_to_pipeline`:
```python
    assert update.effective_message.replies == ["hello back\n\n_[ollama · stub-model]_"]
```

Line ~367, in `test_route_chat_routes_without_authorizer`:
```python
    assert update.effective_message.replies == ["ok\n\n_[ollama · stub-model]_"]
```

Line ~530, in `test_route_chat_intent_miss_falls_through_to_pipeline`:
```python
    assert update.effective_message.replies == ["hi back\n\n_[ollama · stub-model]_"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telegram_bot.py -v`
Expected: FAIL on the 3 updated assertions above (actual reply has no footer yet — `route_chat` hasn't been changed), plus possibly on `test_build_chat_pipeline_happy_path` and similar `build_chat_pipeline` tests if `ChatPipeline(...)`'s new required `provider` kwarg isn't yet passed from `bot.py` (see Step 3) — `build_chat_pipeline` would raise `TypeError` calling `ChatPipeline(...)` without `provider`.

- [ ] **Step 3: Implement**

In `runtime/chat/telegram/bot.py:52`, change:
```python
from runtime.chat.pipeline import ChatPipeline
```
to:
```python
from runtime.chat.pipeline import STUB_REPLY, ChatPipeline
```

In `runtime/chat/telegram/bot.py`, inside `build_chat_pipeline`, the `ChatPipeline(...)` construction (currently lines 802-810):
```python
    return ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=recall,
        builder=builder,
        model=model,
        model_name=target.model,
        events=events,
    )
```
becomes:
```python
    return ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=recall,
        builder=builder,
        model=model,
        model_name=target.model,
        provider=target.provider,
        events=events,
    )
```

In `runtime/chat/telegram/bot.py`, inside `route_chat`, replace (lines 481-493):
```python
    try:
        try:
            reply = await pipeline.turn(str(chat_id), text)
        except Exception:
            logger.exception(
                "telegram.chat.pipeline_crashed", extra={"chat_id": chat_id}
            )
            reply = _CHAT_STUB_REPLY
    finally:
        await _stop_typing_indicator(typing_task)

    if not reply:
        return
```
with:
```python
    try:
        try:
            reply = await pipeline.turn(str(chat_id), text)
            if reply and reply != STUB_REPLY:
                reply = f"{reply}\n\n_[{pipeline.provider} · {pipeline.model_name}]_"
        except Exception:
            logger.exception(
                "telegram.chat.pipeline_crashed", extra={"chat_id": chat_id}
            )
            reply = _CHAT_STUB_REPLY
    finally:
        await _stop_typing_indicator(typing_task)

    if not reply:
        return
```

(The footer is appended to the local `reply` variable only *after* `pipeline.turn()` has already appended the clean, footer-less text to `Tier3Store` internally — `ChatPipeline.turn()`'s own tier-3 write happens before it returns, so this is safe by construction. `reply != STUB_REPLY` excludes `pipeline.py`'s own internal stub-on-LLM-failure text, which is a normal (non-raising) return value, not an exception — attaching a model footer to it would falsely claim that model answered.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telegram_bot.py tests/test_chat_pipeline.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/chat/telegram/bot.py tests/test_telegram_bot.py
git commit -m "feat(telegram): ChatPipeline replies carry a live provider/model footer"
```

---

### Task 5: `HarnessDispatcher` gains `provider`; synthesized replies get the footer

**Files:**
- Modify: `runtime/chat/telegram/harness_dispatcher.py:210-249` (`__init__`), `:581-587` (`dispatch`'s shared send block), `:632-638` (`_execute_confirmed`'s send block)
- Test: `tests/test_harness_dispatcher.py:1588`, `:1827` (the two exact-match assertions)

**Interfaces:**
- Produces: `HarnessDispatcher.__init__(..., provider: str = "ollama", ...)` — defaulted so existing test helpers (`_make_dispatcher`, `_make_multi_step_dispatcher`, `_make_loop_dispatcher`, and inline constructions) need no changes except the two tests asserting exact sent text.
- The footer is appended only at the point of `_send(...)`, never to the `reply_text` value passed to `self._tier3.append(...)` — same clean-storage principle as Task 4.

- [ ] **Step 1: Write the failing tests**

In `tests/test_harness_dispatcher.py`, update the two exact-match assertions (both tests exercise `_make_loop_dispatcher`, whose `synthesis_model="stub-model"` and — after Step 3 — implicit `provider="ollama"` default):

Line ~1588, in `test_multi_step_task_complete_gates_and_skips_chain_synthesis`:
```python
    assert message.replies == ["Found the markdown files.\n\n_[ollama · stub-model]_"]
```

Line ~1827, in the ledger-read-failure test:
```python
    assert message.replies[0] == "Searched for the files.\n\n_[ollama · stub-model]_"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -k "task_complete_gates_and_skips_chain_synthesis or task_complete_identical_args_retry_still_warns_oq7" -v`
Expected: FAIL — actual `message.replies` has no footer yet.

- [ ] **Step 3: Implement**

In `runtime/chat/telegram/harness_dispatcher.py`, in `HarnessDispatcher.__init__` (lines 211-226), add a `provider` parameter and store it:
```python
    def __init__(
        self,
        *,
        classifier: Any,
        registry: SkillRegistry,
        runner: SkillRunner,
        harness: HarnessAdapter,
        synthesizer: ModelClient,
        tier3: Tier3Store,
        tier1_loader: Tier1Loader,
        synthesis_model: str,
        provider: str = "ollama",
        multi_step: bool = False,
        max_steps: int = 5,
        events: EventStream | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._classifier = classifier
        self._registry = registry
        self._runner = runner
        self._harness = harness
        self._synthesizer = synthesizer
        self._tier3 = tier3
        self._tier1_loader = tier1_loader
        self._synthesis_model = synthesis_model
        self._provider = provider
```
(the remaining body of `__init__` — `self._multi_step = multi_step` onward — is unchanged; only the two new lines above are inserted.)

Add a small private helper right after `_now()` (after line 252):
```python
    def _footer(self) -> str:
        return f"\n\n_[{self._provider} · {self._synthesis_model}]_"
```

In `dispatch()`, replace the shared send block (currently lines 581-585):
```python
        logger.info("harness_dispatcher.send_start")
        await _send(reply_text)
        logger.info("harness_dispatcher.send_done")
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
```
with:
```python
        logger.info("harness_dispatcher.send_start")
        await _send(reply_text + self._footer())
        logger.info("harness_dispatcher.send_done")
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
```

In `_execute_confirmed()`, replace the send block (currently lines 635-637):
```python
        await _send(reply_text)
        self._tier3.append(str(chat_id), "user", pending.user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
```
with:
```python
        await _send(reply_text + self._footer())
        self._tier3.append(str(chat_id), "user", pending.user_text)
        self._tier3.append(str(chat_id), "bot", reply_text)
```

Note: the `_send` calls in the CLARIFY intercept (`~467-472`) and the destructive-confirmation-guard path (`~498-511`) are deliberately left untouched — those are deterministic, non-model-generated text (a clarifying question template, a templated confirmation prompt), and attaching a model-attribution footer to them would misrepresent where they came from.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_harness_dispatcher.py -v`
Expected: PASS (all tests — the footer default `"ollama"` matches what the two updated assertions now expect, and every other test's `message.replies` assertion uses substring/`in` checks or empty-list checks, unaffected by the appended footer).

- [ ] **Step 5: Commit**

```bash
git add runtime/chat/telegram/harness_dispatcher.py tests/test_harness_dispatcher.py
git commit -m "feat(harness): synthesized replies carry a live provider/model footer"
```

---

### Task 6: `build_harness_dispatcher` branches on `smart_provider` instead of hardcoding OpenRouter

**Files:**
- Modify: `runtime/chat/telegram/bot.py:839-917` (`build_harness_dispatcher`)
- Test: `tests/test_telegram_bot.py` (new tests, imports)

**Interfaces:**
- Consumes: `ModelRouter.route(ModelTier.SMART)` (Task 2), `HarnessDispatcher(..., provider=...)` (Task 5).
- Produces: `build_harness_dispatcher` now returns a dispatcher whose `Tier1Reasoner`/synthesizer client matches `cfg.models.smart_provider` — `ollama` reuses the classifier's `OllamaClient` instance; `openrouter` builds a separate `OpenRouterClient` and additionally requires `OPENROUTER_API_KEY` (only in that branch — the `ollama` branch has no OpenRouter dependency at all, a deliberate fix: previously *any* missing `OPENROUTER_API_KEY` disabled the whole harness dispatcher, even when only local routing was wanted).

- [ ] **Step 1: Write the failing tests**

In `tests/test_telegram_bot.py`, add these imports. First, add `build_harness_dispatcher` to the existing `from runtime.chat.telegram.bot import (...)` block (`:41-56`) — insert alphabetically after `build_dispatcher`:
```python
from runtime.chat.telegram.bot import (
    MAX_TELEGRAM_CHARS,
    _build_vault_trio,
    _chunk,
    _send_startup_message,
    _startup_message_body,
    build_application,
    build_chat_pipeline,
    build_dispatcher,
    build_harness_dispatcher,
    build_intent_router,
    build_long_running_runner,
    build_scheduler,
    build_skill_arg_resolver,
    route_chat,
    route_command,
)
```

Then add new top-level imports (after the existing `from runtime.skills.registry import ...` line, `:79`):
```python
import httpx
import respx

from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier3 import Tier3Store
from runtime.files.client import FilesClient
```

Add these three tests (anywhere after the existing `build_chat_pipeline` test block, e.g. right after `test_build_chat_pipeline_bgem3_degrades_to_fake_when_ctor_raises` or at the end of the file):

```python
def _harness_cfg(tmp_path: Path, *, smart_provider: str, api_key: str | None) -> AegisConfig:
    return AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(
            smart="minimax/minimax-m2.7",
            smart_local="llama3.1:8b",
            smart_provider=smart_provider,  # type: ignore[arg-type]
        ),
        providers=ProviderConfig(openrouter_api_key=api_key),
        telegram=TelegramConfig(),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
    )


def _harness_deps(tmp_path: Path) -> dict[str, Any]:
    registry = SkillRegistry(
        [
            SkillDescriptor(
                id="list_files",
                description="List a folder.",
                intents=["list_files"],
                tool="files_list",
            )
        ]
    )
    return {
        "skill_registry": registry,
        "tier3": Tier3Store(),
        "tier1_loader": Tier1Loader(tmp_path),
        "files_client": FilesClient(allowed_roots=[tmp_path]),
    }


def test_build_harness_dispatcher_ollama_pin_uses_ollama_for_reasoning(
    tmp_path: Path,
) -> None:
    cfg = _harness_cfg(tmp_path, smart_provider="ollama", api_key=None)
    with respx.mock() as mock:
        mock.get(f"{cfg.providers.ollama_base_url}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        dispatcher = build_harness_dispatcher(cfg, **_harness_deps(tmp_path))
    assert dispatcher is not None
    assert dispatcher._provider == "ollama"  # type: ignore[attr-defined]
    assert dispatcher._synthesis_model == "llama3.1:8b"  # type: ignore[attr-defined]


def test_build_harness_dispatcher_openrouter_pin_uses_openrouter_for_reasoning(
    tmp_path: Path,
) -> None:
    cfg = _harness_cfg(tmp_path, smart_provider="openrouter", api_key="sk-test")
    with respx.mock() as mock:
        mock.get(f"{cfg.providers.ollama_base_url}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        dispatcher = build_harness_dispatcher(cfg, **_harness_deps(tmp_path))
    assert dispatcher is not None
    assert dispatcher._provider == "openrouter"  # type: ignore[attr-defined]
    assert dispatcher._synthesis_model == "minimax/minimax-m2.7"  # type: ignore[attr-defined]


def test_build_harness_dispatcher_openrouter_pin_without_key_returns_none(
    tmp_path: Path,
) -> None:
    cfg = _harness_cfg(tmp_path, smart_provider="openrouter", api_key=None)
    with respx.mock() as mock:
        mock.get(f"{cfg.providers.ollama_base_url}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        dispatcher = build_harness_dispatcher(cfg, **_harness_deps(tmp_path))
    assert dispatcher is None


def test_build_harness_dispatcher_ollama_pin_needs_no_openrouter_key(
    tmp_path: Path,
) -> None:
    """The point of the fix: ollama-pinned reasoning has zero OpenRouter dependency."""
    cfg = _harness_cfg(tmp_path, smart_provider="ollama", api_key=None)
    with respx.mock() as mock:
        mock.get(f"{cfg.providers.ollama_base_url}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        dispatcher = build_harness_dispatcher(cfg, **_harness_deps(tmp_path))
    assert dispatcher is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_telegram_bot.py -k build_harness_dispatcher -v`
Expected: FAIL — `dispatcher._provider` doesn't exist yet on some paths, and today's code unconditionally requires `OPENROUTER_API_KEY` (so `test_build_harness_dispatcher_ollama_pin_needs_no_openrouter_key` fails with `dispatcher is None` when it should be not-None).

- [ ] **Step 3: Implement**

In `runtime/chat/telegram/bot.py`, inside `build_harness_dispatcher`, replace the client-construction block (currently lines 839-860):
```python
    router = ModelRouter(cfg)
    if not router.is_local_ready():
        logger.warning(
            "harness_dispatcher.disabled", extra={"reason": "no_ollama"}
        )
        return None

    try:
        ollama_client = OllamaClient(cfg)
    except OllamaHostError:
        logger.warning(
            "harness_dispatcher.disabled", extra={"reason": "ollama_host"}
        )
        return None

    try:
        openrouter_client = OpenRouterClient(cfg)
    except OpenRouterConfigError:
        logger.warning(
            "harness_dispatcher.disabled", extra={"reason": "no_openrouter"}
        )
        return None
```
with:
```python
    router = ModelRouter(cfg)
    if not router.is_local_ready():
        logger.warning(
            "harness_dispatcher.disabled", extra={"reason": "no_ollama"}
        )
        return None

    try:
        ollama_client = OllamaClient(cfg)
    except OllamaHostError:
        logger.warning(
            "harness_dispatcher.disabled", extra={"reason": "ollama_host"}
        )
        return None

    target = router.route(ModelTier.SMART)
    reasoning_client: ModelClient
    if target.provider == "ollama":
        reasoning_client = ollama_client
    else:
        try:
            reasoning_client = OpenRouterClient(cfg)
        except OpenRouterConfigError:
            logger.warning(
                "harness_dispatcher.disabled", extra={"reason": "no_openrouter"}
            )
            return None
```

Then further down in the same function, replace (currently lines 896-916):
```python
    known_intents = [intent for d in skill_registry.all() for intent in d.intents]
    classifier = ModelBackedClassifier(
        client=ollama_client,
        model=cfg.models.smart_local,
        known_intents=known_intents,
    )
    tier1_reasoner = Tier1Reasoner(client=openrouter_client, model=cfg.models.smart)
    runner = SkillRunner(tier1=tier1_reasoner)

    return HarnessDispatcher(
        classifier=classifier,
        registry=skill_registry,
        runner=runner,
        harness=harness,
        synthesizer=openrouter_client,
        tier3=tier3,
        tier1_loader=tier1_loader,
        synthesis_model=cfg.models.smart,
        multi_step=cfg.harness.multi_step,
        max_steps=cfg.harness.max_steps,
        events=events,
    )
```
with:
```python
    known_intents = [intent for d in skill_registry.all() for intent in d.intents]
    classifier = ModelBackedClassifier(
        client=ollama_client,
        model=cfg.models.smart_local,
        known_intents=known_intents,
    )
    tier1_reasoner = Tier1Reasoner(client=reasoning_client, model=target.model)
    runner = SkillRunner(tier1=tier1_reasoner)

    return HarnessDispatcher(
        classifier=classifier,
        registry=skill_registry,
        runner=runner,
        harness=harness,
        synthesizer=reasoning_client,
        tier3=tier3,
        tier1_loader=tier1_loader,
        synthesis_model=target.model,
        provider=target.provider,
        multi_step=cfg.harness.multi_step,
        max_steps=cfg.harness.max_steps,
        events=events,
    )
```

`ModelClient` is already imported in `bot.py:73` (`from runtime.llm.clients.base import ModelClient`) — no import change needed for the `reasoning_client: ModelClient` annotation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telegram_bot.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add runtime/chat/telegram/bot.py tests/test_telegram_bot.py
git commit -m "feat(harness): build_harness_dispatcher honors smart_provider pin"
```

---

### Task 7: `/status` drops the redundant `prefer_local` field; startup banner shows the live pin

**Files:**
- Modify: `runtime/chat/telegram/status.py:57-113` (`SystemInfo`, `collect_system_info`), `runtime/chat/telegram/formatters.py:156-183` (`render_system_info`), `runtime/chat/telegram/bot.py:920-936` (`_startup_message_body`)
- Test: `tests/test_telegram_bot.py:1183-1189` (already covered by Task 4's `_cfg_with_allowlist` fix — verify, no further change expected)

**Interfaces:**
- `SystemInfo` loses the `prefer_local: bool` field (redundant with the already-present, already-live `smart_provider: str` field).
- `_startup_message_body` now computes `ModelRouter(cfg).route(ModelTier.SMART)` itself instead of reading `cfg.models.smart` directly — a pure computation post-Task-2 (no network probe), so this stays a "pure function" per its own docstring.

- [ ] **Step 1: Write the failing test**

`/status`'s `SystemInfo`/`collect_system_info` has no direct unit test today (confirmed: no `SystemInfo(` construction anywhere in `tests/`), so there is no test to make fail here — this step is a direct implementation + regression-check via the existing `test_telegram_status.py` and `test_telegram_bot.py` suites (Step 4 covers verification). Skip to Step 3.

- [ ] **Step 2: (skipped — no new failing test for this step; see Step 1)**

- [ ] **Step 3: Implement**

In `runtime/chat/telegram/status.py`, remove `prefer_local: bool` from `SystemInfo` (currently line 75):
```python
    smart_model: str
    smart_provider: str
    smart_degraded: bool
    smart_local_model: str
    fast_model: str
    reflection_model: str
    coding_model: str
    prefer_local: bool
    local_ready: bool
```
becomes:
```python
    smart_model: str
    smart_provider: str
    smart_degraded: bool
    smart_local_model: str
    fast_model: str
    reflection_model: str
    coding_model: str
    local_ready: bool
```

In the same file, remove the corresponding line from `collect_system_info` (currently line 105):
```python
        coding_model=cfg.models.coding,
        prefer_local=cfg.models.prefer_local,
        local_ready=router.is_local_ready(),
```
becomes:
```python
        coding_model=cfg.models.coding,
        local_ready=router.is_local_ready(),
```

In `runtime/chat/telegram/formatters.py`, in `render_system_info` (currently lines 170-183), remove the `prefer_local` line:
```python
    return (
        "Models:\n"
        f"• smart:      {smart_line}\n"
        f"• smart_local:{info.smart_local_model}\n"
        f"• fast:       {info.fast_model}\n"
        f"• reflection: {info.reflection_model}\n"
        f"• coding:     {info.coding_model}\n"
        f"• prefer_local: {str(info.prefer_local).lower()}\n"
        "Runtime:\n"
        f"• ollama:     {info.ollama_base_url} ({local_state})\n"
        f"• openrouter: {openrouter_state}\n"
        f"• vault:      {vault_line}\n"
        f"• allowlist:  {info.allowlist_size} chat(s)"
    )
```
becomes:
```python
    return (
        "Models:\n"
        f"• smart:      {smart_line}\n"
        f"• smart_local:{info.smart_local_model}\n"
        f"• fast:       {info.fast_model}\n"
        f"• reflection: {info.reflection_model}\n"
        f"• coding:     {info.coding_model}\n"
        "Runtime:\n"
        f"• ollama:     {info.ollama_base_url} ({local_state})\n"
        f"• openrouter: {openrouter_state}\n"
        f"• vault:      {vault_line}\n"
        f"• allowlist:  {info.allowlist_size} chat(s)"
    )
```

In `runtime/chat/telegram/bot.py`, replace `_startup_message_body` (currently lines 920-936):
```python
def _startup_message_body(cfg: AegisConfig, *, now: datetime | None = None) -> str:
    """Compose the "AEGIS online" notification body. Pure function — tested.

    Includes a UTC timestamp, the configured SMART-tier model, and
    whether the conversational pipeline is wired (OpenRouter key
    present) or running in stub mode. Operator uses /status for the
    full picture; this is just a heartbeat-on-boot signal.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    chat_status = (
        "wired" if cfg.providers.openrouter_api_key else "stub (no OPENROUTER_API_KEY)"
    )
    return (
        f"🟢 AEGIS online — {stamp}\n"
        f"model: {cfg.models.smart}\n"
        f"chat: {chat_status}"
    )
```
with:
```python
def _startup_message_body(cfg: AegisConfig, *, now: datetime | None = None) -> str:
    """Compose the "AEGIS online" notification body. Pure function — tested.

    Includes a UTC timestamp, the live-routed SMART-tier provider:model
    (via `ModelRouter.route` — a pure config lookup post smart_provider
    pin, no network probe), and whether the conversational pipeline is
    wired (OpenRouter key present) or running in stub mode. Operator uses
    /status for the full picture; this is just a heartbeat-on-boot signal.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    chat_status = (
        "wired" if cfg.providers.openrouter_api_key else "stub (no OPENROUTER_API_KEY)"
    )
    target = ModelRouter(cfg).route(ModelTier.SMART)
    return (
        f"🟢 AEGIS online — {stamp}\n"
        f"model: {target.provider}:{target.model}\n"
        f"chat: {chat_status}"
    )
```

(`ModelRouter`/`ModelTier` are already imported at the top of `bot.py` — confirmed by their existing use in `build_chat_pipeline`/`build_harness_dispatcher`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_telegram_bot.py tests/test_telegram_status.py -v`
Expected: PASS. `test_startup_message_body_includes_model_and_timestamp` still passes unmodified — it asserts `"x-ai/grok-4" in body`, and with Task 4's `_cfg_with_allowlist` fix (`smart_provider="openrouter"`) the rendered line is `model: openrouter:x-ai/grok-4`, which still contains that substring.

- [ ] **Step 5: Commit**

```bash
git add runtime/chat/telegram/status.py runtime/chat/telegram/formatters.py runtime/chat/telegram/bot.py
git commit -m "fix(telegram): drop redundant prefer_local from /status, live-route startup banner"
```

---

### Task 8: Migrate `.env`/`.env.example` (repo) and the live `~/.aegis/.env`

**Files:**
- Modify: `.env` (repo template), `.env.example` (repo template), `/home/pookster/.aegis/.env` (live, outside the repo — not committed)

**Interfaces:** None — this is config-file text only, no code.

- [ ] **Step 1: (no test — this is documentation/config-value migration, verified by inspection in Step 4)**

- [ ] **Step 2: (skipped — see Step 1)**

- [ ] **Step 3: Implement**

In both `/home/pookster/projects/aegis/.env` and `/home/pookster/projects/aegis/.env.example`, replace:
```
# --- Models (Tier 0 / Tier 1 / Reflection) ---
MODEL_FAST=gemma4:e2b             # Tier 0 — intent classification
MODEL_SMART=minimax/minimax-m2.7  # Tier 1 — skill reasoning (OpenRouter route)
MODEL_REFLECTION=gemma4:e4b       # Reflection plane (offline / heartbeat)
DEFAULT_MODEL=gemma4:e2b
```
with:
```
# --- Models (Tier 0 / Tier 1 / Reflection) ---
MODEL_FAST=gemma4:e2b             # Tier 0 — intent classification
MODEL_SMART=minimax/minimax-m2.7  # Tier 1 — skill reasoning (OpenRouter route)
MODEL_SMART_LOCAL=llama3.1:8b     # Tier 1 — skill reasoning (Ollama route: local weights or a -cloud tag)
MODEL_SMART_PROVIDER=ollama       # ollama | openrouter — explicit SMART-tier pin, no silent fallback
MODEL_REFLECTION=gemma4:e4b       # Reflection plane (offline / heartbeat)
DEFAULT_MODEL=gemma4:e2b
```

In the live `/home/pookster/.aegis/.env`, replace the line:
```
MODELS_PREFER_LOCAL=true
```
with:
```
MODEL_SMART_PROVIDER=ollama
```
(`MODEL_SMART_LOCAL=gemma4:e2b-mlx`, already present in this file, stays unchanged — it already names a real Ollama tag on the Mac Mini.)

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -c "from runtime.config import get_config; c = get_config(); print(c.models.smart_provider, c.models.smart_local)"` from `/home/pookster/projects/aegis` (with `AEGIS_ROOT` unset so it resolves to `~/.aegis`, matching how the running bot loads config).
Expected output: `ollama gemma4:e2b-mlx`

- [ ] **Step 5: Commit**

```bash
git add .env .env.example
git commit -m "docs(env): document MODEL_SMART_PROVIDER, replace MODELS_PREFER_LOCAL"
```

(The live `~/.aegis/.env` edit is not part of this commit — it lives outside the repo and is never committed, per this project's existing `.gitignore`/CLAUDE.md convention.)

---

### Task 9: Full verification — lint, type, security, full test suite, live bot smoke test

**Files:** None modified — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/pookster/projects/aegis && .venv/bin/python -m pytest -m unit -v 2>&1 | tail -60`
Expected: PASS, 0 failures. Pay particular attention to any other file that imports `ModelConfig`/`prefer_local` not yet caught — grep first:

Run: `grep -rln "prefer_local" --include="*.py" /home/pookster/projects/aegis | grep -v __pycache__`
Expected: no output (every reference was updated across Tasks 1-7).

Note: `tests/test_doctor_openrouter_model.py` was checked during planning and confirmed to have zero `prefer_local`/`smart_provider` dependency — it only exercises `scripts.doctor._check_openrouter_coding_model()` against `MODEL_CODING`/`MODEL_SMART`/`OPENROUTER_API_KEY`, none of which this design touches. No task modifies it; this is intentional, not an oversight.

- [ ] **Step 2: Run lint, type check, security scan**

Run: `cd /home/pookster/projects/aegis && make lint type security 2>&1 | tail -80`
Expected: no new ruff/mypy/bandit findings introduced by this change (pre-existing findings, if any, are out of scope — only check that the touched files are clean).

- [ ] **Step 3: Restart the live bot and confirm the footer + live routing**

The bot was left running from an earlier session (`scripts.telegram_smoke`, PID tracked in this session's background tasks) with the Tailscale-proxied Ollama at `127.0.0.1:11434` → Mac Mini. Restart it so it picks up the new code and the migrated `~/.aegis/.env`:

Run (background): `cd /home/pookster/projects/aegis && .venv/bin/python -u -m scripts.telegram_smoke`

Confirm in the log output: `telegram_smoke preflight:` passes, `starting long-poll` appears, no traceback.

- [ ] **Step 4: Manual confirmation via Telegram**

Ask the user to send `hi` to the bot. Expected: a reply arrives with a visible `_[ollama · gemma4:e2b-mlx]_` (or whichever `MODEL_SMART_LOCAL` is currently set) footer — this is the first real, live, provably-accurate confirmation of which provider answered, closing the exact gap that started this whole design (`deepseek-v4-pro:0813-cloud` shown via a static `/status` echo with no way to verify it was actually used).

- [ ] **Step 5: No commit** — this task is verification only, nothing to stage.
