# Explicit Model-Provider Routing Design

## Goal

Aegis's stated purpose is to develop a harness that gets *local* models to reliably decompose and execute multi-step tool calls. Today two separate reply pipelines pick a SMART-tier model client in two different, inconsistent ways:

- `build_chat_pipeline` (`runtime/chat/telegram/bot.py:717`) uses `ModelRouter.route(SMART)`, which honors `models.prefer_local` — but falls back to OpenRouter silently whenever a 500ms Ollama liveness probe fails, even transiently.
- `build_harness_dispatcher` (`runtime/chat/telegram/bot.py:813`) ignores `ModelRouter` entirely for its `Tier1Reasoner` and final-answer synthesizer, hardcoding `OpenRouterClient` + `cfg.models.smart` unconditionally (`bot.py:902`, `:913`). Only its intent classifier touches Ollama.

Both behaviors mean the operator cannot reliably know, or control, whether a given reply was reasoned about by a local (or Ollama-cloud-proxied) model or by a cloud frontier model via OpenRouter. This surfaced directly: `MODELS_PREFER_LOCAL=true` was set, yet the harness answered via OpenRouter's `deepseek-v4-pro:0813-cloud`, discovered only by manual source-tracing after the fact.

This design replaces implicit, probe-driven fallback with an **explicit, config-pinned provider choice**, applied uniformly to both pipelines, with the actual per-reply decision made visible in the Telegram chat itself.

**Intended deployment stance for this machine**: `smart_provider=ollama` is the everyday default — routed through the Mac Mini's Ollama daemon over the Tailscale proxy, covering both genuinely-local models and Ollama Cloud's hosted open-weight catalog (deepseek, qwen, llama, etc. — Ollama's own infrastructure, *not* a proxy to proprietary APIs). `openrouter` is kept configured as a deliberate escape hatch for reaching proprietary frontier models (Claude, GPT, Gemini) that Ollama Cloud doesn't serve — flipped to on purpose, not fallen into.

## Non-Goals

- No new provider client classes. `OllamaClient` and `OpenRouterClient` (both implementing the existing `ModelClient` protocol in `runtime/llm/clients/base.py`) are sufficient — confirmed OpenRouter already reaches Anthropic/OpenAI/Gemini/etc. under one API, and Ollama already serves both genuinely-local and Ollama-cloud-proxied (`-cloud` tagged) models through the same daemon.
- No change to `FAST` / `FAST_LOCAL` / `REFLECTION` tier routing — these are already unconditionally Ollama and are untouched.
- No change to the existing three-layer reply fallback order (`HarnessDispatcher` → `ChatPipeline` → static stub reply) — only how each layer picks its own provider internally.
- No runtime/Telegram-command-driven provider switching. Config-file only, matching the existing `MODEL_FAST` / `MODEL_SMART` env-var pattern.

## Architecture

### Config schema (`runtime/config.py`)

`ModelConfig.prefer_local: bool` is removed. New field:

```python
smart_provider: Literal["ollama", "openrouter"] = "ollama"
```

Read from a new env var `MODEL_SMART_PROVIDER` (default `"ollama"` if unset — local-first by default, matching project intent). `models.smart` (OpenRouter model string) and `models.smart_local` (Ollama model string — may be a genuinely-local tag or a `-cloud` tag) are unchanged; they are simply no longer selected by fallback heuristics, only by which provider is pinned.

`.env.example` (repo template) replaces:
```
MODELS_PREFER_LOCAL=true
```
with:
```
MODEL_SMART_PROVIDER=ollama   # ollama | openrouter — explicit, no silent fallback
```
The live `~/.aegis/.env` gets the same migration.

### `ModelRouter.route()` (`runtime/llm/router.py`)

`SMART` tier becomes a direct lookup, no liveness probe in the decision:

```python
if tier is ModelTier.SMART:
    if models.smart_provider == "ollama":
        return ModelTarget(tier=tier, model=models.smart_local,
                            provider="ollama", base_url=providers.ollama_base_url)
    return ModelTarget(tier=tier, model=models.smart,
                        provider="openrouter", base_url=providers.openrouter_base_url)
```

The `degraded` field and the silent openrouter/local-fallback branches are removed for `SMART` — there is no fallback within `route()` any more. `is_local_ready()` stays on the class (unchanged implementation), but its only remaining caller-facing purpose is gating whether `build_harness_dispatcher` can construct its classifier at all, which unconditionally requires Ollama regardless of `smart_provider`.

### Builder functions (`runtime/chat/telegram/bot.py`)

Both `build_chat_pipeline` and `build_harness_dispatcher` call `router.route(ModelTier.SMART)` and construct **only** the client the target names:

- `target.provider == "ollama"` → construct `OllamaClient`; on `OllamaHostError` or the caller's own reachability check failing, log and `return None`.
- `target.provider == "openrouter"` → construct `OpenRouterClient`; on `OpenRouterConfigError`, log and `return None`.

This is a fail-closed change: `build_harness_dispatcher` already returns `None` for every other missing hard dependency (`no_ollama`, `ollama_host`, `no_openrouter`, `no_file_tools`, etc.) — pinned-provider failure joins that existing pattern rather than introducing new fallback semantics. No cross-provider reroute happens inside either builder.

`build_harness_dispatcher`'s classifier (`ModelBackedClassifier`) keeps using `OllamaClient` + `cfg.models.smart_local` unconditionally — classification is a separate, always-local concern from SMART-tier reasoning/synthesis, and this was already true today.

### Observability: live per-reply attribution

Every Telegram reply gets a small footer showing the provider and model that actually produced it:

```
hi there! how can I help?

_[ollama · gemma4:e2b-mlx]_
```

This applies to replies from both `ChatPipeline.turn()` (`runtime/chat/pipeline.py`, which already threads `model_name` through per-turn structured logging at lines 245/257/283 — just needs the same value surfaced into the reply text, not only the logger) and `HarnessDispatcher`'s synthesized replies (`FIRED` outcome) and the `startup_message`/`/status` surfaces should also read the live `smart_provider` pin rather than only `cfg.models.smart`, so they can't misleadingly imply a provider that isn't actually pinned.

## Failure Semantics Summary

| `smart_provider` | Provider reachable? | `build_chat_pipeline` | `build_harness_dispatcher` |
|---|---|---|---|
| `ollama` | yes | Ollama client, `smart_local` model | Ollama client, `smart_local` model (reasoning/synthesis) |
| `ollama` | no | `None` → falls through to stub reply | `None` → falls through to `ChatPipeline` (which is also `None` if pinned to the same unreachable provider) |
| `openrouter` | yes (key present) | OpenRouter client, `smart` model | OpenRouter client, `smart` model |
| `openrouter` | no (key missing) | `None` | `None` |

No layer silently substitutes the other provider. This is a deliberate behavior change: previously, an unreachable local Ollama would silently degrade to OpenRouter in `ChatPipeline`; now it fails closed, surfacing the outage instead of masking it with a cloud model standing in unannounced.

Note `build_harness_dispatcher`'s classifier requirement is independent of `smart_provider`: it always needs Ollama reachable (`ModelBackedClassifier` is unconditionally Ollama-backed), even when `smart_provider=openrouter` pins reasoning/synthesis elsewhere. So with `smart_provider=openrouter`, the harness dispatcher needs *both* Ollama (classifier) *and* OpenRouter (reasoning) available — losing either disables it.

## Testing

Existing coverage to update (no new test files needed):

- `tests/test_model_router.py` — replace `prefer_local`/liveness-probe-driven SMART routing tests with `smart_provider` pin tests (both values, plus the removed degrade-to-local branch).
- `tests/test_chat_pipeline.py` — update `build_chat_pipeline` fixtures/assertions for the new fail-closed-only behavior (no more `degraded=True` local fallback case).
- `tests/test_harness_dispatcher.py` — add coverage for `build_harness_dispatcher` actually branching on `smart_provider` (today it has no such test since the client was hardcoded); assert `Tier1Reasoner`/synthesizer receive whichever client the pin names.
- `tests/test_telegram_bot.py` — update any `prefer_local` references (confirmed present via grep) to `smart_provider`.
- `tests/test_doctor_openrouter_model.py` — check for `prefer_local` assumptions in the `doctor` script's config validation; update if present.

## Migration

- `~/.aegis/.env` (this machine, live): add `MODEL_SMART_PROVIDER=ollama`, remove `MODELS_PREFER_LOCAL`.
- `.env.example` (repo, tracked): same substitution, so the documented surface stays accurate for future setups.
- `~/.aegis/config.json`: no change — confirmed earlier this session that `runtime/config.py` never reads a `models`/`providers` block from `config.json`; all model config is env-var driven.

## Open Follow-ups (explicitly out of scope here)

- `MODEL_REFLECTION=gemma4:e4b-mlx` still doesn't match anything on the Mac Mini's Ollama catalog (a plain `gemma4:e4b` exists, not the `-mlx` variant) — the user is pulling it separately; unrelated to this design.
- The Tailscale forwarding proxy (`~/.aegis/ollama-tailscale-proxy.py`) is a manually-started process, not yet supervised (systemd/launchd) — out of scope for this design, noted for a future operational hardening pass.
