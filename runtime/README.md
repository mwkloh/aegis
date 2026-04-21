# `runtime/` — Plane 1 (Runtime)

> This directory is **Plane 1 (Runtime)**. Code here MAY: classify intent, select skills, emit tool-intent contracts, route to the harness, stream progress, render replies. Code here MUST NOT: modify its own files, mutate canonical memory (`USER.md`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `MEMORY.md`, `HEARTBEAT.md`), apply improvements, or write to `improvement/` or `coding_harness/`. Violations are bugs.

## Subpackages

| Path | Purpose |
| --- | --- |
| [`chat/`](chat/) | User-facing surfaces (CLI now; Telegram later). Transport + display only. |
| [`intent/`](intent/) | Tier 0 intent classifier. Small local model. Returns `{intent, confidence}` only. |
| [`skills/`](skills/) | Skill registry + YAML descriptors. **Read-only at runtime.** |
| [`reasoning/`](reasoning/) | Tier 1 skill-scoped reasoning. Produces a tool-intent contract. |
| [`harness/`](harness/) | Adapter over OpenHarness + tool implementations. The only path to side effects. |
| [`model_router/`](model_router/) | Routes a request to the right model client (Ollama / OpenRouter). |
| [`events/`](events/) | Structured event stream. Append-only JSONL under `~/.aegis/workspace/sessions/`. |
| [`config.py`](config.py) | Loads `~/.aegis/.env` and `~/.aegis/config.json`. The single config gateway. |

## Runtime invariants (enforced by tests)

1. No module under `runtime/` imports from `improvement/` or `coding_harness/`.
2. No module under `runtime/` opens any canonical .md file in write mode.
3. The only filesystem writes performed by `runtime/` go to `~/.aegis/workspace/sessions/*.jsonl`.
4. Every external boundary (CLI input, model response, skill descriptor, tool result) is validated by Pydantic.
5. Skill descriptors are loaded with `yaml.safe_load`. No `yaml.load` anywhere.
6. Frontier models (Tier 1) are never required for correctness — Tier 0 + skill scaffolding must produce a valid contract on their own.
