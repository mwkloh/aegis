# `memory/` — Physical memory layer

> This is the **physical** memory layer (sqlite-vec + bge-m3). The **canonical** memory files (`USER.md`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `MEMORY.md`, `HEARTBEAT.md`) live under `~/.aegis/workspace/` and are **never** written by this layer or by `runtime/`. Long-term writes are proposal-only and gated by the Improvement plane.

## Tiers (per [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §8.1)

| Tier | Purpose | Writable by Agent |
| --- | --- | --- |
| Preferences | Habits, defaults | No (proposal only) |
| Identity | Long-term facts | No |
| Episodic | Experiences and outcomes | No |
| External | Obsidian vault, notes | No |
| Execution | Task state | Yes (harness only) |

## Modules

| File | Role | Phase 0 status |
| --- | --- | --- |
| `tiers.py` | Enum + read APIs across tiers | stub |
| `store_sqlite.py` | sqlite-vec backed vector + KV store | stub (creates empty DB) |
| `embeddings.py` | bge-m3 client (lazy) | stub (not invoked Phase 0) |

## Storage location

- DB: `~/.aegis/workspace/memory/aegis-index.db` (sqlite-vec)
- Canonical .md: `~/.aegis/workspace/*.md` (read-only from this layer)
