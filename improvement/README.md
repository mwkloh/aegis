# `improvement/` — Planes 2 + 3 (Reflection + Improvement)

> This directory holds **out-of-band** artifacts and (later) reflection code. Nothing here may be imported by `runtime/`. Every change here is **human-gated**.

See [`docs/CLOSE_LOOP_IMPROVEMENT_ARCHITECTURE.md`](../docs/CLOSE_LOOP_IMPROVEMENT_ARCHITECTURE.md) for the full design.

## Artifacts (existing — preserved)

| File | Role |
| --- | --- |
| `EVENTS.md` | Raw runtime signals (failures, retries, corrections) — append-only |
| `PATTERNS.md` | Clustered recurring issues from reflection |
| `PROPOSALS.md` | Candidate improvements |
| `DECISIONS.md` | Approved / rejected / deferred decisions |
| `CODING_TASKS.md` | Approved implementation tasks for the coding harness |
| `IMPROVEMENT.md` | Governance: what improvement means, what must not change |
| `GEMMA_4B_REFLECTION_CONFIG.yaml` | Reflection model config |
| `END_TO_END_IMPROVEMENT_WALKTHROUGH.md` | Worked example |

## Plane 2 (Reflection) — not yet active

Will read `EVENTS.md` + session JSONLs, cluster, draft proposals. **Read-only.**

## Plane 3 (Improvement) — partially active (manual)

Human edits `PROPOSALS.md` → `DECISIONS.md` → `CODING_TASKS.md`. The coding harness drafts diffs into [`../coding_harness/diffs/`](../coding_harness/diffs/). Merges remain manual.
