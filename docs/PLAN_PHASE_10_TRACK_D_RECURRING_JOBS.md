# Phase 10 Track D — Wire built-in recurring jobs

> Status: Proposed
> Predecessor: `PLAN_PHASE_10_SCHEDULER.md` (Tracks A/B/C + integration ✅)

## Goal

Turn the scheduler from "operator toy" into a load-bearing piece of
AEGIS by seeding four built-in jobs at boot:

1. `morning_brief` — 07:00 local, daily
2. `vault_reindex` — 03:30 local, daily (low traffic window)
3. `tier2_compress` — 03:00 local, daily (before reindex so today's
   compressed records land in the vault)
4. `reflection_sweep` — 03:15 local, daily (between compress + reindex)

All four idempotent, all four silent-on-success (push nothing), all
four driven by existing code — no net-new behavior, just wiring.

## Non-goals

- `/cron run` immediate-fire seam (separate track, blocked on engine
  exposing a one-shot entrypoint).
- Per-user schedules. AEGIS is single-operator by design; these are
  system jobs owned by `created_by = 0`.
- Runtime mutation of built-in job schedules via `/cron`. Operators
  can `pause`/`resume` system jobs but not `rm` or edit cron exprs —
  config changes go through code review.

## Design

### D1 — Store API addition

Add one method to `ScheduledJobStore`:

```python
def upsert_system_job(
    self,
    *,
    job_id: str,         # well-known, e.g. "SYS-morning-brief"
    cron_expr: str,
    skill: str,
    args: tuple[str, ...],
    now: datetime,
) -> ScheduledJob:
    """Insert iff absent. Existing system job rows are left alone so
    operator pauses survive restarts.
    """
```

Invariants:

- `created_by = 0` marks system-owned rows (operators always have a
  positive Telegram user id).
- `job_id` uses `SYS-<slug>` prefix — distinct from operator-minted
  `JOB-<random>` ids. `/cron rm` handler rejects `SYS-*` ids.
- Operator-set `paused` flag is preserved across boots.

### D2 — Seed module

New file: `runtime/scheduler/seed.py`

```python
SYSTEM_JOBS: tuple[SystemJobSpec, ...] = (
    SystemJobSpec(
        id="SYS-morning-brief",
        cron_expr="0 7 * * *",
        skill="morning_brief",
        args=(),
    ),
    SystemJobSpec(
        id="SYS-tier2-compress",
        cron_expr="0 3 * * *",
        skill="tier2_compress",
        args=(),
    ),
    SystemJobSpec(
        id="SYS-reflection-sweep",
        cron_expr="15 3 * * *",
        skill="reflection_sweep",
        args=(),
    ),
    SystemJobSpec(
        id="SYS-vault-reindex",
        cron_expr="30 3 * * *",
        skill="vault_reindex",
        args=(),
    ),
)

def seed_system_jobs(store: ScheduledJobStore, *, now: datetime) -> None:
    for spec in SYSTEM_JOBS:
        store.upsert_system_job(
            job_id=spec.id, cron_expr=spec.cron_expr,
            skill=spec.skill, args=spec.args, now=now,
        )
```

Called from `build_scheduler` after store construction, before
engine instantiation. Never raises — a missing catalog entry for one
job shouldn't block the others; `JobRunner` will classify the fire
as `unknown_skill` and move on.

### D3 — Missing catalog entries (three new YAMLs)

Each mirrors `morning_brief.yaml`'s shape: `tools:` block with
`argv_template` invoking a CLI module.

**`runtime/skills/catalog/vault_reindex.yaml`**

```yaml
id: vault_reindex
version: 0.1.0
description: Full vault reindex (all sources). Silent on success.
tool: vault_reindex
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: vault_reindex
    argv_template:
      - python
      - -m
      - runtime.chat.memory.vault_indexer
      - --reindex
      - --all
    timeout_ms: 300000
    allow_net: false
```

Requires adding `if __name__ == "__main__":` + argparse to
`runtime/chat/memory/vault_indexer.py` (or a thin `scripts/` wrapper).
Keep stdout empty on success so the scheduler stays silent.

**`runtime/skills/catalog/tier2_compress.yaml`**

```yaml
id: tier2_compress
version: 0.1.0
description: Run tier-2 compression sweep. Silent on success.
tool: tier2_compress
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: tier2_compress
    argv_template:
      - python
      - -m
      - runtime.chat.memory.compressor
    timeout_ms: 300000
    allow_net: true        # OpenRouter for compression summary
```

Requires `__main__` in `runtime/chat/memory/compressor.py` that loads
config, finds stale tier-2 records, runs the sweep, exits 0. Stdout
empty on success.

**`runtime/skills/catalog/reflection_sweep.yaml`**

```yaml
id: reflection_sweep
version: 0.1.0
description: Run reflection pattern sweep over recent events. Silent on success.
tool: reflection_sweep
args_schema:
  type: object
  additionalProperties: false
tools:
  - name: reflection_sweep
    argv_template:
      - python
      - -m
      - runtime.reflection.cli
      - --since
      - "24h"
    timeout_ms: 120000
    allow_net: false
```

`runtime/reflection/cli.py` already has `argparse`. Verify exit
codes + silent-success behavior.

### D4 — `/cron rm` guard

`runtime/chat/telegram/cron_handler.py` — in the `rm` branch, reject
job ids starting with `SYS-`:

```python
if job_id.startswith("SYS-"):
    return "System jobs can't be removed. Use /cron pause <id> instead."
```

`list` and `pause`/`resume` continue to work on system jobs.

### D5 — Smoke test extension

Add to `scripts/telegram_smoke.py` checklist:

```
10. /cron list                    (should show 4 SYS-* jobs seeded at boot)
11. /cron pause SYS-morning-brief (pause survives restart)
```

## Test plan

New test files:

- `tests/test_scheduler_seed.py` — `upsert_system_job` idempotency,
  paused-state preservation across re-seeds, `SYS-*` prefix guard.
- `tests/test_skill_catalog_recurring.py` — all four system skill
  YAMLs load cleanly via `SkillRegistry.from_directory`, each
  resolver returns an argv.

Extensions to existing tests:

- `tests/test_telegram_cron.py` — `/cron rm SYS-xxx` returns the
  refusal string; `/cron rm JOB-xxx` unaffected.
- `tests/test_telegram_bot.py` — `build_application` boot seeds four
  rows; second call leaves counts unchanged.

## Rollout sequence

1. **D1 + D2 + tests** — store + seed module, all four seeded with
   currently-missing skills (they'll classify as `unknown_skill` —
   harmless log noise until the CLIs land).
2. **D4 + tests** — `/cron rm` guard. Low-risk.
3. **D3a — morning_brief proof point** — already runnable; confirm
   it fires cleanly before building the others. No catalog change
   needed; `morning_brief.yaml` already ships.
4. **D3b — vault_reindex CLI + YAML** — add `__main__`, YAML, tests.
5. **D3c — tier2_compress CLI + YAML** — add `__main__`, YAML, tests.
6. **D3d — reflection_sweep YAML** — CLI already exists; just add
   the catalog entry + verify exit codes.
7. **D5 smoke update** — verify live bot behavior.

Each step ships independently — partial progress leaves the system
in a working state (seeded jobs that can't resolve just log
`unknown_skill` every 24h; not a regression).

## Design pins (carry from Tracks A-C)

- Never raises out of engine or runner.
- Silent-success path preserved: stdout empty → no push.
- Structural events only: `scheduler.tick`,
  `scheduler.job_{started,succeeded,failed}` — no argv, no output.
- Single-operator delivery: `user_allowlist[0]`.
- Argv-only: no shell, no substitution in scheduler path.

## Open questions

1. **Timezone.** ~~Cron expressions parse in UTC today. If the
   operator is in NZST (UTC+12/13), `0 7 * * *` fires at 19:00 local
   in summer.~~ **Resolved 2026-04-21 (D3a smoke) → option (b):**
   cron stays UTC; make it visible in operator surfaces. See
   "Follow-ups from D3a smoke" below for concrete work.
2. **First-fire timing.** If `tier2_compress` + `reflection_sweep` +
   `vault_reindex` all land within 30 minutes, do we care about the
   busy-skip interaction? Current policy-A will skip later jobs if
   the earlier one is still running. For daily cadence that's fine —
   tomorrow is another chance.

## Follow-ups from D3a smoke (2026-04-21)

D3a smoke ran a real bot with a temp cron UPDATE and confirmed the
end-to-end fire path (engine tick → runner → subprocess → Telegram
delivery). `last_status='ok'`, `last_run_at` stamped. The smoke
surfaced one bug in the operator's mental model and two polish
items in the code:

- **Bug (in the operator, not the code):** first attempt set
  `cron_expr = '51 11 * * *'` expecting NZ 11:51; engine computed
  next fire as UTC 11:51 = NZ 23:51 next day. Fix was trivial
  (set `57 23 * * *` for NZ 11:57), but the failure mode was silent
  — no error, just a 24h delay. This is exactly the surprise
  Open Question #1 worried about. Locking in decision (b) with
  these three visibility fixes:

  1. **`runtime/scheduler/cron.py describe()`** — suffix `(UTC)` on
     labels so `/cron list` reads e.g. `daily @ 07:00 (UTC)` instead
     of `daily @ 07:00`. Label tz ambiguity kills trust first.
  2. **`/cron add` help text** in `runtime/chat/telegram/cron_handler.py`
     — single line: "Cron expressions are UTC." Add to the existing
     help block, no new flags.
  3. **D5 smoke checklist** — first step should be a cron literacy
     check ("today is UTC <date>, add a one-shot for now+2min using
     UTC hour/minute") so the operator exercising the smoke doesn't
     fight the same 24h delay.

- **Polish — seed defaults are semantically off.** `runtime/scheduler/seed.py`
  has:

    ```
    SYS-morning-brief   0 7  * * *   → NZST 19:00 / 20:00 DST  (not morning)
    SYS-tier2-compress  0 3  * * *   → NZST 15:00 / 16:00 DST
    SYS-reflection-sweep 15 3 * * *  → NZST 15:15 / 16:15 DST
    SYS-vault-reindex   30 3 * * *   → NZST 15:30 / 16:30 DST
    ```

  These "look like" reasonable cron times but all fire mid-afternoon
  local. Options:
  - (a) Rewrite defaults in UTC to target NZ local (e.g.
    morning_brief = `0 19 * * *` for NZ 07:00 NZST). Ties the seed
    to one timezone.
  - (b) Leave as-is and document loudly that operators are expected
    to adjust via `/cron` after first boot. Matches decision (b)'s
    spirit: cron is a low-level knob, not a smart scheduler.
  - (c) Pick UTC defaults that at least don't cluster in the
    operator's work-hours. `0 16 * * *` = NZ 04:00 — a genuine
    "overnight maintenance" slot regardless of DST.

  Recommendation: **(c)** for the 3 silent-success jobs (move them
  to `0 16 * * *`, `15 16 * * *`, `30 16 * * *`) and **(a) for
  morning_brief** only (`0 19 * * *` = NZ 07:00 NZST) since its
  semantic *is* "morning" and that word is in the skill name.
  Document in `/cron list` describe output and in plan comments.

- **Observation (not a defect):** engine re-reads `cron_expr` from
  the store on every tick via `list_all()` — DB UPDATEs land on the
  next tick. This made the smoke trivial (edit cron in DB, wait 60s).
  Preserve this property; don't add in-memory caching in the engine
  without a cache-invalidation path.

These follow-ups fit within Track D scope; schedule them after D3d
and before D5 so the smoke checklist can exercise the tz labelling.
