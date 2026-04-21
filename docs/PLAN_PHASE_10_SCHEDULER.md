# Phase 10 — Autonomous Scheduler & Proactive Agent

> Status: **Drafted 2026-04-19**. Builds on Phase 8 (skills +
> intent router) and Phase 7 (Telegram operator surface).
> Phase 9 (Web UI) remains deferred — the scheduler unlocks
> proactive behavior well before a browser console is needed.
>
> **The unlock.** Up to Phase 8, AEGIS is purely reactive — it
> replies only when the operator speaks. Many load-bearing jobs
> (vault reindex, tier-2 compression, reflection sweep, morning
> brief) already exist as one-shot entrypoints; they just need a
> runner. Phase 10 adds that runner *inside the long-lived
> Telegram bot process*, re-using the same EventStream and
> delivery surface — no new daemons, no system cron dependency.

## 1. Goal

Give AEGIS the ability to:

1. **Run scheduled jobs autonomously** (cron-like expressions)
   from inside the Telegram bot process, with no external
   supervisor.
2. **Trigger skills by time**, not just by intent match — e.g.
   "morning_brief at 07:00 daily", "reindex_vault every 6h".
3. **Push results to the operator** over Telegram when a job
   produces user-facing output (so the operator learns of a
   morning brief without having to ask).
4. **Expose schedule management** via `/cron` slashes so the
   operator can add/list/remove/run-now without a config-file
   round-trip.

End-state demos:

```text
You:  /cron add "0 7 * * *" morning_brief
Bot:  ✅ JOB-a3f1 scheduled · morning_brief · daily 07:00 UTC
      next run: 2026-04-20 07:00:00 UTC (in 14h 23m)

You:  /cron list
Bot:  2 active jobs:
       • JOB-a3f1  morning_brief   0 7 * * *    next: 14h 23m
       • JOB-b902  reindex_vault   0 */6 * * *  next:  2h 05m

07:00 UTC → (auto-push, no user input)
Bot:  📅 Morning brief (2026-04-20)
      [rendered skill output]
```

## 2. Non-negotiables

Carry-forward from Phases 0–8 (unchanged). **New for Phase 10:**

1. **In-process, not a new daemon.** The scheduler runs as an
   asyncio task inside `build_application()` alongside the
   Telegram long-poll. Single PID, single restart story,
   `/restart` tears it down cleanly.
2. **Never raises into the bot event loop.** A job failure
   emits `scheduler.job_failed` and carries on. One bad job
   must not take down the long-poll.
3. **Skill invocation goes through the existing registry.** No
   private "scheduled skill" path — if a skill can run by intent,
   it can run by schedule. Unknown skill name → `scheduler.job_failed`
   at registration time, not at tick time.
4. **Busy check honors `InFlightRegistry`.** A scheduled job that
   maps to a long-running skill waits (or is skipped, see §5)
   when the operator already has that skill in flight. Prevents
   double-runs.
5. **Deterministic testability.** The tick loop takes an
   injectable `Clock = Callable[[], datetime]` and an injectable
   `Sleeper = Callable[[float], Awaitable[None]]`. Unit tests
   never wait on wall-clock time.
6. **Every tick emits an event.** `scheduler.tick`,
   `scheduler.job_started`, `scheduler.job_succeeded`,
   `scheduler.job_failed` land on the session shard for `/logs`
   visibility. Payloads are structural only (job_id, skill name,
   latency_ms, error_class) — never skill outputs.
7. **Never deliver stale pushes.** If the process was down when a
   job's scheduled time passed, do **not** retroactively fire on
   startup. Log `scheduler.skipped_stale` and continue. Prevents
   a post-restart storm.

## 3. Track structure

### Track A — scheduler engine (`runtime/scheduler/`)
The pure engine — no Telegram, no skills knowledge.
- Cron-expression parser (5-field Unix cron, subset).
- Job record (sqlite-backed, reuses `aegis-index.db`).
- Async tick loop with injectable clock + sleeper.
- Event emission through `EventStream`.

### Track B — skill integration
- `JobRunner` adapter that looks up the skill in
  `SkillRegistry`, resolves args, and invokes through the
  same path `IntentRouter` uses.
- Delivery callback: if a skill returns user-facing text,
  push it to the operator's Telegram chat.

### Track C — operator surface
- `/cron add <expr> <skill> [args...]` — register.
- `/cron list` — show active jobs sorted by next-run.
- `/cron rm <job_id>` — delete.
- `/cron run <job_id>` — manual trigger (bypasses cron,
  still honors busy check).
- `/cron pause <job_id>` / `/cron resume <job_id>`.
- Help descriptions auto-merged into `/help`.

## 4. Data model

```python
class ScheduledJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str              # "JOB-" + 4 hex chars
    cron_expr: str       # "0 7 * * *"
    skill: str           # "morning_brief"
    args: tuple[str, ...] = ()
    created_at: datetime
    created_by: int      # Telegram user_id
    last_run_at: datetime | None = None
    last_status: Literal["ok", "failed", "skipped"] | None = None
    paused: bool = False
```

Storage: new `scheduled_jobs` table in `aegis-index.db`.
Schema migration is additive — no touch to Phase 7 tables.

## 5. Busy-skip policy (open question for review)

When a scheduled tick fires and the target skill is already
in flight via `InFlightRegistry`, two reasonable policies:

- **A. Skip this tick.** Emit `scheduler.skipped_busy`, wait
  for the next scheduled occurrence. Simpler. Safer.
- **B. Enqueue and retry after N seconds.** More work-preserving
  but introduces an unbounded queue and edge cases.

**Recommendation: A.** Consistent with the "never raise, never
block" posture. Can add B later if real usage shows skips
becoming a problem.

## 6. Cron expression support

Start with a **subset** of standard cron:
- `m h dom mon dow` (5 fields)
- `*`, `*/N`, comma lists (`1,15,30`), ranges (`1-5`)
- **Not** supported initially: `@reboot`, `@daily`, month names,
  day-of-week names, `L`/`W`/`#`.

Ship with a helper library (`croniter` — 6M downloads/month,
maintained, 800 GH stars) rather than custom parsing. One
dependency, well-tested, handles DST correctly.

**Security note:** `croniter` parses text — we validate expressions
at `/cron add` time and reject anything that doesn't round-trip,
so malformed input produces an explicit error rather than a silent
misfire.

## 7. Delivery

When a job's skill returns user-facing text, push it via the
same `bot.send_message` path that `/brief` already uses. The
target chat is `TELEGRAM_CHAT_ID` from config — single-operator
constraint from Phase 7 carries over (no broadcast).

If the skill returns empty / whitespace, the job is still
recorded as "ok" but nothing is pushed. This lets
`reindex_vault`-style jobs run silently.

## 8. File list

**New:**
- `runtime/scheduler/__init__.py` — re-exports
- `runtime/scheduler/cron.py` — croniter wrapper, validation
- `runtime/scheduler/job.py` — `ScheduledJob` model
- `runtime/scheduler/store.py` — sqlite persistence
- `runtime/scheduler/engine.py` — async tick loop
- `runtime/scheduler/runner.py` — skill invocation adapter
- `runtime/chat/telegram/cron_handler.py` — `/cron` slashes

**Modified:**
- `runtime/chat/telegram/bot.py` — `build_application` spawns
  scheduler task alongside long-poll; `/cron` registered.
- `runtime/chat/telegram/handlers.py` — add `cron_handler` to
  `build_write_handlers` (it's a write — mutates schedule state).
- `runtime/events.py` — new `EventType` members:
  `SCHEDULER_TICK`, `SCHEDULER_JOB_STARTED`,
  `SCHEDULER_JOB_SUCCEEDED`, `SCHEDULER_JOB_FAILED`,
  `SCHEDULER_SKIPPED_BUSY`, `SCHEDULER_SKIPPED_STALE`.
- `pyproject.toml` — add `croniter` dependency.

**Tests (target ~30 new):**
- `tests/test_scheduler_cron.py` — expression parsing, next-run
  math, edge cases (midnight, DST, leap day).
- `tests/test_scheduler_store.py` — CRUD on sqlite, concurrent
  access, schema migration.
- `tests/test_scheduler_engine.py` — tick loop with injected
  clock/sleeper; never-raise on job failure; stale-skip on
  startup; busy-skip when registry reports in-flight.
- `tests/test_scheduler_runner.py` — skill invocation, args
  resolution, delivery callback.
- `tests/test_telegram_cron.py` — `/cron add/list/rm/run/pause`
  handlers, auth checks, usage errors.

## 9. Build order (TDD)

1. **Track A1** — `cron.py` + `job.py` (pure, no IO).
2. **Track A2** — `store.py` (sqlite persistence).
3. **Track A3** — `engine.py` (tick loop, no skill wiring yet —
   test with a stub callable).
4. **Track B** — `runner.py` (skill adapter + delivery).
5. **Track C** — `cron_handler.py` + Telegram wiring.
6. **Integration smoke** — update `scripts/telegram_smoke.py`
   checklist; manual test: add a `* * * * *` job, watch it fire.

## 10. Out of scope for Phase 10

- Multi-operator job ownership (Phase 7.5).
- Job chaining / DAGs.
- Skill-level retries with backoff.
- Per-job timeouts (long-running skills already have the
  subprocess runner's timeout).
- Web UI for scheduling (Phase 9, still deferred).
- Automatic job generation by the reflection plane (future).

## 11. Success criteria

- Operator can `/cron add "*/2 * * * *" echo hello` and receive
  a Telegram push every 2 minutes.
- Existing `morning_brief` skill runs daily at 07:00 local with
  no external cron.
- `/restart` tears down and respawns the scheduler cleanly;
  no stale-push storm on restart.
- `ruff` clean, `mypy` clean, all new tests passing, existing
  942 tests still green.
