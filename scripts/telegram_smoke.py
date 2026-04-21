"""Phase 7 Track B — manual Telegram smoke helper.

Preflight-and-run wrapper for driving a real bot against production
wiring: loads `AegisConfig`, sanity-checks the token / allowlist /
OpenRouter key / workspace layout, builds the `Application`, prints
the sessions-shard path and a checklist, then calls `run_polling()`.

This script never ships user secrets anywhere — it only echoes
whether a key is present and the shape of the allowlist. Exits
non-zero on preflight failure so an operator can fix config before
anything touches the network.

Usage:

    python -m scripts.telegram_smoke

Requires `python-telegram-bot>=21` installed (declared in pyproject
but optional for the unit gate), and the usual `~/.aegis/.env` +
`~/.aegis/config.json` that `doctor` validates.
"""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from runtime.config import AegisConfig, get_config


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _warn(msg: str) -> None:
    print(f"  \u26a0 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


def _preflight(cfg: AegisConfig) -> list[str]:
    """Return a list of hard-fail reasons. Empty list = ready to run."""
    errors: list[str] = []

    print("telegram_smoke preflight:")

    # Bot token
    if cfg.telegram.bot_token:
        _ok("TELEGRAM_BOT_TOKEN present")
    else:
        _fail("TELEGRAM_BOT_TOKEN missing — export it or add to ~/.aegis/.env")
        errors.append("bot_token")

    # Allowlist (fail-closed: empty allowlist denies everyone)
    if cfg.telegram.user_allowlist:
        _ok(f"user_allowlist: {len(cfg.telegram.user_allowlist)} id(s)")
    else:
        _fail(
            "user_allowlist empty — add your numeric Telegram user id "
            "under telegram.userAllowlist in ~/.aegis/config.json"
        )
        errors.append("allowlist")

    # OpenRouter (soft — missing key falls back to legacy placeholder chat)
    if cfg.providers.openrouter_api_key:
        _ok(f"OPENROUTER_API_KEY present (model: {cfg.models.smart})")
    else:
        _warn(
            "OPENROUTER_API_KEY missing — slash commands will work, "
            "but free-form chat replies with the legacy placeholder"
        )

    # Workspace layout (soft — missing persona files render as blank sections)
    for name in ("IDENTITY.md", "USER.md"):
        p = cfg.storage.workspace / name
        if p.is_file():
            _ok(f"{p}")
        else:
            _warn(f"{p} missing — system prompt will skip this section")

    # Sessions shard dir (where chat.turn.* + governance.decision land —
    # one `<session_id>.jsonl` file per process under today's date dir).
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    shard_dir = cfg.storage.sessions_dir / today
    print(f"  \u2139 sessions shard dir: {shard_dir}/ (one file per run)")

    return errors


def _print_checklist() -> None:
    print()
    print("smoke checklist — drive these against the live bot:")
    print("  1. /status                         (24h rollup — no writes)")
    print("  2. /pending                        (open proposals)")
    print("  3. /approve IMP-xxxxxxxx ok        (write slash → governance.decision)")
    print("  4. /apply CT-001 --dry-run         (long-running → edited reply)")
    print("  5. hi                              (free-form → chat.turn.context + reply)")
    print("  6. say you ran the cleanup tool    (D2 gate → ⚠️ banner on reply)")
    print("  7. /cron list                      (expect 4 SYS-* rows seeded on first boot:")
    print("                                      SYS-morning-brief, SYS-tier2-compress,")
    print("                                      SYS-reflection-sweep, SYS-vault-reindex)")
    print("  8. /cron pause SYS-morning-brief   (flip to paused; restart bot; /cron list")
    print("                                      should still show it paused — Track D1)")
    print("  9. /cron resume SYS-morning-brief  (restore default state before exiting)")
    print(" 10. /cron rm SYS-morning-brief      (expect refusal: SYS-* are guarded — D4)")
    print(" 11. push-path proof:                (skip — D3a already proved this via")
    print("                                      a temporary cron UPDATE on SYS-morning-brief.")
    print("                                      `echo` is intent-router-only; it has no")
    print("                                      argv_template, so the scheduler emits")
    print("                                      scheduler.job_failed[skill_misconfigured]")
    print("                                      and pushes nothing. To re-prove push, pick")
    print("                                      a schedulable skill (e.g. morning_brief) and")
    print("                                      UPDATE its cron_expr to fire in ~2 min.)")
    print()
    print("reminder: cron expressions are UTC. NZ is UTC+12 (+13 DST), so")
    print("  '0 7 * * *' = 19:00 NZST (winter) / 20:00 NZDT (summer). Adjust")
    print("  the seeded SYS-* jobs via `/cron add` or by editing them in place.")
    print()
    print("ctrl-c to stop. After exit, tail today's sessions shard to")
    print("confirm events landed with structural counts only (no bodies).")
    print("Scheduler emits scheduler.tick / scheduler.job_{started,succeeded,")
    print("failed,skipped_busy,skipped_stale} — structural counts, no argv, no output.")
    print()


def main(argv: list[str] | None = None) -> int:
    del argv  # no flags yet
    try:
        cfg = get_config()
    except (OSError, ValueError) as exc:
        print(f"failed to load config: {exc}", file=sys.stderr)
        return 2

    errors = _preflight(cfg)
    if errors:
        print()
        print(f"preflight failed ({len(errors)} blocker(s)): {', '.join(errors)}",
              file=sys.stderr)
        return 1

    # Lazy import so preflight failures don't require python-telegram-bot.
    try:
        from runtime.chat.telegram import build_application  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        print(f"failed to import build_application: {exc}", file=sys.stderr)
        return 2

    _print_checklist()

    try:
        app = build_application(cfg)
    except RuntimeError as exc:
        print(f"build_application failed: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"python-telegram-bot not installed ({exc}) — "
            "run `uv sync` (or `pip install -e .`) to pick up the telegram extra",
            file=sys.stderr,
        )
        return 2

    print("starting long-poll — ctrl-c to stop")
    # run_polling blocks until SIGINT; delegates its own signal handling.
    app.run_polling()
    return 0


def _storage_summary(cfg: AegisConfig) -> dict[str, Path]:
    """Exported for tests: the paths smoke output references."""
    return {
        "workspace": cfg.storage.workspace,
        "sessions_dir": cfg.storage.sessions_dir,
        "memory_db": cfg.storage.memory_db,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
