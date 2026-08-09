"""Production entry-point for the AEGIS Telegram bot.

Designed for supervised execution (macOS launchd, systemd, etc.).
Unlike `scripts/telegram_smoke.py` this module is quiet by design:
no preflight tables, no smoke checklists. Structured logging only.

Usage:
    python -m runtime.serve

Environment:
    LOG_LEVEL   Logging level (default: INFO)
    AEGIS_ROOT  Root config dir (default: ~/.aegis)
"""
from __future__ import annotations

import logging
import os
import sys

from runtime.config import get_config

# Re-export for test patching — tests patch serve.build_application directly.
try:
    from runtime.chat.telegram import build_application
except ImportError:  # pragma: no cover — the real check is inside main()
    build_application = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def main() -> int:
    _setup_logging()

    # --- fail-fast config validation ---
    cfg = get_config()

    if not cfg.telegram.bot_token:
        logger.error(
            "TELEGRAM_BOT_TOKEN is missing — set it in ~/.aegis/.env "
            "or export TELEGRAM_BOT_TOKEN before starting"
        )
        return 1

    if not cfg.telegram.user_allowlist:
        logger.error(
            "telegram.user_allowlist is empty — add your numeric Telegram "
            "user id under telegram.userAllowlist in ~/.aegis/config.json"
        )
        return 1

    # --- import the Telegram layer (lazy: avoids crash on bare venv) ---
    # Use the module-level name so tests can patch serve.build_application.
    import runtime.serve as _self  # noqa: PLC0415, PLW0406

    _build = _self.build_application
    if _build is None:
        logger.error(
            "python-telegram-bot is not installed — "
            "run `uv sync` (or `pip install -e .`) to pick up the telegram extra"
        )
        return 2

    try:
        app = _build(cfg)
    except ImportError as exc:
        logger.error(
            "python-telegram-bot not installed (%s) — "
            "run `uv sync` (or `pip install -e .`) to pick up the telegram extra",
            exc,
        )
        return 2
    except Exception as exc:
        logger.exception("build_application failed: %s", exc)
        return 1

    logger.info("starting long-poll")
    try:
        app.run_polling()
    except KeyboardInterrupt:
        logger.info("interrupted — shutting down cleanly")
        return 0
    except Exception as exc:
        logger.exception("fatal error during polling: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
