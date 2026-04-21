"""In-place process restart for the Telegram operator surface.

`/restart` replaces the running bot image so operators can pick up new
code or config without shell access. We use `os.execv` — not a
supervisor or `subprocess.Popen` — because there is no external
watcher on the default deployment; the bot is a single long-lived
Python process launched from the shell or systemd user unit. `execv`
replaces the image in place, preserving PID + file descriptors,
which keeps the operator's existing Telegram session alive.

No shell is invoked. `sys.orig_argv[0]` is the interpreter path Python
booted with and the remaining entries are Python-level tokens
(``-m``, module names, flags). This is not a command-injection surface.
"""
from __future__ import annotations

import os
import sys


def reexec_in_place() -> None:
    """Re-exec the current process with its original argv.

    Uses `sys.orig_argv` (Python 3.10+) so flags the operator started
    with (``python -m runtime.chat.telegram.bot``,
    ``python scripts/telegram_smoke.py``, etc.) are preserved across
    the restart. Falls back to `sys.argv` prefixed with the current
    interpreter if `orig_argv` is unavailable.

    Calling this function does not return — `os.execv` replaces the
    process image. Only call from the top-level bot loop.
    """
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    os.execv(argv[0], argv)  # noqa: S606  # argv-only, no shell.


__all__ = ["reexec_in_place"]
