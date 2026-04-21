"""Phase 7 §4.2 — long-running operator slashes.

`/apply` and `/harness` don't fit the sync `Handler = (msg, cmd) -> str`
contract because their work is a subprocess that can take seconds to
minutes and the plan wants progress reported via *edited* messages,
not new ones. So we split them out of the `Dispatcher` and route them
through this module instead.

Design pins (from `docs/PLAN_PHASE_7_TELEGRAM.md` §4.2):

* **One command in flight per `chat_id`.** A second `/apply` while
  the first is still running replies with a "wait for it to finish"
  note — no queueing, no preemption. Enforced by `InFlightRegistry`.
* **Streamed via edited messages.** Version 1 sends one initial
  "Running …" reply, captures the returned message handle, and
  edits it once on completion with the subprocess tail. Per-line
  progress streaming is a future enhancement — keeping v1 to a
  single final edit avoids tangling with Telegram's ~1 edit/sec
  rate limit.
* **Subprocess is injectable.** A `SubprocessRunner` Protocol is
  the only seam this module exposes; the production implementation
  lives in `bot.py` beside the real Telegram SDK wiring, so this
  module stays pure-Python and testable without an event loop that
  spawns real processes.
* **Never raises.** Usage errors, validation failures, and runner
  exceptions all render a human-readable reply — consistent with
  the rest of the Telegram surface.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Protocol

from runtime.chat.telegram.dispatch import ParsedCommand

MAX_LONG_RUNNING_CHARS = 3500  # headroom below Telegram's 4096-char cap


class _Editable(Protocol):
    """Duck-typed Telegram Message returned from `reply_text` — needs `edit_text`."""

    async def edit_text(self, text: str) -> Any: ...


class _Replyable(Protocol):
    """Duck-typed Telegram Message — `reply_text` MUST return an `_Editable`."""

    async def reply_text(self, text: str) -> _Editable: ...


class SubprocessRunner(Protocol):
    """Run argv in cwd, return `(exit_code, combined_output)`.

    Implementations must combine stdout + stderr so the operator sees
    the subprocess tail in the order the process emitted it. No
    partial results — the whole output is returned at once.
    """

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]: ...


class InFlightRegistry:
    """Per-chat in-flight command guard.

    The bot runs on a single event loop so the dict is not racy under
    cooperative scheduling — no `asyncio.Lock` required. Acquire
    returns True only if no other command is currently in flight for
    that `chat_id`; callers MUST pair every successful acquire with a
    `release` (use `try/finally`).
    """

    def __init__(self) -> None:
        self._in_flight: dict[int, str] = {}

    def try_acquire(self, chat_id: int, command: str) -> bool:
        if chat_id in self._in_flight:
            return False
        self._in_flight[chat_id] = command
        return True

    def release(self, chat_id: int) -> None:
        self._in_flight.pop(chat_id, None)

    def current(self, chat_id: int) -> str | None:
        return self._in_flight.get(chat_id)

    def any_in_flight(self) -> bool:
        return bool(self._in_flight)


_CT_PATTERN = re.compile(r"^ct-\d+$")


def _normalize_ct_id(raw: str) -> str | None:
    """Return canonical `CT-<digits>` (uppercase prefix) or None."""
    candidate = raw.strip().lower()
    if not _CT_PATTERN.match(candidate):
        return None
    return "CT-" + candidate.split("-", 1)[1]


def _clip(text: str, limit: int) -> str:
    """Keep the tail — subprocess failures almost always surface near EOF."""
    if len(text) <= limit:
        return text
    return "… " + text[-(limit - 2) :]


class LongRunningRunner:
    """Coordinator for `/apply` and `/harness`.

    Stateless per call except for `InFlightRegistry`. Safe to share
    across handlers; the registry is the only mutable state and its
    per-chat invariants hold as long as every branch of `run` pairs
    `try_acquire` with `release` under a `try/finally`.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        runner: SubprocessRunner,
        registry: InFlightRegistry | None = None,
        python_executable: str | None = None,
        vault_root: Path | None = None,
    ) -> None:
        self._workspace = workspace
        self._runner = runner
        self._registry = registry if registry is not None else InFlightRegistry()
        self._python = (
            python_executable if python_executable is not None else sys.executable
        )
        self._vault_root = vault_root

    @property
    def commands(self) -> frozenset[str]:
        return frozenset({"/apply", "/harness", "/brief"})

    @property
    def registry(self) -> InFlightRegistry:
        return self._registry

    async def run(
        self,
        *,
        chat_id: int,
        cmd: ParsedCommand,
        message: _Replyable,
    ) -> None:
        """Dispatch one long-running slash. Never raises."""
        if cmd.name not in self.commands:
            await message.reply_text(f"unknown long-running command {cmd.name!r}")
            return
        if not self._registry.try_acquire(chat_id, cmd.name):
            current = self._registry.current(chat_id) or "another command"
            await message.reply_text(
                f"Already running {current} in this chat. "
                "Wait for it to finish before starting another."
            )
            return
        try:
            if cmd.name == "/apply":
                await self._run_apply(cmd=cmd, message=message)
            elif cmd.name == "/brief":
                await self._run_brief(message=message)
            else:  # "/harness"
                await self._run_harness(cmd=cmd, message=message)
        finally:
            self._registry.release(chat_id)

    async def _run_apply(
        self, *, cmd: ParsedCommand, message: _Replyable
    ) -> None:
        if not cmd.args:
            await message.reply_text(
                "Usage: /apply CT-NNN [--dry-run|--no-tests|--status]"
            )
            return
        ct = _normalize_ct_id(cmd.args[0])
        if ct is None:
            await message.reply_text(
                f"/apply expects a CT-NNN id, got {cmd.args[0]!r}."
            )
            return
        extra = [a for a in cmd.args[1:] if a]
        argv = [
            self._python,
            "-m",
            "runtime.coding_harness.apply_cli",
            ct,
            *extra,
        ]
        sent = await message.reply_text(f"Running /apply {ct}...")
        exit_code, output = await self._runner.run(argv, cwd=self._workspace)
        status = "succeeded" if exit_code == 0 else f"failed (exit={exit_code})"
        body = f"/apply {ct} {status}"
        if output:
            body = f"{body}\n\n{output}"
        await sent.edit_text(_clip(body, MAX_LONG_RUNNING_CHARS))

    async def _run_brief(self, *, message: _Replyable) -> None:
        # Wraps the `morning_brief` skill script so the operator gets
        # the full markdown brief posted back into chat. `/brief` takes
        # no args — the vault root is resolved from configuration at
        # wire-time (see `build_long_running_runner`). If the config
        # never set `vault_indexing.vault_root`, we fail fast with a
        # human-readable reply instead of exploding under subprocess.
        if self._vault_root is None:
            await message.reply_text(
                "/brief is not configured — set vaultIndexing.vaultRoot "
                "in config.json to enable."
            )
            return
        argv = [
            self._python,
            "-m",
            "runtime.skills.scripts.morning_brief",
            "--vault-root",
            str(self._vault_root),
        ]
        sent = await message.reply_text("Running /brief...")
        exit_code, output = await self._runner.run(argv, cwd=self._workspace)
        if exit_code == 0:
            body = output or "/brief succeeded (empty output)"
        else:
            status = f"failed (exit={exit_code})"
            body = f"/brief {status}"
            if output:
                body = f"{body}\n\n{output}"
        await sent.edit_text(_clip(body, MAX_LONG_RUNNING_CHARS))

    async def run_skill(
        self,
        *,
        chat_id: int,
        skill_id: str,
        argv: list[str],
        message: _Replyable,
        echo_output: bool = True,
    ) -> None:
        """Dispatch a declarative skill's resolved argv under the in-flight guard.

        Called from ``route_chat`` when the deterministic intent router
        short-circuits a free-form message to a known skill. ``argv``
        must be fully resolved by the caller (no ``{placeholder}``
        tokens) — this runner intentionally does no substitution so
        the seam stays argv-only.

        ``echo_output=True`` mirrors ``/brief``: on success we post
        the subprocess stdout verbatim (the operator wanted the brief,
        not a banner). Set to False for skills where a status line +
        output tail is more appropriate.

        Never raises; uses ``chat_id`` as the in-flight guard key so
        a concurrent free-form intent in the same chat gets the usual
        "already running …" reply instead of racing.
        """
        if not self._registry.try_acquire(chat_id, skill_id):
            current = self._registry.current(chat_id) or "another command"
            await message.reply_text(
                f"Already running {current} in this chat. "
                "Wait for it to finish before starting another."
            )
            return
        try:
            sent = await message.reply_text(f"Running {skill_id}...")
            exit_code, output = await self._runner.run(argv, cwd=self._workspace)
            if exit_code == 0:
                if echo_output:
                    body = output or f"{skill_id} succeeded (empty output)"
                else:
                    body = f"{skill_id} succeeded"
                    if output:
                        body = f"{body}\n\n{output}"
            else:
                body = f"{skill_id} failed (exit={exit_code})"
                if output:
                    body = f"{body}\n\n{output}"
            await sent.edit_text(_clip(body, MAX_LONG_RUNNING_CHARS))
        finally:
            self._registry.release(chat_id)

    async def _run_harness(
        self, *, cmd: ParsedCommand, message: _Replyable
    ) -> None:
        # `/harness` uses flag-based task selection (`--task CT-NNN`),
        # not a positional id. Pass args through untouched — the CLI
        # validates them.
        extra = [a for a in cmd.args if a]
        argv = [self._python, "-m", "runtime.coding_harness.cli", *extra]
        header = "/harness" + ((" " + " ".join(extra)) if extra else "")
        sent = await message.reply_text(f"Running {header}...")
        exit_code, output = await self._runner.run(argv, cwd=self._workspace)
        status = "succeeded" if exit_code == 0 else f"failed (exit={exit_code})"
        body = f"{header} {status}"
        if output:
            body = f"{body}\n\n{output}"
        await sent.edit_text(_clip(body, MAX_LONG_RUNNING_CHARS))


__all__ = [
    "MAX_LONG_RUNNING_CHARS",
    "InFlightRegistry",
    "LongRunningRunner",
    "SubprocessRunner",
]
