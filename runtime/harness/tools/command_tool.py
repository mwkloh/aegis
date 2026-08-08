"""Harness tool callable for argv-only command execution (run_command).

No shell, ever. `argv[0]` must be a bare binary name resolved via PATH and
present in `CommandsConfig.allowed_binaries` — the allowlist is the
containment boundary, not path sandboxing. See
docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md Track D item D3.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runtime.config import CommandsConfig

__all__ = ["make_command_tool"]


def make_command_tool(cfg: CommandsConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a synchronous `run_command` callable closed over `cfg`.

    Matches `HarnessAdapter.execute`'s sync contract. `subprocess.TimeoutExpired`
    is deliberately NOT caught here — it propagates to `HarnessAdapter.execute`'s
    catch-all, which turns it into `ToolResult(status="error")` (ledger verdict
    `tool_error`).
    """

    def run_command(args: dict[str, Any]) -> dict[str, Any]:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(t, str) for t in argv
        ):
            raise ValueError("argv must be a non-empty list of strings")
        binary = argv[0]
        if "/" in binary or "\\" in binary or binary != Path(binary).name:
            raise PermissionError(
                f"argv[0] must be a bare binary name, not a path: {binary!r}"
            )
        if binary not in cfg.allowed_binaries:
            raise PermissionError(f"binary not allowlisted: {binary!r}")
        proc = subprocess.run(          # noqa: S603 — argv list, shell=False
            argv,
            capture_output=True,
            timeout=cfg.timeout_ms / 1000,
            shell=False,
            text=False,
            check=False,
        )
        stdout_tail = proc.stdout[-cfg.max_output_bytes :].decode(
            "utf-8", errors="replace"
        )
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "verdict": "verified" if proc.returncode == 0 else "exit_nonzero",
        }

    return run_command
