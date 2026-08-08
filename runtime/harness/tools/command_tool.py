"""Harness tool callable for argv-only command execution (run_command).

No shell, ever. `argv[0]` must be a bare binary name resolved via PATH and
present in `CommandsConfig.allowed_binaries` — the allowlist is the
containment boundary for argv[0]. See
docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md Track D item D3.

`argv[1:]` path-shaped tokens are additionally sandboxed to
`FilesConfig.allowed_roots` via the same containment `FilesClient._validate`
already enforces for files_read/files_write (Phase 11 whole-branch review,
C2) — otherwise `cat ~/.aegis/.env` would exfiltrate the bot's own secrets
regardless of the files sandbox.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runtime.config import CommandsConfig
from runtime.files.client import FilesClient, PathDenied

__all__ = ["make_command_tool"]


def _is_path_shaped(token: str) -> bool:
    """A token that looks like it names a filesystem path.

    Absolute (`/…`), home-relative (`~…`), explicitly relative (`./…`,
    `../…`), or containing any path separator at all. Bare tokens with no
    separator (flags like `-r`, patterns like `TOKEN`, bare relative names
    like `notes.txt`) are NOT path-shaped — see the residual-limitation
    note on `run_command` below.
    """
    return token.startswith(("/", "~", "./", "../")) or os.sep in token


def _has_dotdot_segment(token: str) -> bool:
    """True if any `/`-delimited segment of `token` is exactly `..`."""
    return ".." in token.split("/")


def _check_path_arg(token: str, files_client: FilesClient) -> None:
    """Validate one path-shaped argv token against `files_client`'s roots.

    Raises `PermissionError` (never `PathDenied` — callers of `run_command`
    only need to catch one exception type) if the token traverses via `..`
    or resolves outside `files_client`'s allowed roots.
    """
    if not _is_path_shaped(token):
        return
    if _has_dotdot_segment(token):
        # Defense-in-depth: reject outright even if the resolved path would
        # happen to stay in-root.
        raise PermissionError(f"path argument contains a '..' segment: {token!r}")
    try:
        # Reuse FilesClient's containment check rather than reimplementing
        # it — same expanduser+resolve+within-roots logic files_read uses.
        files_client._validate(token)
    except PathDenied as exc:
        raise PermissionError(str(exc)) from exc


def make_command_tool(
    cfg: CommandsConfig, files_client: FilesClient
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a synchronous `run_command` callable closed over `cfg`.

    Matches `HarnessAdapter.execute`'s sync contract. `subprocess.TimeoutExpired`
    is deliberately NOT caught here — it propagates to `HarnessAdapter.execute`'s
    catch-all, which turns it into `ToolResult(status="error")` (ledger verdict
    `tool_error`).

    `files_client` sandboxes path-shaped `argv[1:]` tokens (see module
    docstring). Known residual limitation: a BARE token with no path
    separator (e.g. a relative filename like `notes.txt`, or a grep
    pattern) is not validated — it resolves against the process's cwd, not
    `files_client`'s roots. This is an accepted, deliberately narrow gap;
    the dangerous vectors (absolute `/…`, home `~…`, and `..` traversal)
    are all closed.
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
        for token in argv[1:]:
            _check_path_arg(token, files_client)
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
