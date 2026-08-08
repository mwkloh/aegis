"""Generic CLI tool harness.

Phase 8 §C3. Runs one invocation of a ``ToolSpec``-declared CLI, under
the same bounded contract the applier uses for ``git apply`` + ``make
test``. Never raises — every failure mode collapses to one of a small
set of verdicts so logging/telemetry stays uniform across tools.

Invariants:

* argv only. ``argv_template`` is a list of tokens; there is no shell
  interpolation path. Placeholder resolution uses ``str.format_map``,
  which fails closed on unknown names (→ ``argv_rejected``).
* Bounded time. ``spec.timeout_ms`` bounds runtime via
  ``asyncio.wait_for``; on expiry the subprocess is killed and the
  verdict is ``timeout``.
* Bounded output. The harness clips captured stdout to
  ``STDOUT_TAIL_BYTES`` for logging, keeping the tail (where failures
  usually surface). Schema validation, when requested, runs on the
  full (unclipped) output before clipping.
* No hidden net. ``network_requested=True`` combined with
  ``spec.allow_net=False`` short-circuits to ``host_denied`` before
  the process ever starts. Actual network sandboxing happens in the
  runner (out of scope for this module — we only gate intent).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, ValidationError

from runtime.skills.registry import ToolSpec

ToolVerdict = Literal[
    "verified",           # exit 0 (+ schema match if declared)
    "argv_rejected",      # unresolved placeholder, empty token, NUL byte
    "exit_nonzero",       # ran to completion, exit code != 0
    "timeout",            # bounded by spec.timeout_ms
    "schema_violation",   # stdout did not match spec.schema
    "host_denied",        # allow_net=False but network was requested
    "tool_error",         # in-process tool raised or returned error status
]

STDOUT_TAIL_BYTES = 32 * 1024  # 32 KB — matches the applier's order of magnitude


class SubprocessRunner(Protocol):
    """Run argv in cwd, return ``(exit_code, combined_output)``.

    Identical shape to ``runtime.chat.telegram.long_running.SubprocessRunner``
    so the production ``AsyncioSubprocessRunner`` can be injected here
    too. Combined stdout+stderr preserves ordering for diagnostics.
    """

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]: ...


@dataclass(frozen=True)
class ToolResult:
    """One tool invocation. Always returned — never raised.

    ``argv`` is the fully-resolved command (tuple so the result is
    hashable and can't be mutated by accident). ``stdout_tail`` is
    clipped to ``STDOUT_TAIL_BYTES``; ``error`` carries a one-line
    human-readable reason for every non-``verified`` verdict.
    """

    verdict: ToolVerdict
    argv: tuple[str, ...]
    stdout_tail: str
    exit_code: int | None
    error: str
    duration_ms: int


def _clip_tail(text: str, limit: int) -> str:
    """Keep the tail. Byte-accurate so non-ASCII output stays honest."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    # Leave headroom for the "… " marker.
    budget = max(limit - 3, 0)
    clipped = encoded[-budget:]
    return "… " + clipped.decode("utf-8", errors="replace")


def _resolve_argv(
    template: list[str], args: dict[str, Any]
) -> tuple[tuple[str, ...] | None, str]:
    """Resolve ``{name}`` placeholders via ``str.format_map``.

    Returns ``(argv, "")`` on success or ``(None, reason)`` on rejection.
    Missing keys, NUL bytes, and non-string arg values all fail closed.
    """
    # Reject non-string/number values at the boundary. Lists, dicts, None,
    # bools etc. would format into argv tokens in ways that surprise — and
    # callers should serialize before getting here.
    safe_args: dict[str, str] = {}
    for key, value in args.items():
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None, f"arg {key!r} has unsupported type {type(value).__name__}"
        safe_args[key] = str(value)

    resolved: list[str] = []
    for idx, token in enumerate(template):
        try:
            formatted = token.format_map(_StrictMap(safe_args))
        except KeyError as exc:
            return None, f"argv[{idx}]: unresolved placeholder {exc.args[0]!r}"
        except (IndexError, ValueError) as exc:
            return None, f"argv[{idx}]: format error: {exc}"
        if "\x00" in formatted:
            return None, f"argv[{idx}] contains NUL byte"
        if not formatted:
            return None, f"argv[{idx}] resolved to empty string"
        resolved.append(formatted)
    return tuple(resolved), ""


class _StrictMap(dict[str, str]):
    """Dict that raises ``KeyError`` instead of silently returning '' on miss.

    ``str.format_map`` already raises on missing keys for plain dicts —
    this subclass exists to keep intent explicit and to block any
    future temptation to use ``defaultdict`` here.
    """

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


async def run_tool(
    spec: ToolSpec,
    args: dict[str, Any],
    *,
    runner: SubprocessRunner,
    cwd: Path,
    network_requested: bool = False,
    clock: Any = None,
) -> ToolResult:
    """Run one CLI tool per ``spec``. Never raises.

    ``args`` provides the values for ``spec.argv_template`` placeholders.
    ``network_requested`` is the caller's signal that this invocation
    needs network; combined with ``spec.allow_net=False`` it yields
    ``host_denied``. ``clock`` is a ``time.monotonic``-compatible callable
    for deterministic duration_ms in tests.
    """
    now = clock if clock is not None else time.monotonic

    if network_requested and not spec.allow_net:
        return ToolResult(
            verdict="host_denied",
            argv=(),
            stdout_tail="",
            exit_code=None,
            error="network requested but spec.allow_net is False",
            duration_ms=0,
        )

    argv, reject_reason = _resolve_argv(list(spec.argv_template), args)
    if argv is None:
        return ToolResult(
            verdict="argv_rejected",
            argv=(),
            stdout_tail="",
            exit_code=None,
            error=reject_reason,
            duration_ms=0,
        )

    timeout_s = spec.timeout_ms / 1000.0
    started = now()
    try:
        exit_code, output = await asyncio.wait_for(
            runner.run(list(argv), cwd=cwd), timeout=timeout_s
        )
    except TimeoutError:
        duration_ms = int((now() - started) * 1000)
        return ToolResult(
            verdict="timeout",
            argv=argv,
            stdout_tail="",
            exit_code=None,
            error=f"exceeded timeout_ms={spec.timeout_ms}",
            duration_ms=duration_ms,
        )
    except (OSError, RuntimeError) as exc:
        # Runner crashed (bad argv, missing binary, event-loop state).
        # Collapse to argv_rejected — operator intent didn't reach the shell.
        duration_ms = int((now() - started) * 1000)
        return ToolResult(
            verdict="argv_rejected",
            argv=argv,
            stdout_tail="",
            exit_code=None,
            error=f"runner error: {exc}",
            duration_ms=duration_ms,
        )

    duration_ms = int((now() - started) * 1000)
    stdout_tail = _clip_tail(output, STDOUT_TAIL_BYTES)

    if exit_code != 0:
        return ToolResult(
            verdict="exit_nonzero",
            argv=argv,
            stdout_tail=stdout_tail,
            exit_code=exit_code,
            error=f"exit_code={exit_code}",
            duration_ms=duration_ms,
        )

    if spec.schema_ is not None:
        schema_error = _validate_schema(output, spec.schema_)
        if schema_error is not None:
            return ToolResult(
                verdict="schema_violation",
                argv=argv,
                stdout_tail=stdout_tail,
                exit_code=exit_code,
                error=schema_error,
                duration_ms=duration_ms,
            )

    return ToolResult(
        verdict="verified",
        argv=argv,
        stdout_tail=stdout_tail,
        exit_code=exit_code,
        error="",
        duration_ms=duration_ms,
    )


def _validate_schema(output: str, schema: dict[str, Any]) -> str | None:
    """Return a one-line error reason or ``None`` if output matches schema."""
    stripped = output.strip()
    if not stripped:
        return "stdout is empty; expected JSON"
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return f"stdout is not valid JSON: {exc.msg}"
    try:
        Draft202012Validator(schema).validate(data)
    except ValidationError as exc:
        path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
        return f"schema mismatch at {path}: {exc.message}"
    return None
