from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from runtime.harness.contract import ToolResult as InProcessToolResult
from runtime.skills.registry import ToolSpec
from runtime.tools import STDOUT_TAIL_BYTES, run_tool
from runtime.tools.record import verdict_for_result

pytestmark = pytest.mark.unit


# --- fake subprocess runner --------------------------------------------------


@dataclass
class FakeRunner:
    """Records the argv/cwd it was called with, returns a scripted response.

    ``hang=True`` blocks forever so callers can drive the timeout path.
    Raising ``raises`` lets us exercise the runner-error branch.
    """

    exit_code: int = 0
    output: str = ""
    hang: bool = False
    raises: BaseException | None = None
    calls: list[tuple[list[str], Path]] = field(default_factory=list)

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]:
        self.calls.append((list(argv), cwd))
        if self.raises is not None:
            raise self.raises
        if self.hang:
            await asyncio.sleep(3600)
        return self.exit_code, self.output


class FakeClock:
    """Monotonic-compatible clock so duration_ms is deterministic in tests."""

    def __init__(self, starts: list[float]) -> None:
        self._values = list(starts)

    def __call__(self) -> float:
        return self._values.pop(0) if self._values else 0.0


def _spec(**overrides: Any) -> ToolSpec:
    base: dict[str, Any] = {
        "name": "search",
        "argv_template": ["aegis", "vault", "search", "--query", "{query}"],
        "timeout_ms": 5_000,
        "allow_net": False,
    }
    base.update(overrides)
    # ToolSpec uses `schema` as the YAML/input alias for the `schema_` field.
    return ToolSpec.model_validate(base)


# --- verdict: verified -------------------------------------------------------


def test_verified_no_schema(tmp_path: Path) -> None:
    runner = FakeRunner(exit_code=0, output="ok\n")
    spec = _spec()
    clock = FakeClock([10.0, 10.25])

    result = asyncio.run(
        run_tool(spec, {"query": "alpha"}, runner=runner, cwd=tmp_path, clock=clock)
    )

    assert result.verdict == "verified"
    assert result.argv == ("aegis", "vault", "search", "--query", "alpha")
    assert result.stdout_tail == "ok\n"
    assert result.exit_code == 0
    assert result.error == ""
    assert result.duration_ms == 250
    # Runner received the resolved argv (list, not template).
    assert runner.calls == [(["aegis", "vault", "search", "--query", "alpha"], tmp_path)]


def test_verified_with_matching_schema(tmp_path: Path) -> None:
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    runner = FakeRunner(exit_code=0, output='{"ok": true}\n')
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "verified"
    assert result.error == ""


# --- verdict: argv_rejected --------------------------------------------------


def test_argv_rejected_on_missing_placeholder(tmp_path: Path) -> None:
    runner = FakeRunner()
    spec = _spec(argv_template=["aegis", "{unknown}"])

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "argv_rejected"
    assert "unknown" in result.error
    assert result.argv == ()
    # Runner was NOT invoked — argv resolution failed before spawn.
    assert runner.calls == []


def test_argv_rejected_on_nul_byte(tmp_path: Path) -> None:
    runner = FakeRunner()
    spec = _spec()

    result = asyncio.run(
        run_tool(spec, {"query": "bad\x00value"}, runner=runner, cwd=tmp_path)
    )

    assert result.verdict == "argv_rejected"
    assert "NUL" in result.error
    assert runner.calls == []


def test_argv_rejected_on_unsupported_arg_type(tmp_path: Path) -> None:
    runner = FakeRunner()
    spec = _spec()

    result = asyncio.run(
        run_tool(spec, {"query": ["not", "a", "string"]}, runner=runner, cwd=tmp_path)
    )

    assert result.verdict == "argv_rejected"
    assert "unsupported type" in result.error
    assert runner.calls == []


def test_argv_rejected_rejects_boolean(tmp_path: Path) -> None:
    # Bools would stringify as "True"/"False" which almost always surprises
    # the downstream CLI — fail closed.
    runner = FakeRunner()
    spec = _spec()

    result = asyncio.run(
        run_tool(spec, {"query": True}, runner=runner, cwd=tmp_path)
    )

    assert result.verdict == "argv_rejected"


def test_argv_rejected_on_runner_oserror(tmp_path: Path) -> None:
    runner = FakeRunner(raises=FileNotFoundError("aegis not installed"))
    spec = _spec()

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "argv_rejected"
    assert "aegis not installed" in result.error
    # argv WAS resolved; runner just couldn't execute it.
    assert result.argv == ("aegis", "vault", "search", "--query", "q")


# --- verdict: exit_nonzero ---------------------------------------------------


def test_exit_nonzero_captures_exit_code_and_output(tmp_path: Path) -> None:
    runner = FakeRunner(exit_code=2, output="boom\n")
    spec = _spec()

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "exit_nonzero"
    assert result.exit_code == 2
    assert result.stdout_tail == "boom\n"
    assert "exit_code=2" in result.error


def test_exit_nonzero_bypasses_schema_validation(tmp_path: Path) -> None:
    # Even if the output happens to be valid JSON, a non-zero exit wins.
    schema = {"type": "object"}
    runner = FakeRunner(exit_code=1, output='{"ok": true}\n')
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "exit_nonzero"


# --- verdict: timeout --------------------------------------------------------


def test_timeout_kills_when_runner_hangs(tmp_path: Path) -> None:
    runner = FakeRunner(hang=True)
    spec = _spec(timeout_ms=100)  # 100ms — minimum floor

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "timeout"
    assert "timeout_ms=100" in result.error
    assert result.exit_code is None
    assert result.argv == ("aegis", "vault", "search", "--query", "q")


# --- verdict: schema_violation ----------------------------------------------


def test_schema_violation_on_invalid_json(tmp_path: Path) -> None:
    schema = {"type": "object"}
    runner = FakeRunner(exit_code=0, output="not json at all")
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "schema_violation"
    assert "not valid JSON" in result.error


def test_schema_violation_on_type_mismatch(tmp_path: Path) -> None:
    schema = {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer"}}}
    runner = FakeRunner(exit_code=0, output=json.dumps({"count": "oops"}))
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "schema_violation"
    assert "count" in result.error


def test_schema_violation_on_empty_stdout(tmp_path: Path) -> None:
    schema = {"type": "object"}
    runner = FakeRunner(exit_code=0, output="   \n  ")
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "schema_violation"
    assert "empty" in result.error


def test_schema_validates_on_full_output_not_clipped_tail(tmp_path: Path) -> None:
    # Schema validation must see the complete JSON payload even when
    # captured stdout exceeds the clip budget. Construct a valid JSON
    # body that's >> STDOUT_TAIL_BYTES so clipping would corrupt it.
    payload = {"blob": "x" * (STDOUT_TAIL_BYTES * 2)}
    full = json.dumps(payload)
    schema = {"type": "object", "required": ["blob"], "properties": {"blob": {"type": "string"}}}
    runner = FakeRunner(exit_code=0, output=full)
    spec = _spec(schema=schema)

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "verified"
    # stdout_tail WAS clipped for logging even though validation used full output.
    assert len(result.stdout_tail.encode("utf-8")) <= STDOUT_TAIL_BYTES + 4


# --- verdict: host_denied ----------------------------------------------------


def test_host_denied_when_network_requested_but_disallowed(tmp_path: Path) -> None:
    runner = FakeRunner()
    spec = _spec(allow_net=False)

    result = asyncio.run(
        run_tool(
            spec,
            {"query": "q"},
            runner=runner,
            cwd=tmp_path,
            network_requested=True,
        )
    )

    assert result.verdict == "host_denied"
    assert "allow_net" in result.error
    # Hard preflight — runner never invoked.
    assert runner.calls == []


def test_host_allowed_when_spec_permits(tmp_path: Path) -> None:
    runner = FakeRunner(exit_code=0, output="ok")
    spec = _spec(allow_net=True)

    result = asyncio.run(
        run_tool(
            spec,
            {"query": "q"},
            runner=runner,
            cwd=tmp_path,
            network_requested=True,
        )
    )

    assert result.verdict == "verified"


# --- stdout tail clipping ----------------------------------------------------


def test_stdout_tail_clipped_to_budget(tmp_path: Path) -> None:
    blob = "A" * (STDOUT_TAIL_BYTES * 3)
    runner = FakeRunner(exit_code=0, output=blob)
    spec = _spec()

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "verified"
    assert len(result.stdout_tail.encode("utf-8")) <= STDOUT_TAIL_BYTES + 4
    assert result.stdout_tail.startswith("… ")
    # Tail preserved (ends with A's, not cut from the middle).
    assert result.stdout_tail.endswith("A")


def test_stdout_tail_handles_unicode_boundary(tmp_path: Path) -> None:
    # Clip budget hits inside a multi-byte codepoint — must not raise,
    # and must not emit mojibake (errors="replace").
    emoji = "🔥"  # 4 bytes in UTF-8
    blob = emoji * (STDOUT_TAIL_BYTES // 4 + 100)  # overshoots the budget
    runner = FakeRunner(exit_code=0, output=blob)
    spec = _spec()

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    assert result.verdict == "verified"
    assert len(result.stdout_tail.encode("utf-8")) <= STDOUT_TAIL_BYTES + 4


# --- duration and immutability ----------------------------------------------


def test_duration_ms_captured_from_clock(tmp_path: Path) -> None:
    runner = FakeRunner(exit_code=0, output="")
    spec = _spec()
    # 1.25 seconds between start and finish.
    clock = FakeClock([100.0, 101.25])

    result = asyncio.run(
        run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path, clock=clock)
    )

    assert result.duration_ms == 1250


def test_result_is_frozen(tmp_path: Path) -> None:
    runner = FakeRunner(exit_code=0, output="")
    spec = _spec()

    result = asyncio.run(run_tool(spec, {"query": "q"}, runner=runner, cwd=tmp_path))

    with pytest.raises((AttributeError, Exception)):
        result.verdict = "exit_nonzero"  # type: ignore[misc]


# --- verdict_for_result (B1) --------------------------------------------------


def test_verdict_for_result_ok_maps_to_verified() -> None:
    result = InProcessToolResult(status="ok", payload={"entries": ["a"]})
    assert verdict_for_result(result) == "verified"


def test_verdict_for_result_error_maps_to_tool_error() -> None:
    result = InProcessToolResult(status="error", error="denied")
    assert verdict_for_result(result) == "tool_error"
