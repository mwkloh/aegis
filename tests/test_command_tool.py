"""Unit tests for runtime.harness.tools.command_tool.make_command_tool()."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from runtime.config import CommandsConfig
from runtime.harness.tools.command_tool import make_command_tool

pytestmark = pytest.mark.unit


def test_allowlisted_binary_runs_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig())

    out = run_command({"argv": ["ls", str(tmp_path)]})

    assert out["verdict"] == "verified"
    assert out["exit_code"] == 0
    assert "hello.txt" in out["stdout_tail"]
    assert out["argv"] == ["ls", str(tmp_path)]


def test_nonzero_exit_yields_exit_nonzero_verdict(tmp_path: Path) -> None:
    target = tmp_path / "haystack.txt"
    target.write_text("nothing to see here\n", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig())

    out = run_command({"argv": ["grep", "nope", str(target)]})

    assert out["verdict"] == "exit_nonzero"
    assert out["exit_code"] != 0


def test_non_allowlisted_binary_raises_permission_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(PermissionError):
        run_command({"argv": ["echo", "hi"]})


def test_absolute_path_argv0_raises_permission_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(PermissionError):
        run_command({"argv": ["/bin/ls", "."]})


def test_relative_path_argv0_raises_permission_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(PermissionError):
        run_command({"argv": ["./ls", "."]})


def test_argv_not_a_list_raises_value_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": "ls -la"})


def test_argv_empty_list_raises_value_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": []})


def test_argv_non_string_items_raise_value_error() -> None:
    run_command = make_command_tool(CommandsConfig())

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": ["ls", 123]})


def test_output_truncated_to_max_output_bytes_keeps_tail(tmp_path: Path) -> None:
    # Deterministic, easily-sliceable content: 100,000 sequential digit chars.
    content = "".join(str(i % 10) for i in range(100_000))
    target = tmp_path / "big.txt"
    target.write_text(content, encoding="utf-8")
    cfg = CommandsConfig(max_output_bytes=1024)
    run_command = make_command_tool(cfg)

    out = run_command({"argv": ["cat", str(target)]})

    assert len(out["stdout_tail"].encode("utf-8")) == 1024
    assert out["stdout_tail"] == content[-1024:]


def test_timeout_propagates_as_subprocess_timeout_expired(tmp_path: Path) -> None:
    # sleep is not in the default allowlist; adding it here is config, not a
    # code change — the tool must still enforce the timeout regardless of
    # which binary is allowlisted.
    cfg = CommandsConfig(allowed_binaries=("sleep",), timeout_ms=100)
    run_command = make_command_tool(cfg)

    with pytest.raises(subprocess.TimeoutExpired):
        run_command({"argv": ["sleep", "2"]})


def test_invalid_utf8_output_decodes_with_replace(tmp_path: Path) -> None:
    target = tmp_path / "binary.bin"
    target.write_bytes(b"before \xff\xfe garbage after")
    run_command = make_command_tool(CommandsConfig())

    out = run_command({"argv": ["cat", str(target)]})

    assert out["verdict"] == "verified"
    assert "before" in out["stdout_tail"]
    assert "after" in out["stdout_tail"]
