"""Unit tests for runtime.harness.tools.command_tool.make_command_tool()."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from runtime.config import CommandsConfig
from runtime.files.client import FilesClient
from runtime.harness.tools.command_tool import make_command_tool

pytestmark = pytest.mark.unit


def test_allowlisted_binary_runs_and_verifies(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["ls", str(tmp_path)]})

    assert out["verdict"] == "verified"
    assert out["exit_code"] == 0
    assert "hello.txt" in out["stdout_tail"]
    assert out["argv"] == ["ls", str(tmp_path)]


def test_nonzero_exit_yields_exit_nonzero_verdict(tmp_path: Path) -> None:
    target = tmp_path / "haystack.txt"
    target.write_text("nothing to see here\n", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["grep", "nope", str(target)]})

    assert out["verdict"] == "exit_nonzero"
    assert out["exit_code"] != 0


def test_non_allowlisted_binary_raises_permission_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["echo", "hi"]})


def test_absolute_path_argv0_raises_permission_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["/bin/ls", "."]})


def test_relative_path_argv0_raises_permission_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["./ls", "."]})


def test_argv_not_a_list_raises_value_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": "ls -la"})


def test_argv_empty_list_raises_value_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": []})


def test_argv_non_string_items_raise_value_error(tmp_path: Path) -> None:
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(ValueError, match="argv must be a non-empty list of strings"):
        run_command({"argv": ["ls", 123]})


def test_output_truncated_to_max_output_bytes_keeps_tail(tmp_path: Path) -> None:
    # Deterministic, easily-sliceable content: 100,000 sequential digit chars.
    content = "".join(str(i % 10) for i in range(100_000))
    target = tmp_path / "big.txt"
    target.write_text(content, encoding="utf-8")
    cfg = CommandsConfig(max_output_bytes=1024)
    run_command = make_command_tool(cfg, FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["cat", str(target)]})

    assert len(out["stdout_tail"].encode("utf-8")) == 1024
    assert out["stdout_tail"] == content[-1024:]


def test_timeout_propagates_as_subprocess_timeout_expired(tmp_path: Path) -> None:
    # sleep is not in the default allowlist; adding it here is config, not a
    # code change — the tool must still enforce the timeout regardless of
    # which binary is allowlisted.
    cfg = CommandsConfig(allowed_binaries=("sleep",), timeout_ms=100)
    run_command = make_command_tool(cfg, FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(subprocess.TimeoutExpired):
        run_command({"argv": ["sleep", "2"]})


def test_invalid_utf8_output_decodes_with_replace(tmp_path: Path) -> None:
    target = tmp_path / "binary.bin"
    target.write_bytes(b"before \xff\xfe garbage after")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["cat", str(target)]})

    assert out["verdict"] == "verified"
    assert "before" in out["stdout_tail"]
    assert "after" in out["stdout_tail"]


# ---------------------------------------------------------------------------
# C2 — path-shaped argv[1:] tokens are sandboxed to the FilesClient's
# allowed_roots, the same containment files_read enforces. See
# runtime/harness/tools/command_tool.py module docstring.
# ---------------------------------------------------------------------------


def test_absolute_path_arg_outside_roots_raises_permission_error(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("nope", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[inside]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["cat", str(outside / "secret.txt")]})


def test_home_relative_path_arg_outside_roots_raises_permission_error(tmp_path: Path) -> None:
    # Mirrors the ~/.aegis/.env exfil vector: a `~`-prefixed token pointing
    # outside the configured roots must be denied, not resolved against the
    # bot process's real home directory.
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["cat", "~/.aegis/.env"]})


def test_path_arg_inside_root_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["cat", str(target)]})

    assert out["verdict"] == "verified"
    assert "hello" in out["stdout_tail"]


def test_dotdot_traversal_arg_raises_permission_error(tmp_path: Path) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[inside]))

    with pytest.raises(PermissionError):
        run_command({"argv": ["cat", str(inside / ".." / "secret.txt")]})


def test_flags_and_bare_patterns_are_not_treated_as_paths(tmp_path: Path) -> None:
    # `-r` (a flag) and `TOKEN` (a bare pattern with no path separator) must
    # not be run through path containment — only the in-root path argument
    # is validated. This documents the residual limitation: bare relative
    # tokens with no separator resolve against the process cwd, not the
    # FilesClient roots.
    target = tmp_path / "haystack.txt"
    target.write_text("TOKEN found here\n", encoding="utf-8")
    run_command = make_command_tool(CommandsConfig(), FilesClient(allowed_roots=[tmp_path]))

    out = run_command({"argv": ["grep", "-r", "TOKEN", str(target)]})

    assert out["verdict"] == "verified"
    assert "TOKEN" in out["stdout_tail"]
