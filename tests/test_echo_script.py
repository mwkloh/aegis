"""Unit tests for the echo subprocess entry point (scheduler path)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "runtime" / "skills" / "_bundle" / "echo" / "echo.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("echo_bundle", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_echo = _load_module()
main = _echo.main

pytestmark = pytest.mark.unit


def test_echo_with_args(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["hello", "world"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "hello world"


def test_echo_no_args_prints_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "echo"


def test_echo_single_word(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["ping"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "ping"
