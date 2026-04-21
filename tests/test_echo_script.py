"""Unit tests for the echo subprocess entry point (scheduler path)."""
from __future__ import annotations

import pytest

from runtime.skills.scripts.echo import main

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
