"""Unit tests for runtime.harness.tools.files_tool.make_files_tools()."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.files.client import FilesClient
from runtime.harness.tools.files_tool import make_files_tools


@pytest.fixture
def client(tmp_path: Path) -> FilesClient:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    return FilesClient(allowed_roots=[tmp_path])


def test_files_open_invokes_open_with_app(monkeypatch, client: FilesClient, tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_open(path: str, app: str | None = None) -> None:
        calls.append((path, app))

    monkeypatch.setattr(client, "open_with_app", fake_open)
    tools = make_files_tools(client)
    assert "files_open" in tools

    out = tools["files_open"]({"path": str(tmp_path / "hello.txt")})

    assert calls == [(str(tmp_path / "hello.txt"), None)]
    assert "Opened" in out["result"]


def test_files_open_passes_app_arg(monkeypatch, client: FilesClient, tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(client, "open_with_app", lambda p, app=None: calls.append((p, app)))

    tools = make_files_tools(client)
    tools["files_open"]({"path": str(tmp_path / "hello.txt"), "app": "Preview"})

    assert calls == [(str(tmp_path / "hello.txt"), "Preview")]
