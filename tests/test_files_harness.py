# tests/test_files_harness.py
"""Unit tests for make_files_tools harness callables."""
from __future__ import annotations

from typing import Any

import pytest

from runtime.files.client import DirEntry, FileInfo, FileTooBig, PathDenied
from runtime.harness.tools.files_tool import make_files_tools

pytestmark = pytest.mark.unit

_NOW = "2026-04-22T00:00:00+00:00"


class _StubClient:
    def __init__(self) -> None:
        self.entries: list[DirEntry] = []
        self.content: str = "content"
        self.file_info = FileInfo(
            path="/x/doc.txt", type="file", size=7,
            created=_NOW, modified=_NOW, accessed=_NOW, mode="100644",
        )
        self.search_results: list[str] = []
        self.write_result: dict[str, Any] = {"path": "/x/written.txt", "bytes_written": 7}
        self.raise_on_next: Exception | None = None

    def _maybe_raise(self) -> None:
        if self.raise_on_next is not None:
            exc = self.raise_on_next
            self.raise_on_next = None
            raise exc

    def list_dir(self, path: str, *, recursive: bool = False) -> list[DirEntry]:
        self._maybe_raise()
        return self.entries

    def read_file(self, path: str) -> str:
        self._maybe_raise()
        return self.content

    def stat(self, path: str) -> FileInfo:
        self._maybe_raise()
        return self.file_info

    def search(self, directory: str, pattern: str, *, kind: str = "any") -> list[str]:
        self._maybe_raise()
        return self.search_results

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        self._maybe_raise()
        return self.write_result


@pytest.fixture()
def stub() -> _StubClient:
    return _StubClient()


@pytest.fixture()
def tools(stub: _StubClient) -> dict[str, Any]:
    return make_files_tools(stub)  # type: ignore[arg-type]


def test_files_list_returns_formatted_entries(tools: dict, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="a.txt", path="/x/a.txt", type="file", size=10, modified=_NOW)]
    result = tools["files_list"]({"path": "/x"})
    assert "[f]" in result["result"]
    assert "a.txt" in result["result"]


def test_files_list_empty_directory(tools: dict) -> None:
    result = tools["files_list"]({"path": "/x"})
    assert "empty" in result["result"].lower()


def test_files_read_returns_content(tools: dict, stub: _StubClient) -> None:
    stub.content = "hello"
    result = tools["files_read"]({"path": "/x/doc.txt"})
    assert result["result"] == "hello"


def test_files_read_truncates_long_content(tools: dict, stub: _StubClient) -> None:
    stub.content = "x" * 5000
    result = tools["files_read"]({"path": "/x/doc.txt"})
    assert "truncated" in result["result"]
    assert len(result["result"]) <= 3514  # 3500 chars + len("… (truncated)")


def test_files_stat_returns_key_value(tools: dict) -> None:
    result = tools["files_stat"]({"path": "/x/doc.txt"})
    assert "type" in result["result"]
    assert "size" in result["result"]


def test_files_search_returns_paths(tools: dict, stub: _StubClient) -> None:
    stub.search_results = ["/x/report.pdf"]
    result = tools["files_search"]({"directory": "/x", "pattern": "*.pdf"})
    assert "/x/report.pdf" in result["result"]


def test_files_search_no_matches(tools: dict) -> None:
    result = tools["files_search"]({"directory": "/x", "pattern": "*.pdf"})
    assert "No matches" in result["result"]


def test_files_write_returns_confirmation(tools: dict, stub: _StubClient) -> None:
    stub.write_result = {"path": "/x/notes.txt", "bytes_written": 42}
    result = tools["files_write"]({"path": "/x/notes.txt", "content": "hello"})
    assert "42" in result["result"]
    assert "/x/notes.txt" in result["result"]


def test_files_write_path_denied_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    with pytest.raises(RuntimeError, match="outside roots"):
        tools["files_write"]({"path": "/etc/passwd", "content": "pwned"})


def test_files_write_file_too_big_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("too big")
    with pytest.raises(RuntimeError, match="too big"):
        tools["files_write"]({"path": "/x/huge.bin", "content": "x" * 10})


def test_path_denied_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    with pytest.raises(RuntimeError, match="outside roots"):
        tools["files_read"]({"path": "/etc/passwd"})


def test_file_too_big_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("too big")
    with pytest.raises(RuntimeError, match="too big"):
        tools["files_read"]({"path": "/x/huge.bin"})


def test_oserror_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = OSError("disk error")
    with pytest.raises(RuntimeError, match="disk error"):
        tools["files_list"]({"path": "/x"})


def test_make_files_tools_returns_six_callables(tools: dict) -> None:
    assert set(tools.keys()) == {
        "files_list",
        "files_read",
        "files_stat",
        "files_search",
        "files_open",
        "files_write",
    }
