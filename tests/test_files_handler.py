# tests/test_files_handler.py
"""Unit tests for the /files slash handler."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.chat.telegram.files_handler import files_handler
from runtime.files.client import DirEntry, FileInfo, FileTooBig, PathDenied

pytestmark = pytest.mark.unit

# ── Stub client ───────────────────────────────────────────────────────────────

_NOW = "2026-04-22T00:00:00+00:00"


class _StubClient:
    """Pre-configured fake FilesClient."""

    def __init__(self) -> None:
        self.entries: list[DirEntry] = []
        self.content: str = "file content"
        self.file_info: FileInfo = FileInfo(
            path="/root/doc.txt",
            type="file",
            size=42,
            created=_NOW,
            modified=_NOW,
            accessed=_NOW,
            mode="100644",
        )
        self.search_results: list[str] = []
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

    def move(self, src: str, dst: str) -> None:
        self._maybe_raise()

    def copy(self, src: str, dst: str) -> None:
        self._maybe_raise()

    def delete(self, path: str, *, confirm: bool = False) -> None:
        self._maybe_raise()

    def mkdir(self, path: str) -> None:
        self._maybe_raise()

    def open_with_app(self, path: str, app: str | None = None) -> None:
        self._maybe_raise()


def _msg(chat_id: int = 1) -> IncomingMessage:
    return IncomingMessage(chat_id=chat_id, user_id=1, text="")


def _cmd(name: str, *args: str) -> ParsedCommand:
    return ParsedCommand(name=name, args=tuple(args))


@pytest.fixture()
def stub() -> _StubClient:
    return _StubClient()


@pytest.fixture()
def handler(stub: _StubClient) -> Handler:
    return files_handler(client=stub)  # type: ignore[arg-type]


# ── No args ───────────────────────────────────────────────────────────────────

def test_no_args_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files"))
    assert "Usage" in result


def test_unknown_subverb_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "bogus"))
    assert "Usage" in result


# ── ls ────────────────────────────────────────────────────────────────────────

def test_ls_formats_file_entry(handler: Handler, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="report.pdf", path="/x/report.pdf", type="file", size=2048, modified=_NOW)]
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "[f]" in result
    assert "report.pdf" in result


def test_ls_formats_directory_entry(handler: Handler, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="docs", path="/x/docs", type="directory", size=0, modified=_NOW)]
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "[d]" in result
    assert "docs" in result


def test_ls_empty_directory(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "empty" in result.lower()


def test_ls_missing_path_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "ls"))
    assert "Usage" in result


# ── read ──────────────────────────────────────────────────────────────────────

def test_read_returns_content(handler: Handler, stub: _StubClient) -> None:
    stub.content = "hello world"
    result = handler(_msg(), _cmd("/files", "read", "/x/file.txt"))
    assert "hello world" in result


def test_read_clips_long_content(handler: Handler, stub: _StubClient) -> None:
    stub.content = "x" * 5000
    result = handler(_msg(), _cmd("/files", "read", "/x/file.txt"))
    assert "truncated" in result
    assert len(result) < 5000


def test_read_missing_path_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "read"))
    assert "Usage" in result


# ── stat ──────────────────────────────────────────────────────────────────────

def test_stat_shows_all_fields(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "stat", "/x/doc.txt"))
    assert "type" in result
    assert "size" in result
    assert "modified" in result
    assert "mode" in result


# ── find ──────────────────────────────────────────────────────────────────────

def test_find_no_matches(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "find", "/x", "*.pdf"))
    assert "No matches" in result


def test_find_shows_paths(handler: Handler, stub: _StubClient) -> None:
    stub.search_results = ["/x/report.pdf"]
    result = handler(_msg(), _cmd("/files", "find", "/x", "*.pdf"))
    assert "/x/report.pdf" in result


def test_find_missing_args_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "find", "/x"))
    assert "Usage" in result


# ── mv ────────────────────────────────────────────────────────────────────────

def test_mv_success_message(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "mv", "/x/a.txt", "/x/b.txt"))
    assert "Moved" in result


def test_mv_missing_dst_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "mv", "/x/a.txt"))
    assert "Usage" in result


# ── cp ────────────────────────────────────────────────────────────────────────

def test_cp_success_message(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "cp", "/x/a.txt", "/x/b.txt"))
    assert "Copied" in result


# ── rm ────────────────────────────────────────────────────────────────────────

def test_rm_without_confirm_flag(handler: Handler, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("non-empty, use --confirm")
    result = handler(_msg(), _cmd("/files", "rm", "/x/dir"))
    assert "Access denied" in result


def test_rm_with_confirm_flag(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "rm", "--confirm", "/x/dir"))
    assert "Deleted" in result


def test_rm_missing_path_returns_usage(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "rm"))
    assert "Usage" in result


# ── mkdir ─────────────────────────────────────────────────────────────────────

def test_mkdir_success(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "mkdir", "/x/newdir"))
    assert "Created" in result


# ── open ──────────────────────────────────────────────────────────────────────

def test_open_success(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "open", "/x/doc.txt"))
    assert "Opened" in result


def test_open_with_named_app(handler: Handler) -> None:
    result = handler(_msg(), _cmd("/files", "open", "/x/doc.txt", "Preview"))
    assert "Preview" in result


# ── Error paths ───────────────────────────────────────────────────────────────

def test_path_denied_shows_access_denied(handler: Handler, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    result = handler(_msg(), _cmd("/files", "read", "/etc/passwd"))
    assert "Access denied" in result


def test_file_too_big_shows_error(handler: Handler, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("10 MB exceeded")
    result = handler(_msg(), _cmd("/files", "read", "/x/huge.bin"))
    assert "too large" in result.lower()


def test_os_error_shows_error(handler: Handler, stub: _StubClient) -> None:
    stub.raise_on_next = OSError("permission denied")
    result = handler(_msg(), _cmd("/files", "read", "/x/locked.txt"))
    assert "Error" in result
