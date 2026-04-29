# tests/test_files_client.py
"""FilesClient — path-sandboxed filesystem operations."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.files.client import (
    MAX_READ_BYTES,
    DirEntry,
    FileInfo,
    FilesClient,
    FileTooBig,
    PathDenied,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def client(tmp_path: Path) -> FilesClient:
    return FilesClient(allowed_roots=[tmp_path])


# ── Construction ──────────────────────────────────────────────────────────────

def test_empty_allowed_roots_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FilesClient(allowed_roots=[])


# ── Path validation ───────────────────────────────────────────────────────────

def test_validate_allows_path_inside_root(client: FilesClient, tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "file.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hi")
    # Should not raise
    client._validate(str(target))


def test_validate_denies_path_outside_root(client: FilesClient) -> None:
    with pytest.raises(PathDenied):
        client._validate("/etc/passwd")


def test_validate_denies_traversal_attack(client: FilesClient, tmp_path: Path) -> None:
    evil = str(tmp_path) + "/../../etc/passwd"
    with pytest.raises(PathDenied):
        client._validate(evil)


def test_validate_denies_bare_filename(client: FilesClient) -> None:
    with pytest.raises(PathDenied, match="absolute"):
        client._validate("ava-selfie.png")


def test_validate_denies_relative_path(client: FilesClient) -> None:
    with pytest.raises(PathDenied, match="absolute"):
        client._validate("Desktop/ava-selfie.png")


def test_validate_denies_dotdot_relative(client: FilesClient) -> None:
    with pytest.raises(PathDenied, match="absolute"):
        client._validate("../ava-selfie.png")


def test_validate_allows_tilde(client: FilesClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect ~ to tmp_path so the resolved path lands inside an allowed root.
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "note.txt").write_text("ok")
    client._validate("~/note.txt")


# ── list_dir ─────────────────────────────────────────────────────────────────

def test_list_dir_returns_entries(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "sub").mkdir()
    entries = client.list_dir(str(tmp_path))
    names = [e.name for e in entries]
    assert "a.txt" in names
    assert "b.txt" in names
    assert "sub" in names


def test_list_dir_types(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    (tmp_path / "d").mkdir()
    entries = {e.name: e for e in client.list_dir(str(tmp_path))}
    assert entries["f.txt"].type == "file"
    assert entries["d"].type == "directory"


def test_list_dir_recursive(client: FilesClient, tmp_path: Path) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text("deep")
    entries = client.list_dir(str(tmp_path), recursive=True)
    names = [e.name for e in entries]
    assert "sub" in names
    assert "deep.txt" in names


def test_list_dir_empty(client: FilesClient, tmp_path: Path) -> None:
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert client.list_dir(str(empty)) == []


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_returns_content(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello world")
    assert client.read_file(str(p)) == "hello world"


def test_read_file_too_big(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "big.bin"
    p.write_text("x")
    import runtime.files.client as mod
    from unittest.mock import patch
    with patch.object(mod, "MAX_READ_BYTES", 0):
        with pytest.raises(FileTooBig):
            client.read_file(str(p))


def test_read_file_denied_outside_root(client: FilesClient) -> None:
    with pytest.raises(PathDenied):
        client.read_file("/etc/passwd")


# ── stat ──────────────────────────────────────────────────────────────────────

def test_stat_file(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("abc")
    info = client.stat(str(p))
    assert isinstance(info, FileInfo)
    assert info.type == "file"
    assert info.size == 3
    assert info.modified  # ISO string present


def test_stat_directory(client: FilesClient, tmp_path: Path) -> None:
    d = tmp_path / "mydir"
    d.mkdir()
    info = client.stat(str(d))
    assert info.type == "directory"


# ── search ────────────────────────────────────────────────────────────────────

def test_search_glob_star(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "report.pdf").write_text("r")
    (tmp_path / "notes.txt").write_text("n")
    results = client.search(str(tmp_path), "*.pdf")
    assert len(results) == 1
    assert "report.pdf" in results[0]


def test_search_kind_file(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    results = client.search(str(tmp_path), "*", kind="file")
    assert all(Path(r).is_file() for r in results)


def test_search_kind_directory(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    results = client.search(str(tmp_path), "*", kind="directory")
    assert all(Path(r).is_dir() for r in results)


def test_search_no_matches(client: FilesClient, tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x")
    results = client.search(str(tmp_path), "*.pdf")
    assert results == []


def test_search_invalid_kind(client: FilesClient, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        client.search(str(tmp_path), "*", kind="bogus")


# ── move ──────────────────────────────────────────────────────────────────────

def test_move_file(client: FilesClient, tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("data")
    client.move(str(src), str(dst))
    assert not src.exists()
    assert dst.read_text() == "data"


# ── copy ──────────────────────────────────────────────────────────────────────

def test_copy_file(client: FilesClient, tmp_path: Path) -> None:
    src = tmp_path / "orig.txt"
    dst = tmp_path / "copy.txt"
    src.write_text("original")
    client.copy(str(src), str(dst))
    assert src.exists()
    assert dst.read_text() == "original"


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_file(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "gone.txt"
    p.write_text("bye")
    client.delete(str(p))
    assert not p.exists()


def test_delete_nonempty_dir_without_confirm_raises(client: FilesClient, tmp_path: Path) -> None:
    d = tmp_path / "nonempty"
    d.mkdir()
    (d / "file.txt").write_text("x")
    with pytest.raises(PathDenied, match="--confirm"):
        client.delete(str(d))


def test_delete_nonempty_dir_with_confirm(client: FilesClient, tmp_path: Path) -> None:
    d = tmp_path / "kill_me"
    d.mkdir()
    (d / "child.txt").write_text("x")
    client.delete(str(d), confirm=True)
    assert not d.exists()


def test_delete_empty_dir(client: FilesClient, tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    client.delete(str(d))
    assert not d.exists()


# ── mkdir ─────────────────────────────────────────────────────────────────────

def test_mkdir_creates_nested(client: FilesClient, tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    client.mkdir(str(target))
    assert target.is_dir()


# ── write_file ────────────────────────────────────────────────────────────────

def test_write_file_creates_file(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "new.txt"
    client.write_file(str(p), "hello")
    assert p.read_text() == "hello"


def test_write_file_creates_parent_dirs(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "file.txt"
    client.write_file(str(p), "deep content")
    assert p.read_text() == "deep content"


# ── open_with_app ─────────────────────────────────────────────────────────────

def test_open_with_app_calls_open(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("x")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        client.open_with_app(str(p))
        mock_run.assert_called_once()
        argv = mock_run.call_args[0][0]
        assert argv[0] == "/usr/bin/open"
        assert str(p) in argv


def test_open_with_app_with_named_app(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "doc.txt"
    p.write_text("x")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        client.open_with_app(str(p), app="TextEdit")
        argv = mock_run.call_args[0][0]
        assert "-a" in argv
        assert "TextEdit" in argv
