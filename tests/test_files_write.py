# tests/test_files_write.py
"""FilesClient.write_file — path-sandboxed, size-capped text writes."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.files.client import MAX_WRITE_BYTES, FilesClient, FileTooBig, PathDenied

pytestmark = pytest.mark.unit


@pytest.fixture
def client(tmp_path: Path) -> FilesClient:
    return FilesClient(allowed_roots=[tmp_path])


# ── Happy path ────────────────────────────────────────────────────────────────

def test_write_inside_root_succeeds(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    content = "héllo wörld 🎉"  # non-ASCII — bytes_written must reflect UTF-8 length
    result = client.write_file(str(p), content)

    assert p.read_text(encoding="utf-8") == content
    assert result["path"] == str(p)
    assert result["bytes_written"] == len(content.encode("utf-8"))
    # Sanity: the non-ASCII content must actually differ in char vs byte length.
    assert result["bytes_written"] != len(content)
    assert result["overwrote"] is False


def test_write_overwrite_existing_file_succeeds(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "existing.txt"
    p.write_text("old content", encoding="utf-8")

    result = client.write_file(str(p), "new content")

    assert p.read_text(encoding="utf-8") == "new content"
    assert result["bytes_written"] == len(b"new content")
    assert result["overwrote"] is True


# ── Atomic write (tmp sibling + rename) ─────────────────────────────────────

def test_write_leaves_no_tmp_sibling_after_success(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "clean.txt"
    client.write_file(str(p), "content")

    tmp = p.with_suffix(p.suffix + ".tmp")
    assert not tmp.exists()


def test_write_replace_failure_preserves_original_and_cleans_tmp(
    client: FilesClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "protected.txt"
    p.write_text("original content", encoding="utf-8")

    def _boom(self: Path, target: object) -> Path:
        raise OSError("simulated crash mid-replace")

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="simulated crash mid-replace"):
        client.write_file(str(p), "new content")

    # A crash during the rename must not truncate/destroy the original...
    assert p.read_text(encoding="utf-8") == "original content"
    # ...and must not leave a half-written tmp file behind.
    tmp = p.with_suffix(p.suffix + ".tmp")
    assert not tmp.exists()


def test_write_tilde_expansion(
    client: FilesClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    result = client.write_file("~/note.txt", "hi")
    p = tmp_path / "note.txt"

    assert p.read_text(encoding="utf-8") == "hi"
    assert result["path"] == str(p)


# ── Containment ───────────────────────────────────────────────────────────────

def test_write_outside_root_rejected(client: FilesClient) -> None:
    with pytest.raises(PathDenied):
        client.write_file("/etc/passwd-aegis-test", "pwned")


def test_write_symlink_escape_rejected(client: FilesClient, tmp_path: Path) -> None:
    # A symlink INSIDE the allowed root pointing OUTSIDE it — the resolved
    # target must be caught by containment, not the pre-resolution path.
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    link = tmp_path / "escape"
    link.symlink_to(outside)

    target = link / "file.txt"
    with pytest.raises(PathDenied):
        client.write_file(str(target), "pwned")

    assert not (outside / "file.txt").exists()


# ── Parent directory must already exist ──────────────────────────────────────

def test_write_missing_parent_dir_raises(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "missing_dir" / "file.txt"
    with pytest.raises(FileNotFoundError):
        client.write_file(str(p), "content")

    assert not p.parent.exists()
    assert not p.exists()


# ── Size cap ──────────────────────────────────────────────────────────────────

def test_write_content_over_cap_rejected(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    content = "x" * (MAX_WRITE_BYTES + 1)
    with pytest.raises(FileTooBig):
        client.write_file(str(p), content)

    assert not p.exists()


def test_write_content_exactly_at_cap_accepted(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "exact.txt"
    content = "x" * MAX_WRITE_BYTES
    result = client.write_file(str(p), content)

    assert result["bytes_written"] == MAX_WRITE_BYTES
    assert p.stat().st_size == MAX_WRITE_BYTES
