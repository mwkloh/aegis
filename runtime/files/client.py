"""Path-sandboxed filesystem operations for Eva.

Ported from ~/.atamai/mcp-servers/macos-files-mcp (Node.js → Python).
All paths are validated against `allowed_roots` before any operation.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB — mirrors MCP server cap
_SEARCH_KIND = frozenset({"file", "directory", "any"})
_APP_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]+$")


class PathDenied(Exception):
    """Path is outside all allowed roots, or a destructive op needs --confirm."""


class FileTooBig(Exception):
    """File exceeds MAX_READ_BYTES."""


@dataclass(frozen=True)
class DirEntry:
    name: str
    path: str
    type: Literal["file", "directory", "other"]
    size: int
    modified: str | None  # ISO 8601, or None if stat failed


@dataclass(frozen=True)
class FileInfo:
    path: str
    type: Literal["file", "directory", "other"]
    size: int
    created: str   # ISO 8601
    modified: str  # ISO 8601
    accessed: str  # ISO 8601
    mode: str      # e.g. "100644" (octal, no 0o prefix)


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.compile(f"^{escaped}$", re.IGNORECASE)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class FilesClient:
    """Path-sandboxed filesystem operations."""

    def __init__(self, allowed_roots: list[Path]) -> None:
        if not allowed_roots:
            raise ValueError("FilesClient requires at least one allowed root")
        self._roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate(self, p: str) -> Path:
        if not p.startswith(("/", "~")):
            raise PathDenied(
                f"Path {p!r} must be absolute (e.g. /Users/you/Desktop/foo.png) "
                "or start with '~'. Relative paths are not accepted."
            )
        resolved = Path(p).expanduser().resolve()
        for root in self._roots:
            if resolved == root or str(resolved).startswith(str(root) + "/"):
                return resolved
        raise PathDenied(
            f"Path '{resolved}' is outside allowed roots: "
            + ", ".join(str(r) for r in self._roots)
        )

    # ── Read ops ──────────────────────────────────────────────────────────────

    def list_dir(self, path: str, *, recursive: bool = False) -> list[DirEntry]:
        dirpath = self._validate(path)
        return self._collect_entries(dirpath, recursive=recursive)

    def _collect_entries(self, dirpath: Path, *, recursive: bool) -> list[DirEntry]:
        entries: list[DirEntry] = []
        for item in sorted(dirpath.iterdir(), key=lambda e: e.name):
            try:
                s = item.stat()
                size = s.st_size
                modified = _iso(s.st_mtime)
                kind: Literal["file", "directory", "other"] = (
                    "directory" if item.is_dir() else "file" if item.is_file() else "other"
                )
            except OSError:
                size = 0
                modified = None
                kind = "other"
            entries.append(DirEntry(name=item.name, path=str(item), type=kind, size=size, modified=modified))
            if recursive and kind == "directory":
                entries.extend(self._collect_entries(item, recursive=True))
        return entries

    def read_file(self, path: str) -> str:
        filepath = self._validate(path)
        s = filepath.stat()
        if s.st_size > MAX_READ_BYTES:
            raise FileTooBig(f"'{filepath}' is {s.st_size} bytes (max {MAX_READ_BYTES})")
        return filepath.read_text(encoding="utf-8", errors="replace")

    def stat(self, path: str) -> FileInfo:
        p = self._validate(path)
        s = p.stat()
        kind: Literal["file", "directory", "other"] = (
            "directory" if p.is_dir() else "file" if p.is_file() else "other"
        )
        created_ts = getattr(s, "st_birthtime", s.st_ctime)
        return FileInfo(
            path=str(p),
            type=kind,
            size=s.st_size,
            created=_iso(created_ts),
            modified=_iso(s.st_mtime),
            accessed=_iso(s.st_atime),
            mode=oct(s.st_mode)[2:],
        )

    def search(self, directory: str, pattern: str, *, kind: str = "any") -> list[str]:
        if kind not in _SEARCH_KIND:
            raise ValueError(f"kind must be one of {sorted(_SEARCH_KIND)}, got {kind!r}")
        dirpath = self._validate(directory)
        regex = _glob_to_regex(pattern)
        results: list[str] = []
        self._walk_search(dirpath, regex, kind, results)
        return sorted(results)

    def _walk_search(self, current: Path, regex: re.Pattern[str], kind: str, results: list[str]) -> None:
        try:
            for item in current.iterdir():
                if regex.match(item.name):
                    if (
                        kind == "any"
                        or (kind == "file" and item.is_file())
                        or (kind == "directory" and item.is_dir())
                    ):
                        results.append(str(item))
                if item.is_dir():
                    self._walk_search(item, regex, kind, results)
        except PermissionError as exc:
            logger.debug(
                "files.search.permission_denied",
                extra={"path": str(current), "err": str(exc)},
            )

    # ── Destructive ops ───────────────────────────────────────────────────────

    def write_file(self, path: str, content: str, *, create_dirs: bool = True) -> None:
        filepath = self._validate(path)
        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    def move(self, src: str, dst: str) -> None:
        src_path = self._validate(src)
        dst_path = self._validate(dst)
        src_path.rename(dst_path)

    def copy(self, src: str, dst: str) -> None:
        src_path = self._validate(src)
        dst_path = self._validate(dst)
        shutil.copy2(src_path, dst_path)

    def delete(self, path: str, *, confirm: bool = False) -> None:
        p = self._validate(path)
        if p.is_dir():
            entries = list(p.iterdir())
            if entries and not confirm:
                raise PathDenied(
                    f"Directory '{p}' is not empty ({len(entries)} items). "
                    "Pass confirm=True or use /files rm --confirm."
                )
            shutil.rmtree(p)
        else:
            p.unlink()

    def mkdir(self, path: str) -> None:
        p = self._validate(path)
        p.mkdir(parents=True, exist_ok=True)

    def open_with_app(self, path: str, app: str | None = None) -> None:
        p = self._validate(path)
        argv = ["/usr/bin/open"]
        if app:
            if len(app) > 64 or not _APP_NAME_RE.match(app):
                raise PathDenied(
                    f"Invalid app name {app!r}. Only alphanumerics, space, "
                    "underscore, dot, and hyphen are allowed (max 64 chars)."
                )
            argv += ["-a", app]
        argv.append(str(p))
        subprocess.run(argv, check=True)


__all__ = ["DirEntry", "FileInfo", "FilesClient", "FileTooBig", "MAX_READ_BYTES", "PathDenied"]
