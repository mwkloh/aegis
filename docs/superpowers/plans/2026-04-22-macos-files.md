# macos-files: Filesystem Capabilities for Eva — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Eva sandboxed filesystem access via a `/files` Telegram slash command (all ops) and four read-only harness tools (LLM-callable via CLI pipeline).

**Architecture:** Python-native `FilesClient` owns path sandboxing (allowed roots from `AegisConfig.files`). Both the `/files` slash handler and the CLI harness tools delegate to it — no cross-process calls to the existing Node.js MCP server. Skill descriptors add intent-routable file queries to the CLI pipeline's `SkillRunner` + `HarnessAdapter` path.

**Tech Stack:** `pathlib`, `subprocess` (for `open_with_app`), `shutil`, Pydantic, pytest, existing AEGIS `Handler` + `HarnessAdapter` patterns.

---

## Files Created / Modified

| File | Action |
|------|--------|
| `runtime/config.py` | Modify — add `FilesConfig` class + `files` field on `AegisConfig` + `_coerce_files()` |
| `runtime/files/__init__.py` | Create |
| `runtime/files/client.py` | Create |
| `runtime/chat/telegram/files_handler.py` | Create |
| `runtime/chat/telegram/handlers.py` | Modify — add `/files` to help catalogue, add `files_client` param to `build_read_only_handlers` |
| `runtime/chat/telegram/bot.py` | Modify — instantiate `FilesClient`, thread through `build_dispatcher` |
| `runtime/harness/tools/files_tool.py` | Create |
| `runtime/harness/adapter.py` | Modify — export `DEFAULT_TOOLS` (rename `_TOOLS`) |
| `runtime/chat/cli.py` | Modify — inject file tools into harness |
| `runtime/skills/catalog/list_files.yaml` | Create |
| `runtime/skills/catalog/read_file.yaml` | Create |
| `runtime/skills/catalog/search_files.yaml` | Create |
| `runtime/skills/catalog/file_info.yaml` | Create |
| `tests/test_files_config.py` | Create |
| `tests/test_files_client.py` | Create |
| `tests/test_files_handler.py` | Create |
| `tests/test_files_harness.py` | Create |

---

### Task 1: `FilesConfig` — config extension

**Files:**
- Modify: `runtime/config.py`
- Create: `tests/test_files_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_files_config.py
"""FilesConfig — Pydantic model + _coerce_files integration."""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import AegisConfig, FilesConfig, _coerce_files, reset_config

pytestmark = pytest.mark.unit


def test_files_config_defaults_are_non_empty() -> None:
    cfg = FilesConfig()
    assert len(cfg.allowed_roots) > 0


def test_files_config_tilde_is_expanded() -> None:
    cfg = FilesConfig(allowed_roots=["~/Documents"])
    assert not str(cfg.allowed_roots[0]).startswith("~")
    assert cfg.allowed_roots[0].is_absolute()


def test_files_config_empty_roots_is_valid() -> None:
    # FilesConfig itself doesn't reject empty — FilesClient does at construction.
    cfg = FilesConfig(allowed_roots=[])
    assert cfg.allowed_roots == []


def test_coerce_files_returns_default_when_raw_is_none() -> None:
    cfg = _coerce_files(None)
    assert isinstance(cfg, FilesConfig)
    assert len(cfg.allowed_roots) > 0


def test_coerce_files_parses_allowed_roots() -> None:
    cfg = _coerce_files({"allowed_roots": ["~/Documents", "~/Downloads"]})
    assert len(cfg.allowed_roots) == 2
    assert all(p.is_absolute() for p in cfg.allowed_roots)


def test_coerce_files_falls_back_on_non_dict() -> None:
    cfg = _coerce_files("garbage")
    assert isinstance(cfg, FilesConfig)


def test_aegis_config_has_files_field() -> None:
    cfg = AegisConfig()
    assert hasattr(cfg, "files")
    assert isinstance(cfg.files, FilesConfig)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
source .venv/bin/activate
pytest tests/test_files_config.py -v
```

Expected: `ImportError: cannot import name 'FilesConfig'` or similar.

- [ ] **Step 3: Add `FilesConfig` to `runtime/config.py`**

Add `FilesConfig` class after `BoardConfig` import (around line 18). Insert before `logger = ...`:

```python
class FilesConfig(BaseModel):
    """Sandboxed filesystem access config for /files and file harness tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_roots: list[Path] = Field(
        default_factory=lambda: [
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Development",
            Path.home() / "data",
        ]
    )

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def _expand_roots(cls, v: object) -> list[Path]:
        if not isinstance(v, list):
            return []
        return [Path(str(r)).expanduser() for r in v if isinstance(r, str) and r.strip()]
```

- [ ] **Step 4: Add `_coerce_files()` to `runtime/config.py`**

Add after `_coerce_board()` (around line 304):

```python
def _coerce_files(raw: Any) -> FilesConfig:
    """Build a `FilesConfig` from `config.json` → `files`.

    Missing / non-dict → default (5 standard macOS dirs).
    """
    if not isinstance(raw, dict):
        return FilesConfig()
    roots_raw = raw.get("allowed_roots") or raw.get("allowedRoots")
    if isinstance(roots_raw, list):
        try:
            return FilesConfig(allowed_roots=roots_raw)
        except (ValueError, TypeError):
            pass
    return FilesConfig()
```

- [ ] **Step 5: Add `files` field to `AegisConfig` and wire `_coerce_files()` into `_coerce()`**

In `AegisConfig` (around line 131), add after `board`:

```python
    files: FilesConfig = Field(default_factory=FilesConfig)
```

In `_coerce()` (around line 219), add `"files"` to the returned dict:

```python
    return {
        "models": models,
        "providers": providers,
        "telegram": telegram,
        "storage": StorageConfig(),
        "vault_indexing": _coerce_vault_indexing(cfg.get("vaultIndexing")),
        "board": board,
        "files": _coerce_files(cfg.get("files")),
    }
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
pytest tests/test_files_config.py -v
```

Expected: 7 passed.

- [ ] **Step 7: Run the full suite — verify no regressions**

```bash
pytest --tb=short -q
```

Expected: all existing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add runtime/config.py tests/test_files_config.py
git commit -m "feat(files): add FilesConfig to AegisConfig with _coerce_files"
```

---

### Task 2: `FilesClient` — core filesystem operations

**Files:**
- Create: `runtime/files/__init__.py`
- Create: `runtime/files/client.py`
- Create: `tests/test_files_client.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    assert client.list_dir(str(tmp_path)) == []


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_returns_content(client: FilesClient, tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello world")
    assert client.read_file(str(p)) == "hello world"


def test_read_file_too_big(client: FilesClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "big.bin"
    p.write_text("x")
    monkeypatch.setattr(p.stat().__class__, "st_size", property(lambda self: MAX_READ_BYTES + 1), raising=False)
    # Monkeypatching stat is tricky; use a simpler approach:
    import runtime.files.client as mod
    original = mod.MAX_READ_BYTES
    mod.MAX_READ_BYTES = 0  # every file is "too big"
    try:
        with pytest.raises(FileTooBig):
            client.read_file(str(p))
    finally:
        mod.MAX_READ_BYTES = original


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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_files_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'runtime.files'`.

- [ ] **Step 3: Create `runtime/files/__init__.py`**

```python
# runtime/files/__init__.py
from .client import DirEntry, FileInfo, FilesClient, FileTooBig, PathDenied

__all__ = ["DirEntry", "FileInfo", "FilesClient", "FileTooBig", "PathDenied"]
```

- [ ] **Step 4: Create `runtime/files/client.py`**

```python
# runtime/files/client.py
"""Path-sandboxed filesystem operations for Eva.

Ported from ~/.atamai/mcp-servers/macos-files-mcp (Node.js → Python).
All paths are validated against `allowed_roots` before any operation.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB — mirrors MCP server cap
_SEARCH_KIND = frozenset({"file", "directory", "any"})


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
    escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
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
        except PermissionError:
            pass

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
            argv += ["-a", app]
        argv.append(str(p))
        subprocess.run(argv, check=True)


__all__ = ["DirEntry", "FileInfo", "FilesClient", "FileTooBig", "MAX_READ_BYTES", "PathDenied"]
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/test_files_client.py -v
```

Expected: all tests pass. If `test_read_file_too_big` is flaky due to monkeypatching, adjust by writing a zero-byte file and setting `MAX_READ_BYTES = -1` in a fixture.

- [ ] **Step 6: Run the full suite — verify no regressions**

```bash
pytest --tb=short -q
```

- [ ] **Step 7: Commit**

```bash
git add runtime/files/ tests/test_files_client.py
git commit -m "feat(files): FilesClient — path-sandboxed filesystem ops"
```

---

### Task 3: `/files` slash handler

**Files:**
- Create: `runtime/chat/telegram/files_handler.py`
- Create: `tests/test_files_handler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_files_handler.py
"""Unit tests for the /files slash handler."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from runtime.chat.telegram.dispatch import IncomingMessage, ParsedCommand
from runtime.chat.telegram.files_handler import files_handler
from runtime.files.client import DirEntry, FileInfo, FilesClient, FileTooBig, PathDenied

pytestmark = pytest.mark.unit

# ── Stub client ───────────────────────────────────────────────────────────────

_NOW = "2026-04-22T00:00:00+00:00"


class _StubClient:
    """Pre-configured fake FilesClient. Every method is overridable."""

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
def handler(stub: _StubClient) -> object:
    return files_handler(client=stub)  # type: ignore[arg-type]


# ── No args ───────────────────────────────────────────────────────────────────

def test_no_args_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files"))
    assert "Usage" in result


def test_unknown_subverb_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "bogus"))
    assert "Usage" in result


# ── ls ────────────────────────────────────────────────────────────────────────

def test_ls_formats_file_entry(handler: object, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="report.pdf", path="/x/report.pdf", type="file", size=2048, modified=_NOW)]
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "[f]" in result
    assert "report.pdf" in result


def test_ls_formats_directory_entry(handler: object, stub: _StubClient) -> None:
    stub.entries = [DirEntry(name="docs", path="/x/docs", type="directory", size=0, modified=_NOW)]
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "[d]" in result
    assert "docs" in result


def test_ls_empty_directory(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "ls", "/x"))
    assert "empty" in result.lower()


def test_ls_missing_path_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "ls"))
    assert "Usage" in result


# ── read ──────────────────────────────────────────────────────────────────────

def test_read_returns_content(handler: object, stub: _StubClient) -> None:
    stub.content = "hello world"
    result = handler(_msg(), _cmd("/files", "read", "/x/file.txt"))
    assert "hello world" in result


def test_read_clips_long_content(handler: object, stub: _StubClient) -> None:
    stub.content = "x" * 5000
    result = handler(_msg(), _cmd("/files", "read", "/x/file.txt"))
    assert "truncated" in result
    assert len(result) < 5000


def test_read_missing_path_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "read"))
    assert "Usage" in result


# ── stat ──────────────────────────────────────────────────────────────────────

def test_stat_shows_all_fields(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "stat", "/x/doc.txt"))
    assert "type" in result
    assert "size" in result
    assert "modified" in result
    assert "mode" in result


# ── find ──────────────────────────────────────────────────────────────────────

def test_find_no_matches(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "find", "/x", "*.pdf"))
    assert "No matches" in result


def test_find_shows_paths(handler: object, stub: _StubClient) -> None:
    stub.search_results = ["/x/report.pdf"]
    result = handler(_msg(), _cmd("/files", "find", "/x", "*.pdf"))
    assert "/x/report.pdf" in result


def test_find_missing_args_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "find", "/x"))
    assert "Usage" in result


# ── mv ────────────────────────────────────────────────────────────────────────

def test_mv_success_message(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "mv", "/x/a.txt", "/x/b.txt"))
    assert "Moved" in result


def test_mv_missing_dst_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "mv", "/x/a.txt"))
    assert "Usage" in result


# ── cp ────────────────────────────────────────────────────────────────────────

def test_cp_success_message(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "cp", "/x/a.txt", "/x/b.txt"))
    assert "Copied" in result


# ── rm ────────────────────────────────────────────────────────────────────────

def test_rm_without_confirm_flag(handler: object, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("non-empty, use --confirm")
    result = handler(_msg(), _cmd("/files", "rm", "/x/dir"))
    assert "Access denied" in result


def test_rm_with_confirm_flag(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "rm", "--confirm", "/x/dir"))
    assert "Deleted" in result


def test_rm_missing_path_returns_usage(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "rm"))
    assert "Usage" in result


# ── mkdir ─────────────────────────────────────────────────────────────────────

def test_mkdir_success(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "mkdir", "/x/newdir"))
    assert "Created" in result


# ── open ──────────────────────────────────────────────────────────────────────

def test_open_success(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "open", "/x/doc.txt"))
    assert "Opened" in result


def test_open_with_named_app(handler: object) -> None:
    result = handler(_msg(), _cmd("/files", "open", "/x/doc.txt", "Preview"))
    assert "Preview" in result


# ── Error paths ───────────────────────────────────────────────────────────────

def test_path_denied_shows_access_denied(handler: object, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    result = handler(_msg(), _cmd("/files", "read", "/etc/passwd"))
    assert "Access denied" in result


def test_file_too_big_shows_error(handler: object, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("10 MB exceeded")
    result = handler(_msg(), _cmd("/files", "read", "/x/huge.bin"))
    assert "too large" in result.lower()


def test_os_error_shows_error(handler: object, stub: _StubClient) -> None:
    stub.raise_on_next = OSError("permission denied")
    result = handler(_msg(), _cmd("/files", "read", "/x/locked.txt"))
    assert "Error" in result
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_files_handler.py -v
```

Expected: `ModuleNotFoundError: No module named 'runtime.chat.telegram.files_handler'`.

- [ ] **Step 3: Create `runtime/chat/telegram/files_handler.py`**

```python
# runtime/chat/telegram/files_handler.py
"""`/files` slash handler — operator-driven filesystem access via FilesClient."""
from __future__ import annotations

from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.files.client import FilesClient, FileTooBig, PathDenied

_MAX_READ_CHARS = 3500
_USAGE = """\
Usage: /files <sub-command>
  ls [-r] <path>           list directory
  read <path>              read file content
  stat <path>              file metadata
  find <dir> <pattern>     search by name (glob)
  mv <src> <dst>           move / rename
  cp <src> <dst>           copy
  rm [--confirm] <path>    delete (--confirm required for non-empty dirs)
  mkdir <path>             create directory
  open <path> [app]        open with macOS app"""


def files_handler(*, client: FilesClient) -> Handler:
    """Factory for the dispatcher — closes over `client`."""

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return _USAGE
        sub = cmd.args[0].strip().lower()
        tail = cmd.args[1:]
        try:
            if sub == "ls":    return _ls(client, tail)
            if sub == "read":  return _read(client, tail)
            if sub == "stat":  return _stat(client, tail)
            if sub == "find":  return _find(client, tail)
            if sub == "mv":    return _mv(client, tail)
            if sub == "cp":    return _cp(client, tail)
            if sub == "rm":    return _rm(client, tail)
            if sub == "mkdir": return _mkdir(client, tail)
            if sub == "open":  return _open(client, tail)
            return _USAGE
        except PathDenied as exc:
            return f"Access denied: {exc}"
        except FileTooBig as exc:
            return f"File too large: {exc}"
        except OSError as exc:
            return f"Error: {exc}"

    return _handle


def _ls(client: FilesClient, args: tuple[str, ...]) -> str:
    remaining = list(args)
    if not remaining:
        return "Usage: /files ls [-r] <path>"
    recursive = False
    if remaining[0] == "-r":
        recursive = True
        remaining.pop(0)
    if not remaining:
        return "Usage: /files ls [-r] <path>"
    entries = client.list_dir(remaining[0], recursive=recursive)
    if not entries:
        return "(empty directory)"
    lines = []
    for e in entries:
        kind = "d" if e.type == "directory" else "f" if e.type == "file" else "?"
        size_str = _human_size(e.size)
        date_str = (e.modified or "")[:10] or "-"
        lines.append(f"[{kind}] {e.name}  ({size_str}, {date_str})")
    return "\n".join(lines)


def _human_size(n: int) -> str:
    for unit, divisor in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= divisor:
            return f"{n / divisor:.1f} {unit}"
    return f"{n} B"


def _read(client: FilesClient, args: tuple[str, ...]) -> str:
    if not args:
        return "Usage: /files read <path>"
    content = client.read_file(args[0])
    if len(content) > _MAX_READ_CHARS:
        truncated = len(content) - _MAX_READ_CHARS
        return content[:_MAX_READ_CHARS] + f"\n… (truncated {truncated} chars)"
    return content


def _stat(client: FilesClient, args: tuple[str, ...]) -> str:
    if not args:
        return "Usage: /files stat <path>"
    info = client.stat(args[0])
    return "\n".join([
        f"path: {info.path}",
        f"type: {info.type}",
        f"size: {_human_size(info.size)} ({info.size} bytes)",
        f"created: {info.created}",
        f"modified: {info.modified}",
        f"accessed: {info.accessed}",
        f"mode: {info.mode}",
    ])


def _find(client: FilesClient, args: tuple[str, ...]) -> str:
    if len(args) < 2:
        return "Usage: /files find <dir> <pattern>"
    matches = client.search(args[0], args[1])
    return "\n".join(matches) if matches else "No matches."


def _mv(client: FilesClient, args: tuple[str, ...]) -> str:
    if len(args) < 2:
        return "Usage: /files mv <src> <dst>"
    client.move(args[0], args[1])
    return f"Moved: {args[0]} → {args[1]}"


def _cp(client: FilesClient, args: tuple[str, ...]) -> str:
    if len(args) < 2:
        return "Usage: /files cp <src> <dst>"
    client.copy(args[0], args[1])
    return f"Copied: {args[0]} → {args[1]}"


def _rm(client: FilesClient, args: tuple[str, ...]) -> str:
    remaining = list(args)
    confirm = False
    if remaining and remaining[0] == "--confirm":
        confirm = True
        remaining.pop(0)
    if not remaining:
        return "Usage: /files rm [--confirm] <path>"
    client.delete(remaining[0], confirm=confirm)
    return f"Deleted: {remaining[0]}"


def _mkdir(client: FilesClient, args: tuple[str, ...]) -> str:
    if not args:
        return "Usage: /files mkdir <path>"
    client.mkdir(args[0])
    return f"Created: {args[0]}"


def _open(client: FilesClient, args: tuple[str, ...]) -> str:
    if not args:
        return "Usage: /files open <path> [app]"
    path = args[0]
    app = args[1] if len(args) > 1 else None
    client.open_with_app(path, app=app)
    return f"Opened: {path} with {app}" if app else f"Opened: {path}"


__all__ = ["files_handler"]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_files_handler.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the full suite**

```bash
pytest --tb=short -q
```

- [ ] **Step 6: Commit**

```bash
git add runtime/chat/telegram/files_handler.py tests/test_files_handler.py
git commit -m "feat(files): /files slash handler with ls/read/stat/find/mv/cp/rm/mkdir/open"
```

---

### Task 4: Wire `/files` into the Telegram bot

**Files:**
- Modify: `runtime/chat/telegram/handlers.py` (two changes)
- Modify: `runtime/chat/telegram/bot.py` (two changes)

No new test file — verify by running the full suite.

- [ ] **Step 1: Add `/files` to `DEFAULT_COMMAND_HELP` in `handlers.py`**

In `runtime/chat/telegram/handlers.py` at line ~234, add after the `/board` entry:

```python
    "/files": (
        "Filesystem access: ls [-r], read, stat, find, mv, cp, rm [--confirm], mkdir, open. "
        "Paths sandboxed to config files.allowed_roots."
    ),
```

- [ ] **Step 2: Add `files_client` parameter to `build_read_only_handlers()` in `handlers.py`**

Change the function signature (around line 445) to:

```python
def build_read_only_handlers(
    workspace: Path,
    *,
    sessions_dir: Path | None = None,
    clock: Clock | None = None,
    cfg: AegisConfig | None = None,
    router: ModelRouter | None = None,
    recall_resolver: ColdRefResolver | None = None,
    recall_reader: ColdStorageReader | None = None,
    vault_loader: VaultBodyLoader | None = None,
    vault_indexer: VaultIndexer | None = None,
    vault_tier2: Tier2Store | None = None,
    vault_state: VaultState | None = None,
    heartbeat_path: Path | None = None,
    health_store: ScheduledJobStore | None = None,
    files_client: object | None = None,   # FilesClient | None — avoid circular import
) -> dict[str, Handler]:
```

Add the import at the top of `handlers.py`:

```python
from runtime.chat.telegram.files_handler import files_handler
```

At the end of `build_read_only_handlers`, before `return handlers`, add:

```python
    if files_client is not None:
        from runtime.files.client import FilesClient  # noqa: PLC0415
        handlers["/files"] = files_handler(client=files_client)  # type: ignore[arg-type]
```

- [ ] **Step 3: Add `files_client` parameter to `build_dispatcher()` in `bot.py`**

Change `build_dispatcher()` signature (around line 122) to accept:

```python
    files_client: object | None = None,
```

Thread it to `build_read_only_handlers`:

```python
    read_only = build_read_only_handlers(
        workspace,
        ...
        files_client=files_client,
    )
```

- [ ] **Step 4: Instantiate `FilesClient` in `build_application()` and pass through**

At the top of `build_application()` (around line 1010), after `events` is set up, add:

```python
    from runtime.files.client import FilesClient  # noqa: PLC0415
    try:
        files_client: object | None = FilesClient(cfg.files.allowed_roots)
    except ValueError:
        logger.warning("files.disabled", extra={"reason": "no_allowed_roots"})
        files_client = None
```

In the `build_dispatcher(...)` call (around line 1095), add `files_client=files_client`.

- [ ] **Step 5: Run the full suite — verify no regressions**

```bash
pytest --tb=short -q
```

Expected: all existing tests still pass. New `/files` handler is wired but not separately tested here.

- [ ] **Step 6: Commit**

```bash
git add runtime/chat/telegram/handlers.py runtime/chat/telegram/bot.py
git commit -m "feat(files): wire /files handler into Telegram dispatcher"
```

---

### Task 5: Harness tools + skill descriptors

**Files:**
- Create: `runtime/harness/tools/files_tool.py`
- Modify: `runtime/harness/adapter.py` (export `DEFAULT_TOOLS`)
- Modify: `runtime/chat/cli.py` (inject file tools)
- Create: `runtime/skills/catalog/list_files.yaml`
- Create: `runtime/skills/catalog/read_file.yaml`
- Create: `runtime/skills/catalog/search_files.yaml`
- Create: `runtime/skills/catalog/file_info.yaml`
- Create: `tests/test_files_harness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_files_harness.py
"""Unit tests for make_files_tools harness callables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runtime.files.client import DirEntry, FileInfo, FilesClient, FileTooBig, PathDenied
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
    assert len(result["result"]) < 5000


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


def test_path_denied_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = PathDenied("outside roots")
    with pytest.raises(RuntimeError, match="outside roots"):
        tools["files_read"]({"path": "/etc/passwd"})


def test_file_too_big_raises_runtime_error(tools: dict, stub: _StubClient) -> None:
    stub.raise_on_next = FileTooBig("too big")
    with pytest.raises(RuntimeError, match="too big"):
        tools["files_read"]({"path": "/x/huge.bin"})


def test_make_files_tools_returns_four_callables(tools: dict) -> None:
    assert set(tools.keys()) == {"files_list", "files_read", "files_stat", "files_search"}
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_files_harness.py -v
```

Expected: `ModuleNotFoundError: No module named 'runtime.harness.tools.files_tool'`.

- [ ] **Step 3: Create `runtime/harness/tools/files_tool.py`**

```python
# runtime/harness/tools/files_tool.py
"""Harness tool callables for read-only filesystem access.

Used by the CLI pipeline's HarnessAdapter. The four callables wrap
FilesClient read ops and collapse PathDenied/FileTooBig into RuntimeError
so HarnessAdapter's `except Exception` block catches them uniformly.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from runtime.files.client import FilesClient, FileTooBig, PathDenied

_MAX_TOOL_CHARS = 3500


def make_files_tools(
    client: FilesClient,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Return the four read-only harness callables closed over `client`."""

    def _wrap(fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            try:
                return fn(args)
            except (PathDenied, FileTooBig, OSError) as exc:
                raise RuntimeError(str(exc)) from exc
        return wrapper

    def files_list(args: dict[str, Any]) -> dict[str, Any]:
        entries = client.list_dir(args["path"], recursive=bool(args.get("recursive", False)))
        lines = []
        for e in entries:
            kind = "d" if e.type == "directory" else "f" if e.type == "file" else "?"
            lines.append(f"[{kind}] {e.name}  ({e.size} B, {e.modified or '-'})")
        return {"result": "\n".join(lines) or "(empty directory)"}

    def files_read(args: dict[str, Any]) -> dict[str, Any]:
        content = client.read_file(args["path"])
        if len(content) > _MAX_TOOL_CHARS:
            content = content[:_MAX_TOOL_CHARS] + "… (truncated)"
        return {"result": content}

    def files_stat(args: dict[str, Any]) -> dict[str, Any]:
        info = client.stat(args["path"])
        text = (
            f"type: {info.type}\n"
            f"size: {info.size} B\n"
            f"created: {info.created}\n"
            f"modified: {info.modified}\n"
            f"mode: {info.mode}"
        )
        return {"result": text}

    def files_search(args: dict[str, Any]) -> dict[str, Any]:
        matches = client.search(
            args["directory"], args["pattern"], kind=str(args.get("kind", "any"))
        )
        return {"result": "\n".join(matches) if matches else "No matches."}

    return {
        "files_list": _wrap(files_list),
        "files_read": _wrap(files_read),
        "files_stat": _wrap(files_stat),
        "files_search": _wrap(files_search),
    }


__all__ = ["make_files_tools"]
```

- [ ] **Step 4: Run harness tests — verify they pass**

```bash
pytest tests/test_files_harness.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Export `DEFAULT_TOOLS` from `runtime/harness/adapter.py`**

In `runtime/harness/adapter.py`, rename `_TOOLS` → `DEFAULT_TOOLS` and export it:

```python
# Before:
_TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "echo": echo,
    "respond": respond,
    "time": time_tool,
}

class HarnessAdapter:
    def __init__(self, tools=None) -> None:
        self._tools = tools if tools is not None else _TOOLS
```

```python
# After:
DEFAULT_TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "echo": echo,
    "respond": respond,
    "time": time_tool,
}

class HarnessAdapter:
    def __init__(
        self,
        tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else DEFAULT_TOOLS
```

Update `runtime/harness/__init__.py` to export `DEFAULT_TOOLS`:

```python
from .adapter import DEFAULT_TOOLS, HarnessAdapter
from .contract import ToolIntent, ToolResult

__all__ = ["DEFAULT_TOOLS", "HarnessAdapter", "ToolIntent", "ToolResult"]
```

- [ ] **Step 6: Inject file tools into the CLI pipeline in `runtime/chat/cli.py`**

Find the line (around line 160):

```python
    harness = HarnessAdapter()
```

Replace with:

```python
    from runtime.files.client import FilesClient  # noqa: PLC0415
    from runtime.harness import DEFAULT_TOOLS  # noqa: PLC0415
    from runtime.harness.tools.files_tool import make_files_tools  # noqa: PLC0415

    try:
        _files_client = FilesClient(cfg.files.allowed_roots)
        _file_tools = make_files_tools(_files_client)
    except ValueError:
        _file_tools = {}
    harness = HarnessAdapter(tools={**DEFAULT_TOOLS, **_file_tools})
```

- [ ] **Step 7: Create the four skill descriptor YAML files**

```yaml
# runtime/skills/catalog/list_files.yaml
id: list_files
version: 0.1.0
description: List files and directories at a local path.
intents:
  - list_files
  - browse_directory
tool: files_list
args_schema:
  type: object
  properties:
    path:
      type: string
      minLength: 1
      maxLength: 512
    recursive:
      type: boolean
  required:
    - path
requires_tier1: true
```

```yaml
# runtime/skills/catalog/read_file.yaml
id: read_file
version: 0.1.0
description: Read the text content of a local file.
intents:
  - read_file
  - open_file
tool: files_read
args_schema:
  type: object
  properties:
    path:
      type: string
      minLength: 1
      maxLength: 512
  required:
    - path
requires_tier1: true
```

```yaml
# runtime/skills/catalog/search_files.yaml
id: search_files
version: 0.1.0
description: Search for files matching a glob pattern within a local directory.
intents:
  - search_files
  - find_files
tool: files_search
args_schema:
  type: object
  properties:
    directory:
      type: string
      minLength: 1
      maxLength: 512
    pattern:
      type: string
      minLength: 1
      maxLength: 256
    kind:
      type: string
      enum:
        - file
        - directory
        - any
  required:
    - directory
    - pattern
requires_tier1: true
```

```yaml
# runtime/skills/catalog/file_info.yaml
id: file_info
version: 0.1.0
description: Get metadata (size, dates, type) for a local file or directory.
intents:
  - file_info
  - file_metadata
tool: files_stat
args_schema:
  type: object
  properties:
    path:
      type: string
      minLength: 1
      maxLength: 512
  required:
    - path
requires_tier1: true
```

- [ ] **Step 8: Run the full suite — verify everything passes**

```bash
pytest --tb=short -q
```

Expected: all existing tests + new harness tests pass. If `SkillRegistry.from_directory` loads the catalog and validates descriptor shapes, the YAML files must parse cleanly — the suite will tell you if not.

- [ ] **Step 9: Commit**

```bash
git add \
  runtime/harness/tools/files_tool.py \
  runtime/harness/adapter.py \
  runtime/harness/__init__.py \
  runtime/chat/cli.py \
  runtime/skills/catalog/list_files.yaml \
  runtime/skills/catalog/read_file.yaml \
  runtime/skills/catalog/search_files.yaml \
  runtime/skills/catalog/file_info.yaml \
  tests/test_files_harness.py
git commit -m "feat(files): harness tools + skill descriptors for LLM-callable file ops"
```

---

## Context for implementers

**Existing patterns to follow:**
- `runtime/board/config.py` — how `BoardConfig` is structured (frozen Pydantic, `field_validator` for path expansion)
- `runtime/config.py::_coerce_board()` — how a config section is parsed from `config.json` with graceful fallback
- `runtime/chat/telegram/handlers.py::build_read_only_handlers()` — how optional handlers are registered (`if x is not None: handlers["/x"] = x_handler(...)`)
- `runtime/chat/telegram/board_handler.py` — full async handler with Protocol deps
- `runtime/harness/adapter.py` — the `_TOOLS` / `HarnessAdapter` pattern

**Running tests:**
```bash
source .venv/bin/activate
pytest tests/test_files_config.py tests/test_files_client.py tests/test_files_handler.py tests/test_files_harness.py -v
```

**Key invariants:**
- `PathDenied` and `FileTooBig` must never propagate past the handler or harness tool boundary — always collapse to user-facing strings
- `FilesClient.__init__` with empty `allowed_roots` must raise `ValueError` immediately
- `delete()` on a non-empty directory without `confirm=True` must raise `PathDenied`, not `OSError`
- Skill YAML descriptors must load cleanly via `SkillRegistry.from_directory` — run the full suite after adding them
