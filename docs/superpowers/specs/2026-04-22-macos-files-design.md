# macos-files: Filesystem Capabilities for Eva

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give Eva (AEGIS Telegram bot) sandboxed filesystem access via two surfaces: a `/files` operator slash command (read + destructive ops) and four read-only LLM skill descriptors (intent-routed for autonomous reasoning).

**Source reference:** `~/.atamai/mcp-servers/macos-files-mcp/src/index.ts` — Node.js MCP server with 10 filesystem tools. AEGIS reimplements the same operations Python-native; the Node.js server remains Kai-only.

---

## Architecture

Three layers sharing one core:

```
FilesClient (runtime/files/client.py)
  ├── /files slash handler (runtime/chat/telegram/files_handler.py)
  │     operator-driven: ls, read, stat, find, mv, cp, rm, mkdir, open
  └── harness tools (runtime/harness/tools/files_tool.py)
        LLM-driven read-only: list, read, stat, search
        wired via 4 skill descriptors in runtime/skills/catalog/
```

**Design decisions:**
- `write_file` exists on `FilesClient` but is not exposed via slash (inline multi-line content is awkward over Telegram) or LLM (read-only gate for autonomous use).
- Slash handler is a sync `Handler` callable — pathlib operations complete in << 100 ms; no async needed.
- `PathDenied` / `FileTooBig` always collapse to user-facing error strings, never propagate as exceptions past the handler or harness tool boundary.
- If `files.allowed_roots` is configured but empty, `FilesClient.__init__` raises `ValueError` immediately so bot startup fails fast.

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `runtime/files/__init__.py` | Create — exports `FilesClient`, `DirEntry`, `FileInfo`, `PathDenied`, `FileTooBig` |
| `runtime/files/client.py` | Create — `FilesClient` with path sandboxing |
| `runtime/chat/telegram/files_handler.py` | Create — `/files` slash handler factory |
| `runtime/harness/tools/files_tool.py` | Create — `make_files_tools(client)` harness callables |
| `runtime/skills/catalog/list_files.yaml` | Create — intent descriptor |
| `runtime/skills/catalog/read_file.yaml` | Create — intent descriptor |
| `runtime/skills/catalog/search_files.yaml` | Create — intent descriptor |
| `runtime/skills/catalog/file_info.yaml` | Create — intent descriptor |
| `runtime/config.py` | Modify — add `FilesConfig` + `files: FilesConfig` field to `AegisConfig` |
| `runtime/chat/telegram/handlers.py` | Modify — wire `/files` handler + add to `/help` catalogue |
| `runtime/chat/telegram/bot.py` | Modify — instantiate `FilesClient`, inject into harness and `/files` handler |
| `runtime/harness/adapter.py` | Modify — merge `make_files_tools` into `_TOOLS` at construction |
| `tests/test_files_client.py` | Create |
| `tests/test_files_handler.py` | Create |
| `tests/test_files_harness.py` | Create |

---

## Component Designs

### `runtime/files/client.py`

```python
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB


class PathDenied(Exception):
    """Raised when a path falls outside all allowed roots."""


class FileTooBig(Exception):
    """Raised when a file exceeds MAX_READ_BYTES."""


@dataclass(frozen=True)
class DirEntry:
    name: str
    path: str
    type: Literal["file", "directory", "other"]
    size: int
    modified: str | None  # ISO 8601 or None if stat failed


@dataclass(frozen=True)
class FileInfo:
    path: str
    type: Literal["file", "directory", "other"]
    size: int
    created: str   # ISO 8601
    modified: str  # ISO 8601
    accessed: str  # ISO 8601
    mode: str      # octal string e.g. "100644"


class FilesClient:
    def __init__(self, allowed_roots: list[Path]) -> None:
        if not allowed_roots:
            raise ValueError("FilesClient requires at least one allowed root")
        self._roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    # ── Read ops (exposed to slash + LLM) ─────────────────────────────────

    def list_dir(self, path: str, *, recursive: bool = False) -> list[DirEntry]: ...
    def read_file(self, path: str) -> str: ...
    def stat(self, path: str) -> FileInfo: ...
    def search(self, directory: str, pattern: str, *, kind: str = "any") -> list[str]: ...

    # ── Destructive ops (slash only) ───────────────────────────────────────

    def write_file(self, path: str, content: str, *, create_dirs: bool = True) -> None: ...
    def move(self, src: str, dst: str) -> None: ...
    def copy(self, src: str, dst: str) -> None: ...
    def delete(self, path: str, *, confirm: bool = False) -> None: ...
    def mkdir(self, path: str) -> None: ...
    def open_with_app(self, path: str, app: str | None = None) -> None: ...

    # ── Internal ───────────────────────────────────────────────────────────

    def _validate(self, p: str) -> Path:
        """Resolve ~ and check against allowed roots. Raises PathDenied."""
        resolved = Path(p).expanduser().resolve()
        for root in self._roots:
            if resolved == root or str(resolved).startswith(str(root) + "/"):
                return resolved
        raise PathDenied(
            f"Path '{resolved}' is outside allowed roots: "
            + ", ".join(str(r) for r in self._roots)
        )
```

**Implementation notes:**

- `list_dir`: calls `Path.iterdir()`, collects `DirEntry` for each entry; if `recursive=True` recurses into subdirectories. Silently skips entries where `stat()` fails (permission error).
- `read_file`: checks `stat().st_size` before reading; raises `FileTooBig` if over `MAX_READ_BYTES`. Reads with `encoding="utf-8", errors="replace"`.
- `stat`: returns `FileInfo` with `mode=oct(stat_result.st_mode)` (no `0o` prefix stripped to 6 chars).
- `search`: recursive walk via `Path.rglob`-equivalent manual walk; converts glob pattern to regex (`*` → `.*`, `?` → `.`, rest `re.escape`d); `kind` filters by `"file"`, `"directory"`, or `"any"`. Case-insensitive match on filename only (not full path).
- `delete`: if target is a directory and non-empty and `confirm=False`, raises `PathDenied` with a message instructing the operator to use `--confirm`. For files, `confirm` is ignored. Uses `shutil.rmtree` for directories when `confirm=True`, `Path.unlink` for files.
- `open_with_app`: calls `subprocess.run(["/usr/bin/open", path] + (["-a", app] if app else []))` with `check=True`. Raises `OSError` on failure (caller formats the error).

---

### `runtime/chat/telegram/files_handler.py`

```python
def files_handler(*, client: FilesClient) -> Handler:
    """Factory for the dispatcher. Closes over FilesClient."""
    def _handle(msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return _USAGE
        sub = cmd.args[0].strip().lower()
        tail = cmd.args[1:]
        try:
            if sub == "ls":       return _ls(client, tail)
            if sub == "read":     return _read(client, tail)
            if sub == "stat":     return _stat(client, tail)
            if sub == "find":     return _find(client, tail)
            if sub == "mv":       return _mv(client, tail)
            if sub == "cp":       return _cp(client, tail)
            if sub == "rm":       return _rm(client, tail)
            if sub == "mkdir":    return _mkdir(client, tail)
            if sub == "open":     return _open(client, tail)
            return _USAGE
        except PathDenied as exc:
            return f"Access denied: {exc}"
        except FileTooBig as exc:
            return f"File too large: {exc}"
        except OSError as exc:
            return f"Error: {exc}"
    return _handle
```

**Sub-verb specs:**

| Sub-verb | Args | Behaviour |
|----------|------|-----------|
| `ls [-r] <path>` | optional `-r` flag, then path | lists entries; `-r` makes it recursive |
| `read <path>` | path | returns file content clipped to 3 500 chars with `… (truncated N chars)` note |
| `stat <path>` | path | key: value block (type, size, created, modified, mode) |
| `find <dir> <pattern>` | dir, glob pattern | one absolute path per line; "No matches." if empty |
| `mv <src> <dst>` | two paths | `Moved: <src> → <dst>` |
| `cp <src> <dst>` | two paths | `Copied: <src> → <dst>` |
| `rm [--confirm] <path>` | optional `--confirm`, then path | deletes; non-empty dir without `--confirm` returns usage hint |
| `mkdir <path>` | path | `Created: <path>` |
| `open <path> [app]` | path, optional app name | `Opened: <path>` or `Opened: <path> with <app>` |

**`ls` output format** (one line per entry):
```
[d] Documents/  (4.1 KB, 2026-04-22)
[f] report.pdf  (2.3 MB, 2026-04-21)
[?] socket      (0 B, -)
```

**Usage string:**
```
Usage: /files <sub-command>
  ls [-r] <path>       list directory
  read <path>          read file content
  stat <path>          file metadata
  find <dir> <pattern> search by name (glob)
  mv <src> <dst>       move / rename
  cp <src> <dst>       copy
  rm [--confirm] <path> delete (--confirm required for non-empty dirs)
  mkdir <path>         create directory
  open <path> [app]    open with macOS app
```

Added to `/help` catalogue:
```python
"/files": "Filesystem access. ls, read, stat, find, mv, cp, rm, mkdir, open. Paths sandboxed to configured allowed_roots.",
```

---

### `runtime/skills/catalog/list_files.yaml`

```yaml
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

### `runtime/skills/catalog/read_file.yaml`

```yaml
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

### `runtime/skills/catalog/search_files.yaml`

```yaml
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
      enum: [file, directory, any]
  required:
    - directory
    - pattern
requires_tier1: true
```

### `runtime/skills/catalog/file_info.yaml`

```yaml
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

---

### `runtime/harness/tools/files_tool.py`

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from runtime.files.client import FilesClient, FileTooBig, PathDenied

_MAX_TOOL_CHARS = 3500


def make_files_tools(client: FilesClient) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    def files_list(args: dict[str, Any]) -> dict[str, Any]:
        entries = client.list_dir(args["path"], recursive=bool(args.get("recursive", False)))
        lines = [f"[{'d' if e.type == 'directory' else 'f' if e.type == 'file' else '?'}] {e.name}  ({e.size} B, {e.modified or '-'})" for e in entries]
        return {"result": "\n".join(lines) or "(empty directory)"}

    def files_read(args: dict[str, Any]) -> dict[str, Any]:
        content = client.read_file(args["path"])
        if len(content) > _MAX_TOOL_CHARS:
            content = content[:_MAX_TOOL_CHARS] + f"… (truncated)"
        return {"result": content}

    def files_stat(args: dict[str, Any]) -> dict[str, Any]:
        info = client.stat(args["path"])
        text = f"type: {info.type}\nsize: {info.size} B\ncreated: {info.created}\nmodified: {info.modified}\nmode: {info.mode}"
        return {"result": text}

    def files_search(args: dict[str, Any]) -> dict[str, Any]:
        matches = client.search(args["directory"], args["pattern"], kind=args.get("kind", "any"))
        return {"result": "\n".join(matches) if matches else "No matches."}

    def _wrap(fn: Callable) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            try:
                return fn(args)
            except (PathDenied, FileTooBig) as exc:
                raise RuntimeError(str(exc)) from exc  # HarnessAdapter catches Exception
        return wrapper

    return {
        "files_list": _wrap(files_list),
        "files_read": _wrap(files_read),
        "files_stat": _wrap(files_stat),
        "files_search": _wrap(files_search),
    }
```

---

### Config (`runtime/config.py`)

```python
class FilesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_roots: list[Path] = Field(
        default_factory=lambda: [
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Development",
            Path.home() / "data",
        ]
    )
```

`AegisConfig` gains `files: FilesConfig = Field(default_factory=FilesConfig)`.

`~/.aegis/config.json` example:
```json
{ "files": { "allowed_roots": ["~/Documents", "~/Projects"] } }
```

---

### `bot.py` wiring

```python
files_client = FilesClient(cfg.files.allowed_roots)

# /files slash handler
dispatcher.register("/files", files_handler(client=files_client))

# LLM harness tools
from runtime.harness.tools.files_tool import make_files_tools
harness = HarnessAdapter(tools={**_TOOLS, **make_files_tools(files_client)})
```

---

## Testing

### `tests/test_files_client.py` (uses `tmp_path`, real filesystem)

- `test_validate_allows_path_inside_root` — path inside root resolves correctly
- `test_validate_denies_path_outside_root` — `PathDenied` raised
- `test_validate_denies_path_traversal` — `../../etc/passwd` style attack denied
- `test_list_dir_flat` — creates 3 files + 1 subdir, verifies `DirEntry` list
- `test_list_dir_recursive` — verifies nested entries appear
- `test_read_file_happy` — write file, read back, content matches
- `test_read_file_too_big` — create file > MAX_READ_BYTES (mock `stat` or write sparse), `FileTooBig` raised
- `test_stat_file` — stat a file, verify type/size/mode fields
- `test_stat_directory` — stat a dir, verify type="directory"
- `test_search_glob_star` — search `*.txt`, only txt files returned
- `test_search_kind_file` — kind="file" excludes directories
- `test_search_kind_directory` — kind="directory" excludes files
- `test_search_no_matches` — returns empty list
- `test_move_file` — src disappears, dst appears with same content
- `test_copy_file` — both src and dst exist after copy
- `test_delete_file` — file gone
- `test_delete_nonempty_dir_without_confirm_raises` — `PathDenied` raised
- `test_delete_nonempty_dir_with_confirm` — directory gone
- `test_mkdir_creates_nested` — nested dirs created in one call
- `test_open_with_app_calls_subprocess` — monkeypatches `subprocess.run`, verifies argv
- `test_empty_allowed_roots_raises` — `ValueError` on construction

### `tests/test_files_handler.py` (stub `FilesClient`)

Uses a `_StubClient` dataclass with pre-set return values or configured exceptions.

- `test_ls_formats_entries` — verify `[d]`/`[f]` prefix, name, size in output
- `test_ls_recursive_flag` — `-r` in args sets `recursive=True` on stub call
- `test_read_clips_long_content` — content > 3500 chars → truncation note present
- `test_stat_shows_all_fields` — all FileInfo fields appear in output
- `test_find_no_matches` — "No matches." returned
- `test_find_results` — one path per line
- `test_mv_success_message` — "Moved:" prefix
- `test_cp_success_message` — "Copied:" prefix
- `test_rm_without_confirm_flag` — `PathDenied` raised by stub → "Access denied:" in output
- `test_rm_with_confirm_flag` — passes `confirm=True` to stub
- `test_mkdir_success` — "Created:" prefix
- `test_open_success` — "Opened:" prefix
- `test_unknown_subverb_returns_usage` — unknown arg → usage string
- `test_no_args_returns_usage` — empty args → usage string
- `test_os_error_returns_error_string` — `OSError` → "Error:" prefix

### `tests/test_files_harness.py` (stub `FilesClient`)

- `test_files_list_returns_formatted_entries`
- `test_files_list_empty_directory`
- `test_files_read_returns_content`
- `test_files_read_truncates_long_content`
- `test_files_stat_returns_key_value`
- `test_files_search_returns_paths`
- `test_files_search_no_matches`
- `test_path_denied_raises_runtime_error` — `PathDenied` from stub → `RuntimeError` (caught by `HarnessAdapter`)

All three files: `pytestmark = pytest.mark.unit`

---

## Error handling summary

| Exception | Where caught | User-visible output |
|-----------|-------------|---------------------|
| `PathDenied` | slash handler top-level `except` | `Access denied: <reason>` |
| `FileTooBig` | slash handler top-level `except` | `File too large: <reason>` |
| `OSError` / `PermissionError` | slash handler top-level `except` | `Error: <os message>` |
| `PathDenied` in harness tool | `_wrap` → re-raised as `RuntimeError` | `HarnessAdapter` returns `ToolResult(status="error")` |
| `FileTooBig` in harness tool | `_wrap` → re-raised as `RuntimeError` | same |
| Empty `allowed_roots` | `FilesClient.__init__` | `ValueError` → bot startup crash with clear message |
