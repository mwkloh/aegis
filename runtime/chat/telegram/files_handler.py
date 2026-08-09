"""`/files` slash handler — operator-driven filesystem access via FilesClient."""
from __future__ import annotations

from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.files.client import FilesClient, FileTooBig, PathDenied

_MAX_READ_CHARS = 3500
_FIND_MIN_ARGS = 2  # <dir> <pattern>
_MV_MIN_ARGS = 2  # <src> <dst>
_CP_MIN_ARGS = 2  # <src> <dst>
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
            if sub == "ls":
                return _ls(client, tail)
            if sub == "read":
                return _read(client, tail)
            if sub == "stat":
                return _stat(client, tail)
            if sub == "find":
                return _find(client, tail)
            if sub == "mv":
                return _mv(client, tail)
            if sub == "cp":
                return _cp(client, tail)
            if sub == "rm":
                return _rm(client, tail)
            if sub == "mkdir":
                return _mkdir(client, tail)
            if sub == "open":
                return _open(client, tail)
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
    if len(args) < _FIND_MIN_ARGS:
        return "Usage: /files find <dir> <pattern>"
    matches = client.search(args[0], args[1])
    return "\n".join(matches) if matches else "No matches."


def _mv(client: FilesClient, args: tuple[str, ...]) -> str:
    if len(args) < _MV_MIN_ARGS:
        return "Usage: /files mv <src> <dst>"
    client.move(args[0], args[1])
    return f"Moved: {args[0]} → {args[1]}"


def _cp(client: FilesClient, args: tuple[str, ...]) -> str:
    if len(args) < _CP_MIN_ARGS:
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
