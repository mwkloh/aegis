"""Harness tool callables for read-only filesystem access."""
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
