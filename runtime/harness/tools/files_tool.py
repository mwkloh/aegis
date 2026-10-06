"""Harness tool callables for filesystem access (read-only ops, plus the
guarded destructive `files_write`)."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from runtime.files.client import FilesClient, FileTooBig, PathDenied
from runtime.harness.limits import FILES_READ_PAGE_CHARS

_BAD_OFFSET = "offset must be a non-negative integer"


def _parse_offset(raw: Any) -> int:
    """A model-supplied `offset`: None means the start, numeric strings are accepted
    (models sometimes emit integers as strings), anything else is rejected."""
    if raw is None:
        return 0
    if isinstance(raw, bool):
        raise ValueError(_BAD_OFFSET)
    try:
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(_BAD_OFFSET)
            value = int(raw)
        else:
            value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(_BAD_OFFSET) from None
    if value < 0:
        raise ValueError(_BAD_OFFSET)
    return value


def _page(content: str, offset: int) -> str:
    """One page of a file, plus a notice saying where it sits and how to read on.

    This replaces a bare "… (truncated)": with the total size and the next offset a
    model can tell a file slightly too long from one 80 KB long, and can fetch the
    rest -- paying for it only when it asks. A file that fits is returned whole,
    with no notice.
    """
    total = len(content)
    if offset > 0 and offset >= total:
        return f"(offset {offset:,} is past the end; the file has {total:,} chars)"
    page = content[offset : offset + FILES_READ_PAGE_CHARS]
    end = offset + len(page)
    if offset == 0 and end >= total:
        return page
    how = (
        f"call files_read again with offset={end} to continue"
        if end < total
        else "end of file"
    )
    return f"{page}\n…[showing chars {offset + 1:,}-{end:,} of {total:,}; {how}]"


def make_files_tools(
    client: FilesClient,
) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """Return the six harness callables closed over `client`."""

    def _wrap(
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        @functools.wraps(fn)
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
        return {"result": _page(content, _parse_offset(args.get("offset")))}

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

    def files_open(args: dict[str, Any]) -> dict[str, Any]:
        path = args["path"]
        app = args.get("app")
        client.open_with_app(path, app=app if app else None)
        target = path if not app else f"{path} with {app}"
        return {"result": f"Opened {target}."}

    def files_write(args: dict[str, Any]) -> dict[str, Any]:
        result = client.write_file(args["path"], args["content"])
        suffix = " (overwrote existing file)" if result.get("overwrote") else ""
        return {"result": f"Wrote {result['bytes_written']} bytes to {result['path']}.{suffix}"}

    return {
        "files_list": _wrap(files_list),
        "files_read": _wrap(files_read),
        "files_stat": _wrap(files_stat),
        "files_search": _wrap(files_search),
        "files_open": _wrap(files_open),
        "files_write": _wrap(files_write),
    }


__all__ = ["make_files_tools"]
