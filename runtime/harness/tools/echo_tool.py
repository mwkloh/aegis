"""Trivial echo tool — proves the runtime → harness → tool round-trip works."""
from __future__ import annotations

from typing import Any


def echo(args: dict[str, Any]) -> dict[str, Any]:
    """Return the message verbatim. Used by the walking-skeleton e2e test."""
    message = args.get("message", "")
    if not isinstance(message, str):
        raise TypeError("echo: 'message' must be a string")
    return {"echoed": message, "length": len(message)}
