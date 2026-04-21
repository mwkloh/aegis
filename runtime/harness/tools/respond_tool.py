"""Pass-through tool that returns a pre-composed message.

Used by Tier 1 skills (and the Tier-1-unavailable fallback) where the reply
text has already been composed upstream. The harness stays the only side-effect
boundary, so even a "just say this" reply goes through a tool call.
"""
from __future__ import annotations

from typing import Any

_MAX_MESSAGE_CHARS: int = 8192


def respond(args: dict[str, Any]) -> dict[str, Any]:
    message = args.get("message", "")
    if not isinstance(message, str):
        raise TypeError("respond: 'message' must be a string")
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS]
    return {"message": message, "length": len(message)}
