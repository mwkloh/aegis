"""Harness adapter — validates ToolIntents and dispatches to local tools.

Phase 0: dispatches to in-process Python callables in `runtime/harness/tools/`.
Later phases swap this for an OpenHarness subprocess adapter without changing
callers — the **public surface is `HarnessAdapter.execute(intent) -> ToolResult`**.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contract import ToolIntent, ToolResult
from .tools import echo, respond, time_tool

# Tool registry. Keep small and explicit — no dynamic discovery in Plane 1.
DEFAULT_TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "echo": echo,
    "respond": respond,
    "time": time_tool,
}

__all__ = ["DEFAULT_TOOLS", "HarnessAdapter"]


class HarnessAdapter:
    """Single side-effect boundary for the runtime."""

    def __init__(
        self,
        tools: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self._tools = tools if tools is not None else DEFAULT_TOOLS

    def execute(self, intent: ToolIntent) -> ToolResult:
        fn = self._tools.get(intent.tool)
        if fn is None:
            return ToolResult(
                status="error",
                error=f"unknown tool: {intent.tool!r}",
            )
        try:
            payload = fn(intent.args)
        except Exception as exc:
            return ToolResult(status="error", error=f"{type(exc).__name__}: {exc}")
        return ToolResult(status="ok", payload=payload)
