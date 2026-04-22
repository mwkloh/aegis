"""Adapter over OpenHarness + tool implementations.

This is the **only** path to side effects in Plane 1.
"""

from .adapter import DEFAULT_TOOLS, HarnessAdapter
from .contract import ToolIntent, ToolResult

__all__ = ["DEFAULT_TOOLS", "HarnessAdapter", "ToolIntent", "ToolResult"]
