"""Adapter over OpenHarness + tool implementations.

This is the **only** path to side effects in Plane 1.
"""

from .adapter import HarnessAdapter
from .contract import ToolIntent, ToolResult

__all__ = ["HarnessAdapter", "ToolIntent", "ToolResult"]
