"""Tool-intent contract — Pydantic models the LLM/skill emits and the harness consumes.

Per ARCHITECTURE.md §6: the LLM never executes side effects directly.
It emits a **ToolIntent** (validated here); the harness validates and dispatches.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolIntent(BaseModel):
    """Structured intent — what the runtime asks the harness to do."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str = Field(min_length=1, max_length=64, description="Tool identifier in registry.")
    args: dict[str, Any] = Field(default_factory=dict)
    skill_id: str = Field(min_length=1, max_length=64, description="Skill that emitted this.")
    rationale: str = Field(default="", max_length=2048, description="Free-text reason; logged.")


class ToolResult(BaseModel):
    """Result the harness returns. Status is explicit; payload is opaque."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
