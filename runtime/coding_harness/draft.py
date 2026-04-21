"""Typed envelope for one coding-harness draft.

A `Draft` is the validated package the coder produces for a single
`CT-NNN`: summary, opaque unified diff text, test notes, rollback,
plus provenance (model, timestamp, status). The diff is treated as
an opaque blob — we never parse or apply it; a human reviews the
rendered patch.md.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DraftStatus = Literal["ok", "stub", "refused"]

_MAX_SUMMARY = 512
_MAX_DIFF = 32_000
_MAX_NOTES = 2_048
_MAX_ROLLBACK = 2_048
_MAX_REASON = 256


class Draft(BaseModel):
    """One coding-harness draft. Validated at every boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ct_id: str = Field(min_length=4, max_length=16)
    imp_id: str = Field(min_length=12, max_length=12)
    model: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=0, max_length=_MAX_SUMMARY)
    unified_diff: str = Field(min_length=0, max_length=_MAX_DIFF)
    test_notes: str = Field(min_length=0, max_length=_MAX_NOTES)
    rollback: str = Field(min_length=0, max_length=_MAX_ROLLBACK)
    drafted_at: datetime
    status: DraftStatus
    reason: str = Field(default="", max_length=_MAX_REASON)
