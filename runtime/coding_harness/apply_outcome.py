"""Typed envelope for one apply attempt.

An `ApplyOutcome` is the validated record produced by the applier for a
single `CT-NNN`: which branch was created, the verdict from running
`git apply` + `make test`, the captured test-output tail, and provenance.
The applier NEVER raises on subprocess failure — it returns an outcome
with the appropriate verdict instead.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ApplyVerdict = Literal[
    "applied_clean",          # patch applied + tests passed
    "applied_test_failed",    # patch applied + tests failed (or timed out)
    "apply_conflict",         # `git apply --check` or `git apply` rejected
    "precondition_failed",    # dirty tree, protected branch, etc.
]

_MAX_BRANCH = 128
_MAX_REASON = 512
_MAX_PATCH_PATH = 512
_MAX_STDOUT_TAIL = 8192      # 8 KB cap on captured test output


class ApplyOutcome(BaseModel):
    """One apply attempt. Always returned — never raised."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ct_id: str = Field(min_length=4, max_length=16)
    imp_id: str = Field(min_length=12, max_length=12)
    verdict: ApplyVerdict
    reason: str = Field(default="", max_length=_MAX_REASON)
    branch: str = Field(default="", max_length=_MAX_BRANCH)
    patch_path: str = Field(default="", max_length=_MAX_PATCH_PATH)
    tests_exit_code: int | None = None
    tests_duration_s: float | None = Field(default=None, ge=0.0)
    tests_stdout_tail: str = Field(default="", max_length=_MAX_STDOUT_TAIL)
    applied_at: datetime
