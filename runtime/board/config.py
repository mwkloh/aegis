"""Pydantic config for the `/board` multi-panelist feature.

Frozen throughout — these objects travel through `AegisConfig` which
treats every child as immutable. `output_dir` is expanded eagerly so
downstream callers never need to think about `~`.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PanelistConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=32)
    persona: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=32)
    persona: str = Field(
        default=(
            "You are a synthesis assistant. Given multiple expert perspectives "
            "on a question, identify areas of agreement, key tensions, and "
            "produce a concise bottom-line summary."
        ),
        min_length=1,
        max_length=4000,
    )
    max_tokens: int = Field(default=512, ge=1, le=8192)


class BoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    panelists: list[PanelistConfig] = Field(default_factory=list)
    synthesis: SynthesisConfig | None = None
    output_dir: Path = Field(
        default_factory=lambda: Path.home() / ".aegis" / "boards"
    )
    excerpt_chars: int = Field(default=300, ge=50, le=1000)
    panelist_timeout_s: float = Field(default=60.0, ge=5.0, le=300.0)

    @field_validator("output_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return Path(v).expanduser()
