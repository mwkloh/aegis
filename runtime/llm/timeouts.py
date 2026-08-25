"""Provider-agnostic read-timeout override, for the eval harness.

A single fixed read timeout is a different *token* budget for every model --
measured 2026-08-24, warm throughput spans 117.8 tok/s (`gemma4:e2b-mlx`) to
15.0 (`qwen3-vl:4b`). Grading capability against it conflates "could not" with
"was cut off", which is how `qwen3-vl:4b` came to be published as a flat 0%
when it in fact reaches 40% TGC given room to finish.

Shared rather than per-client on purpose: an override honoured by only one
provider would produce a results table mixing capability-measured rows with
budget-capped ones and present them as comparable.

Production is untouched -- with no active override each client keeps its own
shipped default. See
`docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md`.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_override: ContextVar[float | None] = ContextVar("llm_read_timeout_override", default=None)


@contextmanager
def read_timeout_override(seconds: float) -> Iterator[None]:
    """Apply `seconds` as the read timeout for every client call in this block."""
    token = _override.set(seconds)
    try:
        yield
    finally:
        _override.reset(token)


def resolve_read_timeout(default: float) -> float:
    """The read timeout a client should use: the override if set, else its default."""
    override = _override.get()
    return default if override is None else override


__all__ = ["read_timeout_override", "resolve_read_timeout"]
