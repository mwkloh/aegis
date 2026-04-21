"""Memory tier enum + read-API surface (Phase 0 stubs).

Per docs/ARCHITECTURE.md §8.1.
"""
from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    PREFERENCES = "preferences"
    IDENTITY = "identity"
    EPISODIC = "episodic"
    EXTERNAL = "external"
    EXECUTION = "execution"


# Phase 0: no read APIs implemented. They land in Phase 2 alongside sqlite-vec
# and bge-m3 wiring. This module exists today so tests can assert the boundary.
