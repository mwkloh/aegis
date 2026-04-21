"""Physical memory layer (sqlite-vec + bge-m3).

Canonical .md files live under `~/.aegis/workspace/` and are NEVER written
by this layer. Long-term writes are proposal-only and gated by the
Improvement plane.
"""

from .tiers import Tier

__all__ = ["Tier"]
