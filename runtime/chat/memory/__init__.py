"""Phase 7 chat memory package.

Three tiers per `docs/PLAN_PHASE_7_TELEGRAM.md` §3:

* Tier 1 — IDENTITY/USER + chat-local prefs (always loaded)
* Tier 2 — episodic + vault notes (RAG, on-demand)
* Tier 3 — recent raw turns (rolling window)
"""
from __future__ import annotations

from .cold_storage import (
    ColdStorageError,
    ColdStorageMismatch,
    ColdStorageMissing,
    ColdStorageRead,
    ColdStorageReader,
)
from .compressor import CompressionResult, Compressor, Summarizer
from .context_builder import (
    DEFAULT_TURN_BUDGET_BYTES,
    ContextBuilder,
    Lookup,
    LookupKind,
    TurnContext,
)
from .recall import (
    DEFAULT_TOP_EPISODIC,
    DEFAULT_TOP_VAULT,
    RecallPolicy,
    VaultBodyLoader,
)
from .tier1 import Tier1Loader, Tier1LoadError, Tier1Snapshot, default_workspace_root
from .tier2 import (
    ColdRef,
    EpisodicHit,
    EpisodicMemory,
    Tier2Store,
    VaultHit,
    VaultNote,
)
from .tier3 import TIER3_KEEP_TURNS, Tier3Store, Turn

__all__ = [
    "DEFAULT_TOP_EPISODIC",
    "DEFAULT_TOP_VAULT",
    "DEFAULT_TURN_BUDGET_BYTES",
    "TIER3_KEEP_TURNS",
    "ColdRef",
    "ColdStorageError",
    "ColdStorageMismatch",
    "ColdStorageMissing",
    "ColdStorageRead",
    "ColdStorageReader",
    "CompressionResult",
    "Compressor",
    "ContextBuilder",
    "EpisodicHit",
    "EpisodicMemory",
    "Lookup",
    "LookupKind",
    "RecallPolicy",
    "Summarizer",
    "Tier1LoadError",
    "Tier1Loader",
    "Tier1Snapshot",
    "Tier2Store",
    "Tier3Store",
    "Turn",
    "TurnContext",
    "VaultBodyLoader",
    "VaultHit",
    "VaultNote",
    "default_workspace_root",
]
