"""Phase 7 step 7 — auto-recall policy.

Turns a user message into a ranked `tuple[Lookup, ...]` for the
context builder. Per `docs/PLAN_PHASE_7_TELEGRAM.md` §3.3 and §3.6:

* Episodic search is `chat_id`-scoped. Vault search is global.
* Score-ranked, score-clamped to >= 0 (cosine can be negative; the
  `Lookup` contract forbids negative scores, and the context builder
  relies on score-order to decide what to drop on overflow).
* **Never raises in the request path.** Any exception from tier 2,
  the embedder, or the vault body loader is caught and the failing
  source contributes zero lookups. The policy returns `()` as the
  ultimate fallback — per §2.8 stub-on-failure — because the
  context builder must always produce a usable `TurnContext`.
* Vault bodies are pulled through an injected `VaultBodyLoader` so
  the policy never has to know about filesystem paths. If no loader
  is wired, vault hits surface as bookmark-style pointers (`[vault
  pointer: <rel_path>]`) so the trail survives until the body plane
  is connected.
"""
from __future__ import annotations

from typing import Protocol

from runtime.chat.memory.context_builder import Lookup
from runtime.chat.memory.tier2 import Tier2Store

DEFAULT_TOP_EPISODIC = 3
DEFAULT_TOP_VAULT = 2


class VaultBodyLoader(Protocol):
    """Resolves a vault `rel_path` to its body text.

    Implementations MUST be deterministic and side-effect-free.
    Raising is allowed but must mean "skip this hit" — the recall
    policy catches every exception and drops the hit rather than
    propagating. Never block the request path.
    """

    def load(self, rel_path: str) -> str: ...


class RecallPolicy:
    """Queries tier 2 and shapes the result into a ranked lookup list."""

    def __init__(
        self,
        *,
        tier2: Tier2Store,
        vault_loader: VaultBodyLoader | None = None,
        top_episodic: int = DEFAULT_TOP_EPISODIC,
        top_vault: int = DEFAULT_TOP_VAULT,
    ) -> None:
        if top_episodic < 0:
            raise ValueError("top_episodic must be >= 0")
        if top_vault < 0:
            raise ValueError("top_vault must be >= 0")
        self._tier2 = tier2
        self._vault_loader = vault_loader
        self._top_episodic = top_episodic
        self._top_vault = top_vault

    def recall(
        self,
        chat_id: str,
        user_text: str,
        *,
        vault_label: str | None = None,
    ) -> tuple[Lookup, ...]:
        """Return ranked lookups for one turn. Never raises."""
        if not chat_id or not user_text.strip():
            return ()
        episodic = self._recall_episodic(chat_id, user_text)
        vault = self._recall_vault(user_text, vault_label)
        lookups = list(episodic) + list(vault)
        lookups.sort(key=lambda lk: lk.score, reverse=True)
        return tuple(lookups)

    def _recall_episodic(self, chat_id: str, query: str) -> tuple[Lookup, ...]:
        if self._top_episodic == 0:
            return ()
        try:
            hits = self._tier2.search_episodic(
                chat_id=chat_id, query=query, top_k=self._top_episodic
            )
        except Exception:
            return ()
        out: list[Lookup] = []
        for hit in hits:
            ended = hit.record.ended_at.isoformat()
            origin = f"episodic:{hit.record.chat_id}:{ended}"
            out.append(
                Lookup(
                    kind="episodic",
                    text=hit.record.summary,
                    score=max(hit.score, 0.0),
                    origin=origin,
                )
            )
        return tuple(out)

    def _recall_vault(
        self, query: str, label_filter: str | None
    ) -> tuple[Lookup, ...]:
        if self._top_vault == 0:
            return ()
        try:
            hits = self._tier2.search_vault(
                query=query,
                top_k=self._top_vault,
                label_filter=label_filter,
            )
        except Exception:
            return ()
        out: list[Lookup] = []
        for hit in hits:
            body = self._load_body(hit.record.rel_path)
            origin = f"vault:{hit.record.rel_path}"
            out.append(
                Lookup(
                    kind="vault",
                    text=body,
                    score=max(hit.score, 0.0),
                    origin=origin,
                )
            )
        return tuple(out)

    def _load_body(self, rel_path: str) -> str:
        if self._vault_loader is None:
            return f"[vault pointer: {rel_path}]"
        try:
            body = self._vault_loader.load(rel_path)
        except Exception:
            return f"[vault pointer: {rel_path}]"
        if not body:
            return f"[vault pointer: {rel_path}]"
        return body


__all__ = [
    "DEFAULT_TOP_EPISODIC",
    "DEFAULT_TOP_VAULT",
    "RecallPolicy",
    "VaultBodyLoader",
]
