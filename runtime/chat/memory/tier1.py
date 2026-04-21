"""Tier 1 chat memory — IDENTITY.md + USER.md + chat-local prefs.

Phase 7 build-order step 3. Pure file I/O, no network. Loaded on
every turn per `docs/PLAN_PHASE_7_TELEGRAM.md` §3.1.

Contract:

* Layout under `root/`:

      IDENTITY.md                 — shared across all chats
      USER.md                     — shared across all chats
      chats/<chat_id>/prefs.json  — per-chat preferences (JSON object)

* Missing files are permitted — absent content is an empty string
  (or `{}` for prefs). Tier 1 is *always present* in the turn
  context, but the operator fills IDENTITY/USER lazily.
* Malformed `prefs.json` (invalid JSON or non-object root) is a
  typed `Tier1LoadError`. Silent success would hide operator
  mistakes; this is a config surface, not a request-path dep.
* `Tier1Snapshot` is frozen + `extra="forbid"`. Carries byte counts
  so the context builder (step 4) enforces the token budget without
  re-measuring.
* The loader caches parsed content keyed by (path, mtime_ns, size).
  A stat call on each `load` refreshes the cache only when the file
  changed on disk — the hot path is three stats.
* Path-traversal defense: `chat_id` must match `[A-Za-z0-9_-]+`, so
  it cannot escape `chats/`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Tier1LoadError(RuntimeError):
    """Raised when a tier-1 file exists but is malformed."""


class Tier1Snapshot(BaseModel):
    """One turn's worth of always-on identity + user + prefs context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: str
    user: str
    prefs: dict[str, Any] = Field(default_factory=dict)
    bytes_identity: int = Field(ge=0)
    bytes_user: int = Field(ge=0)
    bytes_prefs: int = Field(ge=0)

    @property
    def total_bytes(self) -> int:
        """Sum of UTF-8 byte counts for the three sources."""
        return self.bytes_identity + self.bytes_user + self.bytes_prefs


_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def default_workspace_root() -> Path:
    """The `~/.aegis/workspace` default used by Phase 7 production."""
    return Path.home() / ".aegis" / "workspace"


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    mtime_ns: int
    size: int
    content: Any
    nbytes: int


class Tier1Loader:
    """Load IDENTITY.md / USER.md / chat-local prefs.json with mtime-based caching.

    Not thread-safe — Phase 7 runs the Telegram dispatcher
    single-threaded per chat (plan §4.2). Wrap in a lock when the
    dispatcher moves to a worker pool.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[str, _CacheEntry] = {}

    @property
    def root(self) -> Path:
        return self._root

    def load(self, chat_id: str) -> Tier1Snapshot:
        """Return the current tier-1 view for one `chat_id`."""
        if not _CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError(
                f"chat_id must match [A-Za-z0-9_-]+, got {chat_id!r}"
            )
        identity_text, nb_identity = self._read_text(self._root / "IDENTITY.md")
        user_text, nb_user = self._read_text(self._root / "USER.md")
        prefs, nb_prefs = self._read_prefs(
            self._root / "chats" / chat_id / "prefs.json"
        )
        return Tier1Snapshot(
            identity=identity_text,
            user=user_text,
            prefs=prefs,
            bytes_identity=nb_identity,
            bytes_user=nb_user,
            bytes_prefs=nb_prefs,
        )

    def invalidate(self, chat_id: str | None = None) -> None:
        """Drop cache. With no arg clears everything; else only that chat's prefs."""
        if chat_id is None:
            self._cache.clear()
            return
        if not _CHAT_ID_RE.fullmatch(chat_id):
            raise ValueError(
                f"chat_id must match [A-Za-z0-9_-]+, got {chat_id!r}"
            )
        self._cache.pop(str(self._root / "chats" / chat_id / "prefs.json"), None)

    # -- internal --------------------------------------------------------

    def _read_text(self, path: Path) -> tuple[str, int]:
        cached = self._fetch_cache(path)
        if cached is not None:
            text: str = cached.content
            return text, cached.nbytes
        try:
            stat = path.stat()
        except FileNotFoundError:
            return "", 0
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise Tier1LoadError(f"failed to read {path}: {exc}") from exc
        nbytes = len(text.encode("utf-8"))
        self._cache[str(path)] = _CacheEntry(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content=text,
            nbytes=nbytes,
        )
        return text, nbytes

    def _read_prefs(self, path: Path) -> tuple[dict[str, Any], int]:
        cached = self._fetch_cache(path)
        if cached is not None:
            parsed: dict[str, Any] = cached.content
            # Defensive copy so callers can't mutate the cache.
            return dict(parsed), cached.nbytes
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {}, 0
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise Tier1LoadError(f"failed to read {path}: {exc}") from exc
        if raw.strip() == "":
            parsed_any: Any = {}
        else:
            try:
                parsed_any = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise Tier1LoadError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(parsed_any, dict):
            raise Tier1LoadError(
                f"{path} must be a JSON object at the root, "
                f"got {type(parsed_any).__name__}"
            )
        nbytes = len(raw.encode("utf-8"))
        self._cache[str(path)] = _CacheEntry(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content=parsed_any,
            nbytes=nbytes,
        )
        return dict(parsed_any), nbytes

    def _fetch_cache(self, path: Path) -> _CacheEntry | None:
        key = str(path)
        cached = self._cache.get(key)
        if cached is None:
            return None
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._cache.pop(key, None)
            return None
        if stat.st_mtime_ns == cached.mtime_ns and stat.st_size == cached.size:
            return cached
        self._cache.pop(key, None)
        return None


__all__ = [
    "Tier1LoadError",
    "Tier1Loader",
    "Tier1Snapshot",
    "default_workspace_root",
]
