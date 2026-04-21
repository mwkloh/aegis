"""Per-chat skill enable/disable state.

Phase 8 §C4. Some operator chats won't want every installed skill
reachable. The loader still returns a descriptor on intent match, but
the pipeline consults ``ChatSkillState.is_enabled`` first — disabled
skills are treated as if they didn't claim the intent, so the chat
falls through to the general-answer path.

Design pins:
* **Default enabled.** A skill that has never been toggled for a
  chat is considered enabled. ``set_enabled(..., False)`` writes an
  explicit disable row; re-enabling deletes it. Keeping the table
  sparse means the common case (all skills enabled) costs zero rows
  per chat.
* **SQLite for durability.** Per-chat toggles outlive one process
  lifetime. Shared sqlite file keeps writes atomic and makes cross-
  surface (Telegram + CLI) consistency free.
* **Never raises on read.** ``is_enabled`` treats any database
  failure as "enabled" to fail open — a broken state file must
  never lock the operator out of the skill surface.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_toggles (
    chat_id  INTEGER NOT NULL,
    skill_id TEXT    NOT NULL,
    enabled  INTEGER NOT NULL,
    PRIMARY KEY (chat_id, skill_id)
) WITHOUT ROWID;
"""


class ChatSkillState:
    """Sparse per-chat toggle store. Thread-safe via sqlite locking."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- reads ---------------------------------------------------------------

    def is_enabled(self, chat_id: int, skill_id: str) -> bool:
        """Return True unless there is an explicit disable row.

        Any sqlite error fails open (returns True) — the skill surface
        is a convenience layer, not a security boundary.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT enabled FROM skill_toggles "
                    "WHERE chat_id = ? AND skill_id = ? LIMIT 1",
                    (chat_id, skill_id),
                ).fetchone()
        except sqlite3.Error:
            return True
        if row is None:
            return True
        return bool(row[0])

    def list_disabled(self, chat_id: int) -> list[str]:
        """Return the skill ids explicitly disabled for ``chat_id``."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT skill_id FROM skill_toggles "
                    "WHERE chat_id = ? AND enabled = 0 "
                    "ORDER BY skill_id",
                    (chat_id,),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [r[0] for r in rows]

    # -- writes --------------------------------------------------------------

    def set_enabled(self, chat_id: int, skill_id: str, enabled: bool) -> None:
        """Persist an explicit toggle. Re-enabling deletes the row."""
        if enabled:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM skill_toggles WHERE chat_id = ? AND skill_id = ?",
                    (chat_id, skill_id),
                )
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO skill_toggles (chat_id, skill_id, enabled) "
                "VALUES (?, ?, 0) "
                "ON CONFLICT(chat_id, skill_id) DO UPDATE SET enabled = 0",
                (chat_id, skill_id),
            )

    # -- internals -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


__all__ = ["ChatSkillState"]
