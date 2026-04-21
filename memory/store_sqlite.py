"""sqlite vector + KV store.

Schema lives here; the chat-memory tiers (`runtime/chat/memory/tier2.py`)
own the read/write APIs on top.

Schema versions:

* **v1** — Phase 0 stub. Just a `schema_version` table. No data tables.
* **v2** — Phase 7 step 2. Adds the four tier-2 tables:
  `episodic_memory`, `vault_note`, `episodic_embedding`,
  `vault_embedding`. Embeddings are raw float32 BLOBs (atamai
  pattern), small at AEGIS scale (<10k vectors), zero extension load.
* **v3** — Phase 10 Track A2. Adds `scheduled_jobs` for the
  autonomous scheduler. Additive — no touch to tier-2 tables.

Migration is idempotent. `ensure(db_path)` always brings a DB to the
latest version, regardless of where it started.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

SCHEMA_VERSION = 3

_Migration = Callable[[sqlite3.Connection], None]


def ensure(db_path: Path) -> Path:
    """Open / create the DB and migrate to `SCHEMA_VERSION`.

    Safe to call repeatedly; each migration step checks before running.
    """
    db_path = Path(db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )
        cur = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        current = int(cur.fetchone()[0])
        for target_version, migration in _MIGRATIONS:
            if current < target_version:
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_version(version) VALUES (?)",
                    (target_version,),
                )
                current = target_version
        conn.commit()
    return db_path


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    """v0 → v1. The Phase 0 stub had no data tables — just stamp the version."""
    del conn  # nothing to create; ensure() stamps the version


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """Add tier-2 episodic + vault tables. Foreign-key cascades on delete."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id         TEXT NOT NULL,
            started_at      TEXT NOT NULL,
            ended_at        TEXT NOT NULL,
            summary         TEXT NOT NULL,
            decisions_cited TEXT NOT NULL DEFAULT '[]',
            imp_ids_cited   TEXT NOT NULL DEFAULT '[]',
            cold_session_id TEXT,
            cold_jsonl_path TEXT,
            cold_turn_start INTEGER,
            cold_turn_end   INTEGER,
            cold_sha256     TEXT
        );
        CREATE INDEX IF NOT EXISTS episodic_memory_chat_idx
            ON episodic_memory (chat_id);

        CREATE TABLE IF NOT EXISTS vault_note (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            rel_path    TEXT NOT NULL UNIQUE,
            label       TEXT,
            priority    REAL NOT NULL DEFAULT 1.0,
            mtime       TEXT NOT NULL,
            body_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS vault_note_label_idx
            ON vault_note (label);

        CREATE TABLE IF NOT EXISTS episodic_embedding (
            episodic_id INTEGER PRIMARY KEY
                REFERENCES episodic_memory(id) ON DELETE CASCADE,
            dim         INTEGER NOT NULL,
            vec         BLOB    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vault_embedding (
            vault_id INTEGER PRIMARY KEY
                REFERENCES vault_note(id) ON DELETE CASCADE,
            dim      INTEGER NOT NULL,
            vec      BLOB    NOT NULL
        );
        """
    )


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """Add `scheduled_jobs` for Phase 10. Additive — no FK into tier-2 tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            id          TEXT PRIMARY KEY,
            cron_expr   TEXT NOT NULL,
            skill       TEXT NOT NULL,
            args_json   TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            created_by  INTEGER NOT NULL,
            last_run_at TEXT,
            last_status TEXT,
            paused      INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS scheduled_jobs_created_idx
            ON scheduled_jobs (created_at, id);
        """
    )


_MIGRATIONS: tuple[tuple[int, _Migration], ...] = (
    (1, _migrate_to_v1),
    (2, _migrate_to_v2),
    (3, _migrate_to_v3),
)
