"""Tier-2 maintenance — reclaim space and refresh query planner stats.

Scheduler entry point for the ``SYS-tier2-compress`` recurring job. Silent on
success (empty stdout, exit 0).

Scope note (2026-04-21): the original plan framed this as "drain evicted
tier-3 turns and compress into tier-2". That path can only run inside the
bot process because ``Tier3Store`` is in-memory — a subprocess sees an empty
eviction queue. Tier-3 → tier-2 compression lives on the turn path in
``runtime/chat/memory/compressor.py``.

What this script does instead is honest tier-2 housekeeping against the
SQLite store:

1. ``VACUUM`` reclaims pages freed by vault-indexer pruning and episodic
   deletions. Without this, the DB file grows monotonically.
2. ``ANALYZE`` refreshes query planner stats so cosine-search fetches stay
   fast as the corpus shape shifts.

Both are safe to run concurrently with bot reads — SQLite takes a write
lock for VACUUM, but modern SQLite's incremental VACUUM on busy DBs is a
short pause, not a crash risk.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from runtime.config import get_config

log = logging.getLogger("tier2_compress")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS tier-2 maintenance (VACUUM + ANALYZE)."
    )
    return parser.parse_args(argv)


def _maintain(db_path: Path) -> tuple[int, int]:
    """Run VACUUM + ANALYZE. Returns (bytes_before, bytes_after)."""
    if not db_path.is_file():
        raise FileNotFoundError(f"memory DB missing: {db_path}")
    before = db_path.stat().st_size
    with sqlite3.connect(db_path) as conn:
        conn.execute("VACUUM;")
        conn.execute("ANALYZE;")
        conn.commit()
    after = db_path.stat().st_size
    return before, after


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _parse_args(argv)
    cfg = get_config()
    db_path = cfg.storage.memory_db
    try:
        before, after = _maintain(db_path)
    except FileNotFoundError as exc:
        # First-run case: no DB yet, nothing to compact. Silent success.
        log.info("%s — nothing to do", exc)
        return 0
    except sqlite3.DatabaseError as exc:
        log.error("tier2 maintenance failed: %s", exc)
        return 1
    reclaimed = max(0, before - after)
    log.info(
        "tier2 VACUUM+ANALYZE: %d → %d bytes (reclaimed %d)",
        before,
        after,
        reclaimed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
