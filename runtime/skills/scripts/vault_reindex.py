"""Vault reindex — walk configured vault sources, upsert changed notes, prune stale rows.

Scheduler entry point for the ``SYS-vault-reindex`` recurring job. Always silent
on success (empty stdout, exit 0) so the scheduler push layer pushes nothing.
Counters are logged to stderr for operator observability via ``journalctl`` /
event log.

Embedder selection mirrors ``build_chat_pipeline`` — try the real
``Bgem3Embedder`` (dim=1024 via Ollama loopback) first, fall back to
``FakeEmbedder`` when bge-m3 isn't reachable. This keeps the rows we write
readable by chat recall (which uses the same logic).
"""
from __future__ import annotations

import argparse
import logging
import sys

from memory.embeddings import build_embedder
from runtime.chat.memory.tier2 import Tier2Store
from runtime.chat.memory.vault_indexer import ReindexResult, VaultIndexer
from runtime.config import get_config

log = logging.getLogger("vault_reindex")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AEGIS vault reindex (full or label-scoped)."
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Restrict reindex to one VaultSource label. Default: all sources.",
    )
    return parser.parse_args(argv)


def _format_summary(result: ReindexResult, label: str | None) -> str:
    scope = f"label={label!r}" if label else "all sources"
    return (
        f"reindex {scope}: added={result.added} updated={result.updated} "
        f"skipped={result.skipped} pruned={result.pruned} "
        f"errors={len(result.errors)}"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args(argv)
    cfg = get_config()
    if not cfg.vault_indexing.is_enabled():
        log.info("vault_indexing disabled; nothing to do")
        return 0

    embedder = build_embedder()
    tier2 = Tier2Store(cfg.storage.memory_db, embedder)
    indexer = VaultIndexer(tier2=tier2, config=cfg.vault_indexing)
    result = indexer.reindex(only_label=args.label)
    log.info(_format_summary(result, args.label))
    for err in result.errors:
        log.warning("reindex error: %s", err)

    if result.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
