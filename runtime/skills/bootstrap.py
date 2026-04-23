"""Seed the built-in skill bundle into the sovereign workspace on first boot.

Every skill shipped with this repo lives under ``runtime/skills/_bundle/`` as a
directory-per-skill tree (``<id>/skill.yaml`` plus any co-located scripts).
At startup we copy each bundle entry into ``catalog_dir`` (default
``~/.aegis/workspace/skills/``) unless the operator already has a directory
for that skill — in which case the workspace copy wins and we never touch it.

Design pins:

* **Idempotent.** Re-seeding is a no-op for any skill already present in
  the catalog. Safe to call on every bot start.
* **Workspace always wins.** An operator editing a descriptor in the
  workspace must survive every subsequent boot. This is the same contract
  as ``~/.aegis/workspace/*.md`` — repo code never overwrites it.
* **Never raises.** A copy failure for one skill is logged and skipped so
  the bot still boots with a partially-seeded catalog.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def seed_builtin_skills(*, bundle_dir: Path, catalog_dir: Path) -> int:
    """Copy missing built-in skills into ``catalog_dir``.

    Returns the count of skill directories actually created. Already-present
    workspace directories are untouched regardless of their content.
    """
    bundle_dir = Path(bundle_dir)
    catalog_dir = Path(catalog_dir)
    if not bundle_dir.is_dir():
        return 0

    catalog_dir.mkdir(parents=True, exist_ok=True)
    inserted = 0
    for entry in sorted(bundle_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        target = catalog_dir / entry.name
        if target.exists():
            continue
        try:
            shutil.copytree(entry, target)
            inserted += 1
        except OSError:
            logger.exception(
                "skills.seed_failed",
                extra={"skill_id": entry.name, "target": str(target)},
            )
    return inserted


__all__ = ["seed_builtin_skills"]
