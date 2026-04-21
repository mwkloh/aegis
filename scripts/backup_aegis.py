"""Backup script for ~/.aegis/workspace/.

Usage:
    python -m scripts.backup_aegis

Environment variables:
    AEGIS_ROOT        — root of AEGIS config (default: ~/.aegis)
    AEGIS_BACKUP_DEST — destination directory for the backup (required, no default)

The script uses rsync -av --delete to mirror $AEGIS_ROOT/workspace/ to
$AEGIS_BACKUP_DEST. The destination directory is created if it does not exist.

Exit codes:
    0 — success
    1 — missing AEGIS_BACKUP_DEST env var, or rsync returned non-zero
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _ts() -> str:
    """Return a UTC timestamp prefix for log lines."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    # --- Read configuration from environment ---
    aegis_root = Path(os.environ.get("AEGIS_ROOT", "~/.aegis")).expanduser()
    backup_dest_raw = os.environ.get("AEGIS_BACKUP_DEST")

    if not backup_dest_raw:
        print(
            f"[{_ts()}] ERROR: AEGIS_BACKUP_DEST is not set. "
            "Please set it to the desired backup destination path "
            "(e.g. /Volumes/ExternalDrive/aegis-backup or ~/Backups/aegis)."
        )
        return 1

    source = aegis_root / "workspace"
    dest = Path(backup_dest_raw).expanduser()

    print(f"[{_ts()}] Starting AEGIS workspace backup")
    print(f"[{_ts()}]   source : {source}/")
    print(f"[{_ts()}]   dest   : {dest}/")

    # --- Ensure destination directory exists ---
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{_ts()}] Destination directory ensured: {dest}")

    # --- Run rsync ---
    cmd = ["rsync", "-av", "--delete", f"{source}/", f"{dest}/"]
    print(f"[{_ts()}] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout, end="")

    if result.returncode != 0:
        print(f"[{_ts()}] ERROR: rsync failed with exit code {result.returncode}")
        if result.stderr:
            print(result.stderr, end="")
        return 1

    print(
        f"[{_ts()}] SUCCESS: backup complete — "
        f"source={source}/ dest={dest}/ finished_at={_ts()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
