"""Idempotently provision `~/.aegis/` (canon, env, config).

Safe to run repeatedly. Never overwrites existing files; only fills gaps.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

AEGIS_ROOT = Path.home() / ".aegis"
AEGIS_HOME = AEGIS_ROOT / "workspace"

CANONICAL_FILES = (
    "AGENTS.md",
    "USER.md",
    "IDENTITY.md",
    "SOUL.md",
    "HEARTBEAT.md",
    "SKILLS_INDEX.md",
)


def ensure_dirs() -> list[Path]:
    created: list[Path] = []
    for d in (AEGIS_HOME, AEGIS_HOME / "memory", AEGIS_HOME / "sessions", AEGIS_HOME / "skills"):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


def ensure_canon(template_dir: Path) -> list[Path]:
    """Copy any missing canonical .md files from the template directory."""
    copied: list[Path] = []
    for name in CANONICAL_FILES:
        target = AEGIS_HOME / name
        source = template_dir / name
        if target.exists() or not source.is_file():
            continue
        shutil.copy2(source, target)
        copied.append(target)
    memory_md = AEGIS_HOME / "memory" / "MEMORY.md"
    memory_src = template_dir / "memory" / "MEMORY.md"
    if not memory_md.exists() and memory_src.is_file():
        shutil.copy2(memory_src, memory_md)
        copied.append(memory_md)
    return copied


def ensure_env(env_example: Path) -> Path | None:
    target = AEGIS_ROOT / ".env"
    if target.exists() or not env_example.is_file():
        return None
    AEGIS_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_example, target)
    target.chmod(0o600)
    return target


def ensure_config() -> Path | None:
    target = AEGIS_ROOT / "config.json"
    if target.exists():
        return None
    AEGIS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "providers": {"ollama": {"baseUrl": "http://127.0.0.1:11434"}},
        "models": {
            "fast": "gemma4:e2b",
            "smart": "openrouter/auto",
            "reflection": "gemma4:e4b",
        },
        "telegram": {"enabled": False, "userAllowlist": []},
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    default_template = Path(__file__).parent.parent / "templates" / "workspace"
    template_dir = Path(argv[0]).expanduser() if argv else default_template
    print(f"AEGIS bootstrap → {AEGIS_ROOT}")
    created = ensure_dirs()
    for d in created:
        print(f"  + dir   {d}")
    if template_dir.is_dir():
        for f in ensure_canon(template_dir):
            print(f"  + canon {f}")
    else:
        print(f"  · template dir {template_dir} not found — skipping canon copy")
    env_example = Path(__file__).parent.parent / ".env.example"
    env = ensure_env(env_example)
    if env:
        print(f"  + env   {env} (chmod 600)")
    cfg = ensure_config()
    if cfg:
        print(f"  + cfg   {cfg}")
    print("done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
