"""Idempotently provision `~/.aegis/` (canon, env, config).

Safe to run repeatedly. Never overwrites existing files; only fills gaps. Honours
`AEGIS_ROOT` / `AEGIS_HOME` exactly as the runtime does, so an operator who relocates
the root gets the same layout from bootstrap, doctor and the bot.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from runtime.skills.bootstrap import seed_builtin_skills

CANONICAL_FILES = (
    "AGENTS.md",
    "USER.md",
    "IDENTITY.md",
    "SOUL.md",
    "HEARTBEAT.md",
    "SKILLS_INDEX.md",
)

# Only keys `runtime.config` reads from config.json. Models are chosen in `.env`
# (MODEL_*), so a `models` block here would be silently ignored. `gemma4:e4b` runs in
# native mode because the 2026-10 ablation measured it ahead of the planner on the
# `-mlx` build of this model (docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md,
# sections 22-23); the plain tag is the same weights but has not been measured itself.
STARTER_CONFIG: dict[str, object] = {
    "telegram": {"enabled": False, "userAllowlist": []},
    "harness": {"multi_step": True, "max_steps": 5},
    "modelProfiles": {"gemma4:e4b": {"harness_mode": "native"}},
}

# What the starter `.env` asks Ollama for; printed so the operator can pull them.
STARTER_MODELS = ("gemma4:e2b", "gemma4:e4b")


def aegis_root() -> Path:
    return Path(os.environ.get("AEGIS_ROOT", str(Path.home() / ".aegis"))).expanduser()


def aegis_home(root: Path) -> Path:
    return Path(os.environ.get("AEGIS_HOME", str(root / "workspace"))).expanduser()


def ensure_dirs(home: Path) -> list[Path]:
    created: list[Path] = []
    for d in (home, home / "memory", home / "sessions", home / "skills"):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
    return created


def ensure_canon(template_dir: Path, home: Path) -> list[Path]:
    """Copy any missing canonical .md files from the template directory."""
    copied: list[Path] = []
    for name in CANONICAL_FILES:
        target = home / name
        source = template_dir / name
        if target.exists() or not source.is_file():
            continue
        shutil.copy2(source, target)
        copied.append(target)
    memory_md = home / "memory" / "MEMORY.md"
    memory_src = template_dir / "memory" / "MEMORY.md"
    if not memory_md.exists() and memory_src.is_file():
        shutil.copy2(memory_src, memory_md)
        copied.append(memory_md)
    return copied


def ensure_env(root: Path, env_example: Path) -> Path | None:
    target = root / ".env"
    if target.exists() or not env_example.is_file():
        return None
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_example, target)
    target.chmod(0o600)
    return target


def ensure_config(root: Path) -> Path | None:
    target = root / "config.json"
    if target.exists():
        return None
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(STARTER_CONFIG, indent=2) + "\n", encoding="utf-8")
    return target


def _print_next_steps() -> None:
    print("\nNext steps:")
    print("  1. Install Ollama (https://ollama.com) and start it: ollama serve")
    for step, model in enumerate(STARTER_MODELS, start=2):
        print(f"  {step}. ollama pull {model}")
    last = 2 + len(STARTER_MODELS)
    print(f"  {last}. make doctor    # re-runnable health check")
    print(f"  {last + 1}. make run       # chat in the terminal")
    print("  Optional: add TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOW_FROM to the .env above.")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = aegis_root()
    home = aegis_home(root)
    repo = Path(__file__).resolve().parent.parent
    template_dir = Path(argv[0]).expanduser() if argv else repo / "templates" / "workspace"
    print(f"AEGIS bootstrap → {root}")
    for d in ensure_dirs(home):
        print(f"  + dir   {d}")
    if template_dir.is_dir():
        for f in ensure_canon(template_dir, home):
            print(f"  + canon {f}")
    else:
        print(f"  · template dir {template_dir} not found — skipping canon copy")
    seeded = seed_builtin_skills(
        bundle_dir=repo / "runtime" / "skills" / "_bundle", catalog_dir=home / "skills"
    )
    if seeded:
        print(f"  + skills {seeded} bundled skill(s) → {home / 'skills'}")
    env = ensure_env(root, repo / ".env.example")
    if env:
        print(f"  + env   {env} (chmod 600)")
    cfg = ensure_config(root)
    if cfg:
        print(f"  + cfg   {cfg}")
    print("done.")
    _print_next_steps()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
