"""First-run provisioning: what a stranger gets from `make bootstrap` on a fresh clone.

Every test points `AEGIS_ROOT` at a tmp dir -- the same variable the runtime reads -- so
none of them can touch the real `~/.aegis`.
"""
from __future__ import annotations

import json
import shutil
import subprocess  # nosec
from pathlib import Path

import pytest

from runtime.config import _coerce, _coerce_model_profiles
from scripts import bootstrap_aegis_home as bootstrap

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "templates" / "workspace"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / ".aegis"
    monkeypatch.setenv("AEGIS_ROOT", str(target))
    monkeypatch.delenv("AEGIS_HOME", raising=False)
    return target


def test_repo_ships_every_canonical_template() -> None:
    for name in bootstrap.CANONICAL_FILES:
        assert (TEMPLATE / name).is_file(), f"templates/workspace/{name} is missing"
    assert (TEMPLATE / "memory" / "MEMORY.md").is_file()


def test_templates_are_not_swallowed_by_gitignore() -> None:
    # `.gitignore` blocks stray `workspace/` dirs; that rule once also matched
    # `templates/workspace/`, so a clean clone had no starter files at all.
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    for rel in ("IDENTITY.md", "memory/MEMORY.md"):
        result = subprocess.run(  # fixed argv, no shell
            [git, "check-ignore", "-q", str(TEMPLATE / rel)],
            cwd=REPO, capture_output=True, check=False,
        )
        assert result.returncode == 1, f"templates/workspace/{rel} is gitignored"


def test_templates_carry_nothing_operator_specific() -> None:
    for path in TEMPLATE.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path
        assert "@gmail" not in text, path


def test_bootstrap_provisions_a_fresh_root(root: Path) -> None:
    assert bootstrap.main([]) == 0
    home = root / "workspace"
    for name in bootstrap.CANONICAL_FILES:
        assert (home / name).is_file(), name
    assert (home / "memory" / "MEMORY.md").is_file()
    assert (root / "config.json").is_file()
    assert (root / ".env").stat().st_mode & 0o777 == 0o600


def test_bootstrap_seeds_the_bundled_skills(root: Path) -> None:
    bootstrap.main([])

    assert (root / "workspace" / "skills" / "echo" / "skill.yaml").is_file()


def test_bootstrap_keeps_an_operators_edited_skill(root: Path) -> None:
    edited = root / "workspace" / "skills" / "echo"
    edited.mkdir(parents=True)
    (edited / "skill.yaml").write_text("mine: true\n", encoding="utf-8")

    bootstrap.main([])

    assert (edited / "skill.yaml").read_text(encoding="utf-8") == "mine: true\n"


def test_bootstrap_never_overwrites_existing_files(root: Path) -> None:
    (root / "workspace").mkdir(parents=True)
    (root / "workspace" / "USER.md").write_text("mine", encoding="utf-8")
    (root / ".env").write_text("MODEL_FAST=mine\n", encoding="utf-8")
    (root / "config.json").write_text('{"mine": true}', encoding="utf-8")

    bootstrap.main([])
    bootstrap.main([])

    assert (root / "workspace" / "USER.md").read_text(encoding="utf-8") == "mine"
    assert (root / ".env").read_text(encoding="utf-8") == "MODEL_FAST=mine\n"
    assert (root / "config.json").read_text(encoding="utf-8") == '{"mine": true}'


def test_starter_config_only_holds_keys_the_runtime_reads(root: Path) -> None:
    bootstrap.main([])
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))

    # Models come from `.env` (MODEL_*); a `models` block here would be silently ignored.
    assert "models" not in cfg
    kwargs = _coerce({}, cfg)
    assert kwargs["harness"].multi_step is True
    assert kwargs["telegram"].enabled is False


def test_starter_config_runs_gemma_in_native_mode(root: Path) -> None:
    bootstrap.main([])
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))

    profiles = _coerce_model_profiles(cfg["modelProfiles"])
    assert profiles["gemma4:e4b"].harness_mode == "native"


def test_env_template_defines_every_key_the_doctor_requires() -> None:
    from scripts.doctor import REQUIRED_ENV_KEYS

    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for key in REQUIRED_ENV_KEYS:
        assert f"\n{key}=" in text, f"{key} is required by the doctor but absent from .env.example"


def test_bootstrap_tells_the_operator_what_to_do_next(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bootstrap.main([])
    out = capsys.readouterr().out
    assert "ollama pull" in out
    assert "make doctor" in out
