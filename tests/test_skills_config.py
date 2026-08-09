"""SkillsConfig — catalog directory + bundle-source knobs."""
from __future__ import annotations

from runtime.config import AegisConfig, SkillsConfig


def test_default_catalog_dir_is_under_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGIS_ROOT", str(tmp_path))
    monkeypatch.delenv("AEGIS_HOME", raising=False)
    cfg = AegisConfig()
    assert cfg.skills.catalog_dir == tmp_path / "workspace" / "skills"


def test_catalog_dir_respects_aegis_home(monkeypatch, tmp_path):
    custom = tmp_path / "custom-workspace"
    monkeypatch.setenv("AEGIS_HOME", str(custom))
    cfg = AegisConfig()
    assert cfg.skills.catalog_dir == custom / "skills"


def test_bundle_dir_points_at_repo():
    cfg = SkillsConfig()
    # Default bundle dir resolves relative to runtime/skills/_bundle
    assert cfg.bundle_dir.name == "_bundle"
    assert cfg.bundle_dir.parent.name == "skills"
