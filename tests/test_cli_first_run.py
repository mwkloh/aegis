"""`make run` on a brand-new install: the CLI must not start with an empty skill catalog.

The e2e skill tests seed the catalog by hand before building the pipeline, which is how
a fresh install shipped with a CLI that answered "I don't know how to handle intent 'echo'".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.chat.cli import build_pipeline
from runtime.config import get_config, reset_config


def test_cli_seeds_bundled_skills_into_an_empty_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("AEGIS_HOME", str(tmp_path / "home"))
    reset_config()
    try:
        cfg = get_config()
        assert not cfg.skills.catalog_dir.exists()

        pipeline = build_pipeline(config=cfg)

        assert pipeline.registry.for_intent("echo") is not None
        assert (cfg.skills.catalog_dir / "echo" / "skill.yaml").is_file()
    finally:
        reset_config()
