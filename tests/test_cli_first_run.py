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


def test_cli_refuses_to_start_while_the_model_is_the_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stops before building anything rather than sending YOUR_MODEL to Ollama."""
    from runtime.chat import cli
    from runtime.config import AegisConfig, ModelConfig

    cfg = AegisConfig(models=ModelConfig(smart_provider="ollama", smart_local="YOUR_MODEL"))
    built: list[object] = []
    monkeypatch.setattr(cli, "get_config", lambda: cfg)
    monkeypatch.setattr(cli, "build_pipeline", lambda *a, **k: built.append(a))

    assert cli.main([]) == 1
    assert built == []
    assert "YOUR_MODEL" in capsys.readouterr().err
