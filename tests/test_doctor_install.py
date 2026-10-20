"""Doctor rows a first-time, all-local operator sees on a fresh install."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts import doctor


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / ".aegis"
    target.mkdir()
    monkeypatch.setenv("AEGIS_ROOT", str(target))
    monkeypatch.delenv("AEGIS_HOME", raising=False)
    for key in ("OPENROUTER_API_KEY", "MODEL_SMART_PROVIDER", "MODEL_FAST", "MODEL_SMART_LOCAL"):
        monkeypatch.delenv(key, raising=False)
    return target


def _errors(rows: list[doctor.Row]) -> list[str]:
    return [name for name, _ok, _detail, severity in rows if severity == "error"]


def test_layout_follows_aegis_root(root: Path) -> None:
    (root / "workspace").mkdir()
    (root / "workspace" / "AGENTS.md").write_text("x", encoding="utf-8")

    rows = {name: sev for name, _ok, _d, sev in doctor._check_layout()}

    assert rows[str(root / "workspace" / "AGENTS.md")] == "ok"
    assert rows[str(root / "workspace" / "IDENTITY.md")] == "error"


def test_all_local_setup_does_not_need_an_openrouter_key(root: Path) -> None:
    (root / ".env").write_text("MODEL_FAST=gemma4:e2b\nOPENROUTER_API_KEY=\n", encoding="utf-8")

    assert _errors(doctor._check_env()) == []


def test_openrouter_smart_tier_requires_a_key(root: Path) -> None:
    (root / ".env").write_text(
        "MODEL_FAST=gemma4:e2b\nMODEL_SMART_PROVIDER=openrouter\nOPENROUTER_API_KEY=\n",
        encoding="utf-8",
    )

    assert ".env:OPENROUTER_API_KEY" in _errors(doctor._check_env())


def test_openrouter_smart_tier_is_satisfied_by_a_key(root: Path) -> None:
    (root / ".env").write_text(
        "MODEL_FAST=gemma4:e2b\nMODEL_SMART_PROVIDER=openrouter\nOPENROUTER_API_KEY=sk-x\n",
        encoding="utf-8",
    )

    assert _errors(doctor._check_env()) == []


def test_missing_fast_model_setting_is_an_error(root: Path) -> None:
    (root / ".env").write_text("OLLAMA_BASE_URL=http://127.0.0.1:11434\n", encoding="utf-8")

    assert ".env:MODEL_FAST" in _errors(doctor._check_env())


def test_models_check_covers_the_smart_local_model_with_a_pull_hint(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_list_ollama_models", lambda: ["gemma4:e2b"])
    monkeypatch.setenv("MODEL_SMART_LOCAL", "gemma4:e4b")

    rows = {name: (sev, detail) for name, _ok, detail, sev in doctor._check_models()}

    sev, detail = rows["ollama:model:gemma4:e4b"]
    assert sev == "error"
    assert "ollama pull gemma4:e4b" in detail
    assert rows["ollama:model:gemma4:e2b"][0] == "ok"


def test_a_different_tag_of_the_same_family_is_not_installed() -> None:
    # gemma4:e2b on disk must not make gemma4:e4b look pulled.
    assert not doctor._model_present(["gemma4:e2b"], "gemma4:e4b")
    assert doctor._model_present(["gemma4:e4b"], "gemma4:e4b")


def test_an_untagged_model_matches_any_installed_tag() -> None:
    assert doctor._model_present(["llama3.2:3b"], "llama3.2")
    assert doctor._model_present(["llama3.2:latest"], "llama3.2")


def test_models_check_skips_smart_local_when_smart_tier_is_openrouter(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_list_ollama_models", lambda: ["gemma4:e2b", "gemma4:e4b"])
    monkeypatch.setenv("MODEL_SMART_PROVIDER", "openrouter")
    monkeypatch.setenv("MODEL_SMART_LOCAL", "llama3.1:8b")

    names = [name for name, *_ in doctor._check_models()]

    assert "ollama:model:llama3.1:8b" not in names
