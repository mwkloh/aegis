"""Doctor rows a first-time, all-local operator sees on a fresh install."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

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


def _install_skill(root: Path, name: str, text: str) -> None:
    skill = root / "workspace" / "skills" / name
    skill.mkdir(parents=True)
    (skill / "skill.yaml").write_text(text, encoding="utf-8")


def test_unfilled_username_placeholder_in_installed_skills_warns(root: Path) -> None:
    _install_skill(root, "read_file", "e.g. /Users/YOUR_USERNAME/Desktop/foo.png\n")
    _install_skill(root, "list_files", "e.g. /Users/YOUR_USERNAME/Desktop\n")
    _install_skill(root, "echo", "no paths here\n")

    name, ok, detail, severity = doctor._check_skill_placeholders()

    assert (name, ok, severity) == ("skills:example_paths", False, "warn")
    assert "2 skill(s)" in detail
    assert "YOUR_USERNAME" in detail
    assert "sed" in detail  # the fix is in the row, copy-pasteable


def test_filled_in_skills_pass(root: Path) -> None:
    _install_skill(root, "read_file", "e.g. /Users/alice/Desktop/foo.png\n")

    _name, ok, _detail, severity = doctor._check_skill_placeholders()

    assert (ok, severity) == (True, "ok")


def test_no_skills_installed_yet_is_not_a_warning(root: Path) -> None:
    _name, ok, _detail, severity = doctor._check_skill_placeholders()

    assert (ok, severity) == (True, "ok")


def test_an_unreadable_skill_is_left_to_the_load_check(root: Path) -> None:
    _install_skill(root, "read_file", "e.g. /Users/alice/Desktop\n")
    bad = root / "workspace" / "skills" / "binary" / "skill.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\xff\xfe not utf-8")

    assert doctor._check_skill_placeholders()[3] == "ok"  # not a crash, not its job


_VALID_SKILL = (
    "id: echo\nversion: '0.1.0'\ndescription: Echo text back.\nintents: [echo]\n"
    "tool: echo\nargs_schema: {}\n"
)


def test_a_skill_the_registry_cannot_load_is_an_error(root: Path) -> None:
    """Start-up loads every skill.yaml and crashes on a bad one, so doctor must fail
    (exit 1) rather than warn and let `make run` die."""
    _install_skill(root, "echo", _VALID_SKILL)
    bad = root / "workspace" / "skills" / "binary" / "skill.yaml"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\xff\xfe not utf-8")
    _install_skill(root, "notes", "just a sentence, not a mapping\n")

    name, ok, detail, severity = doctor._check_skills_load()

    assert (name, ok, severity) == ("skills:load", False, "error")
    assert "binary" in detail and "notes" in detail and "echo" not in detail


def test_skills_that_load_pass_the_load_check(root: Path) -> None:
    _install_skill(root, "echo", _VALID_SKILL)
    assert doctor._check_skills_load()[1:] == (True, "ok", "ok")


def test_placeholder_fix_quotes_a_catalog_path_with_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "Application Support" / ".aegis"
    monkeypatch.setenv("AEGIS_ROOT", str(target))
    monkeypatch.delenv("AEGIS_HOME", raising=False)
    _install_skill(target, "read_file", "e.g. /Users/YOUR_USERNAME/Desktop\n")

    _name, _ok, detail, _severity = doctor._check_skill_placeholders()

    assert f"'{target / 'workspace' / 'skills'}'/*/skill.yaml" in detail


def test_smart_local_placeholder_says_to_choose_a_model(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(doctor, "_list_ollama_models", lambda: ["gemma4:e2b"])
    monkeypatch.setenv("MODEL_SMART_LOCAL", "YOUR_MODEL")

    rows = {name: (sev, detail) for name, _ok, detail, sev in doctor._check_models()}

    sev, detail = rows["ollama:model:YOUR_MODEL"]
    assert sev == "error"
    assert "MODEL_SMART_LOCAL" in detail
    assert "ollama pull YOUR_MODEL" not in detail


def test_a_different_tag_of_the_same_family_is_not_installed() -> None:
    # gemma4:e2b on disk must not make gemma4:e4b look pulled.
    assert not doctor._model_present(["gemma4:e2b"], "gemma4:e4b")
    assert doctor._model_present(["gemma4:e4b"], "gemma4:e4b")


def test_installed_models_leave_out_pulled_cloud_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pulled cloud model is listed in /api/tags with a `remote_host` and no weights,
    so it must not pass a model check as installed."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    tags = {
        "models": [
            {"name": "gemma4:e2b"},
            {"name": "glm-5.3:cloud", "remote_host": "https://ollama.com"},
        ]
    }
    with respx.mock() as mock:
        mock.get("http://127.0.0.1:11434/api/tags").mock(
            return_value=httpx.Response(200, json=tags)
        )
        assert doctor._list_ollama_models() == ["gemma4:e2b"]


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
