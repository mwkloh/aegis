"""Preflight tests for `scripts/telegram_smoke.py`.

Pins:

* Missing bot_token → hard fail ("bot_token" in errors).
* Empty allowlist → hard fail ("allowlist" in errors).
* Missing OpenRouter key → warning only (no error — chat falls back
  to legacy placeholder, slashes still work).
* Fully-configured cfg → zero errors.

We never import `python-telegram-bot`; the script itself imports it
lazily inside `main()`, so preflight is safe on a bare venv.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
)
from scripts.telegram_smoke import _preflight, _storage_summary

pytestmark = pytest.mark.unit


def _cfg(
    tmp_path: Path,
    *,
    token: str | None = "tg-token",  # noqa: S107
    allowlist: tuple[int, ...] = (12345,),
    api_key: str | None = "sk-test",
) -> AegisConfig:
    return AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(smart="minimax/minimax-m2.7"),
        providers=ProviderConfig(openrouter_api_key=api_key),
        telegram=TelegramConfig(bot_token=token, user_allowlist=list(allowlist)),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
    )


def test_preflight_passes_on_full_config(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("id", encoding="utf-8")
    (tmp_path / "USER.md").write_text("u", encoding="utf-8")
    errors = _preflight(_cfg(tmp_path))
    assert errors == []


def test_preflight_flags_missing_bot_token(tmp_path: Path) -> None:
    errors = _preflight(_cfg(tmp_path, token=None))
    assert "bot_token" in errors


def test_preflight_flags_empty_allowlist(tmp_path: Path) -> None:
    errors = _preflight(_cfg(tmp_path, allowlist=()))
    assert "allowlist" in errors


def test_preflight_missing_openrouter_is_warning_not_error(tmp_path: Path) -> None:
    # Missing API key must NOT block the smoke — the legacy placeholder
    # still serves chat and every slash command works.
    errors = _preflight(_cfg(tmp_path, api_key=None))
    assert errors == []


def test_storage_summary_returns_the_three_paths(tmp_path: Path) -> None:
    summary = _storage_summary(_cfg(tmp_path))
    assert set(summary.keys()) == {"workspace", "sessions_dir", "memory_db"}
    assert summary["sessions_dir"] == tmp_path / "sessions"
