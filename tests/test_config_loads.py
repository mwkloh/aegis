"""Config loads from `~/.aegis/.env` + `~/.aegis/config.json` only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.config import get_config, reset_config

pytestmark = pytest.mark.unit


def test_get_config_returns_defaults_when_no_env(aegis_sandbox: Path) -> None:
    cfg = get_config()
    assert cfg.models.fast == "gemma4:e2b"
    assert cfg.models.reflection == "gemma4:e4b"
    assert cfg.providers.ollama_base_url.startswith("http://")
    assert cfg.providers.openrouter_api_key is None
    assert cfg.telegram.enabled is False
    assert cfg.storage.workspace == aegis_sandbox / "workspace"


def test_config_picks_up_env_overrides(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_FAST", "test:fast")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    reset_config()
    cfg = get_config()
    assert cfg.models.fast == "test:fast"
    assert cfg.providers.openrouter_api_key == "sk-test"


def test_config_reads_telegram_allowlist(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"telegram": {"enabled": True, "userAllowlist": [111, "222", "x"]}}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.telegram.enabled is True
    assert cfg.telegram.user_allowlist == [111, 222]


def test_config_ignores_unsubstituted_allowlist_placeholder(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A config.json with `"userAllowlist": "${TELEGRAM_ALLOW_FROM}"` (string,
    # not list) used to iterate the string character-by-character and yield
    # an empty list silently. Now it short-circuits to [] without crashing.
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"telegram": {"userAllowlist": "${TELEGRAM_ALLOW_FROM}"}}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.telegram.user_allowlist == []


def test_config_reads_telegram_allowlist_from_env(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # TELEGRAM_USER_ALLOWLIST=env overrides whatever is in config.json —
    # mirrors how TELEGRAM_BOT_TOKEN works. Comma-separated ints, trims
    # whitespace, skips non-numeric tokens.
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"telegram": {"userAllowlist": [999]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_USER_ALLOWLIST", "111, 222 , junk , -33")
    reset_config()
    cfg = get_config()
    assert cfg.telegram.user_allowlist == [111, 222, -33]


def test_config_env_allowlist_empty_string_falls_back_to_json(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty string env var = "env set, but no ids" → still overrides json
    # to the empty list (explicit null). This is the safe fail-closed choice.
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"telegram": {"userAllowlist": [999]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_USER_ALLOWLIST", "")
    reset_config()
    cfg = get_config()
    assert cfg.telegram.user_allowlist == []


def test_vault_indexing_defaults_to_disabled(aegis_sandbox: Path) -> None:
    cfg = get_config()
    assert cfg.vault_indexing.vault_root is None
    assert cfg.vault_indexing.sources == ()
    assert cfg.vault_indexing.is_enabled() is False
    assert cfg.vault_indexing.reindex_interval_hours == 6


def test_vault_indexing_reads_sources_from_config(aegis_sandbox: Path) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vaultIndexing": {
                    "vaultRoot": "~/data/obsidian",
                    "reindexIntervalHours": 12,
                    "sources": [
                        {
                            "path": "Research",
                            "priority": 1.5,
                            "glob": "**/*.md",
                            "exclude": ["**/Archive/**"],
                            "label": "research",
                        },
                        {"path": "Wiki", "priority": 1.2, "label": "wiki"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.vault_indexing.is_enabled() is True
    assert cfg.vault_indexing.reindex_interval_hours == 12
    assert len(cfg.vault_indexing.sources) == 2
    first = cfg.vault_indexing.sources[0]
    assert first.path == "Research"
    assert first.priority == 1.5
    assert first.exclude == ("**/Archive/**",)
    assert first.label == "research"
    assert cfg.vault_indexing.sources[1].glob == "**/*.md"  # default


def test_vault_indexing_drops_malformed_sources(aegis_sandbox: Path) -> None:
    # Entries without a `path`, non-dict entries, and priority out of [0.5, 4.0]
    # are silently skipped so the rest of the config still boots.
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "vaultIndexing": {
                    "vaultRoot": "~/data/obsidian",
                    "sources": [
                        {"path": "Good", "priority": 1.0},
                        "not a dict",
                        {"priority": 2.0},  # missing path
                        {"path": "BadPriority", "priority": 99.0},  # out of range
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    paths = [s.path for s in cfg.vault_indexing.sources]
    assert paths == ["Good"]


def test_vault_indexing_ignores_placeholder_string(aegis_sandbox: Path) -> None:
    # Mirrors the telegram allowlist quirk: a `${VAR}` placeholder is a
    # string, not the expected dict — we should fall back to disabled,
    # not crash.
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"vaultIndexing": "${VAULT_CONFIG}"}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.vault_indexing.is_enabled() is False


def test_config_defaults_board_to_empty(aegis_sandbox: Path) -> None:
    reset_config()
    cfg = get_config()
    assert cfg.board.panelists == []
    assert cfg.board.synthesis is None


def test_config_reads_board_from_json(aegis_sandbox: Path) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "board": {
                    "panelists": [
                        {
                            "name": "Analyst",
                            "model": "minimax/minimax-m2.7",
                            "provider": "openrouter",
                            "persona": "Be rigorous.",
                        }
                    ],
                    "synthesis": {
                        "model": "minimax/minimax-m2.7",
                        "provider": "openrouter",
                    },
                    "output_dir": "~/obsidian/Boards",
                    "excerpt_chars": 400,
                }
            }
        ),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert len(cfg.board.panelists) == 1
    assert cfg.board.panelists[0].name == "Analyst"
    assert cfg.board.synthesis is not None
    assert cfg.board.synthesis.model == "minimax/minimax-m2.7"
    assert "~" not in str(cfg.board.output_dir)
    assert cfg.board.excerpt_chars == 400


def test_config_ignores_malformed_board(aegis_sandbox: Path) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"board": "not-a-dict"}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.board.panelists == []


def test_harness_defaults_to_single_shot(aegis_sandbox: Path) -> None:
    """Step 1 of the multi-step plan must NOT change default behaviour."""
    cfg = get_config()
    assert cfg.harness.multi_step is False
    assert cfg.harness.max_steps == 5


def test_harness_reads_multi_step_from_json(aegis_sandbox: Path) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"harness": {"multi_step": True, "max_steps": 8}}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.harness.multi_step is True
    assert cfg.harness.max_steps == 8


def test_harness_env_override_flips_multi_step(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`HARNESS_MULTI_STEP=1` from the launchd plist must flip the flag."""
    monkeypatch.setenv("HARNESS_MULTI_STEP", "1")
    reset_config()
    cfg = get_config()
    assert cfg.harness.multi_step is True


def test_harness_ignores_malformed_max_steps(aegis_sandbox: Path) -> None:
    config_path = aegis_sandbox / "config.json"
    config_path.write_text(
        json.dumps({"harness": {"max_steps": "not-an-int"}}),
        encoding="utf-8",
    )
    reset_config()
    cfg = get_config()
    assert cfg.harness.max_steps == 5  # falls back to default

