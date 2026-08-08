"""Unit tests for runtime/serve.py — production entry-point.

Covers:
* main() returns 1 when bot_token is missing
* main() returns 1 when user_allowlist is empty
* main() calls build_application + run_polling on happy path (returns 0)
* main() returns 2 on ImportError from build_application
"""
from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import (
    AegisConfig,
    ModelConfig,
    ProviderConfig,
    StorageConfig,
    TelegramConfig,
    VaultIndexingConfig,
    reset_config,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(
    tmp_path: Path,
    *,
    token: str | None = "tg-test-token",
    allowlist: tuple[int, ...] = (99999,),
) -> AegisConfig:
    return AegisConfig(
        aegis_home=tmp_path,
        aegis_root=tmp_path,
        models=ModelConfig(),
        providers=ProviderConfig(),
        telegram=TelegramConfig(bot_token=token, user_allowlist=list(allowlist)),
        storage=StorageConfig(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            memory_db=tmp_path / "memory" / "index.db",
        ),
        vault_indexing=VaultIndexingConfig(),
    )


@pytest.fixture(autouse=True)
def _clear_cache() -> Generator[None, None, None]:
    """Reset the config cache before and after every test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _reload_serve() -> Generator[None, None, None]:
    """Ensure runtime.serve is freshly importable each test run."""
    # Remove cached module so patches in one test don't bleed into the next.
    sys.modules.pop("runtime.serve", None)
    yield
    sys.modules.pop("runtime.serve", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServeValidation:
    def test_returns_1_when_bot_token_missing(self, tmp_path: Path) -> None:
        """main() should fail-fast with exit 1 when bot_token is None."""
        cfg = _cfg(tmp_path, token=None)
        import runtime.serve as serve
        with patch.object(serve, "get_config", return_value=cfg):
            result = serve.main()
        assert result == 1

    def test_returns_1_when_allowlist_empty(self, tmp_path: Path) -> None:
        """main() should fail-fast with exit 1 when user_allowlist is empty."""
        cfg = _cfg(tmp_path, allowlist=())
        import runtime.serve as serve
        with patch.object(serve, "get_config", return_value=cfg):
            result = serve.main()
        assert result == 1


class TestServeHappyPath:
    def test_calls_build_application_and_run_polling(self, tmp_path: Path) -> None:
        """main() should build and start the bot on a valid config."""
        cfg = _cfg(tmp_path)
        mock_app = MagicMock()
        mock_build = MagicMock(return_value=mock_app)

        import runtime.serve as serve
        with (
            patch.object(serve, "get_config", return_value=cfg),
            patch.object(serve, "build_application", mock_build),
        ):
            result = serve.main()

        mock_build.assert_called_once_with(cfg)
        mock_app.run_polling.assert_called_once()
        assert result == 0


class TestServeImportError:
    def test_returns_2_on_import_error_from_build_application(
        self, tmp_path: Path
    ) -> None:
        """main() returns 2 when build_application raises ImportError."""
        cfg = _cfg(tmp_path)
        mock_build = MagicMock(side_effect=ImportError("python-telegram-bot not installed"))

        import runtime.serve as serve
        with (
            patch.object(serve, "get_config", return_value=cfg),
            patch.object(serve, "build_application", mock_build),
        ):
            result = serve.main()

        assert result == 2

    def test_returns_2_when_build_application_is_none(self, tmp_path: Path) -> None:
        """main() returns 2 when build_application is None (optional dep absent)."""
        cfg = _cfg(tmp_path)
        import runtime.serve as serve
        with (
            patch.object(serve, "get_config", return_value=cfg),
            patch.object(serve, "build_application", None),
        ):
            result = serve.main()
        assert result == 2


class TestServeKeyboardInterrupt:
    def test_returns_0_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        """main() returns 0 when the user interrupts with Ctrl-C."""
        cfg = _cfg(tmp_path)
        mock_app = MagicMock()
        mock_app.run_polling.side_effect = KeyboardInterrupt
        mock_build = MagicMock(return_value=mock_app)
        import runtime.serve as serve
        with (
            patch.object(serve, "get_config", return_value=cfg),
            patch.object(serve, "build_application", mock_build),
        ):
            result = serve.main()
        assert result == 0
