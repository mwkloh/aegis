from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from runtime.chat.telegram.dispatch import IncomingMessage, ParsedCommand
from runtime.chat.telegram.skills_slash import skills_handler
from runtime.skills.chat_state import ChatSkillState
from runtime.skills.installer import stage_skill
from runtime.skills.loader import SkillLoader

pytestmark = pytest.mark.unit


def _clean_descriptor(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "vault_search",
        "version": "0.1.0",
        "description": "Search the operator's vault.",
        "intents": ["vault_search"],
        "tool": "vault_search",
        "args_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        "tools": [
            {
                "name": "search",
                "argv_template": ["aegis", "vault", "search", "--query", "{query}"],
                "timeout_ms": 5_000,
                "allow_net": False,
            }
        ],
    }
    data.update(overrides)
    return data


def _write_descriptor(dirpath: Path, data: dict[str, Any]) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / f"{data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _msg(chat_id: int = 42, text: str = "/skills") -> IncomingMessage:
    return IncomingMessage(chat_id=chat_id, user_id=1, text=text)


def _cmd(*args: str) -> ParsedCommand:
    return ParsedCommand(name="/skills", args=args)


def _fixture(tmp_path: Path) -> tuple[Any, Path, Path, SkillLoader, ChatSkillState]:
    catalog = tmp_path / "catalog"
    staging = tmp_path / "staging"
    catalog.mkdir()
    loader = SkillLoader(catalog)
    state = ChatSkillState(tmp_path / "skills.db")
    handler = skills_handler(
        loader=loader, state=state, staging_dir=staging, catalog_dir=catalog
    )
    return handler, staging, catalog, loader, state


# --- usage / routing ---------------------------------------------------------


def test_no_args_returns_usage(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd())
    assert "Usage:" in out


def test_unknown_subverb_returns_usage(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd("hatch"))
    assert "Usage:" in out


# --- list --------------------------------------------------------------------


def test_list_empty_catalog(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    assert handler(_msg(), _cmd("list")) == "No skills installed."


def test_list_shows_installed_with_checkmark(tmp_path: Path) -> None:
    handler, _, catalog, *_ = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())

    out = handler(_msg(), _cmd("list"))

    assert "vault_search" in out
    assert "✓" in out


def test_list_marks_disabled_for_this_chat(tmp_path: Path) -> None:
    handler, _, catalog, _, state = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())
    state.set_enabled(chat_id=42, skill_id="vault_search", enabled=False)

    out = handler(_msg(chat_id=42), _cmd("list"))

    assert "✗" in out
    assert "disabled for this chat" in out


def test_list_is_chat_scoped(tmp_path: Path) -> None:
    # A disable for chat=1 must not bleed into chat=2's listing.
    handler, _, catalog, _, state = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())
    state.set_enabled(chat_id=1, skill_id="vault_search", enabled=False)

    out = handler(_msg(chat_id=2), _cmd("list"))

    assert "✓" in out
    assert "✗" not in out


# --- show --------------------------------------------------------------------


def test_show_requires_id(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    assert handler(_msg(), _cmd("show")) == "Usage: /skills show <id>"


def test_show_unknown_skill(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd("show", "ghost"))
    assert "Unknown skill" in out
    assert "ghost" in out


def test_show_renders_progressive_disclosure_blob(tmp_path: Path) -> None:
    handler, _, catalog, *_ = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())

    out = handler(_msg(), _cmd("show", "vault_search"))

    # Progressive-disclosure invariant: prompt fields present, version/intents absent.
    assert "id: vault_search" in out
    assert "args_schema" in out
    assert "tools:" in out
    assert "version" not in out
    assert "intents" not in out


def test_show_truncates_oversized_bodies(tmp_path: Path) -> None:
    handler, _, catalog, *_ = _fixture(tmp_path)
    # Inflate args_schema with many properties so rendered YAML exceeds the cap.
    big_props: dict[str, Any] = {
        f"field_{i}": {"type": "string"} for i in range(400)
    }
    big_props["query"] = {"type": "string"}  # preserve tool placeholder
    _write_descriptor(
        catalog,
        _clean_descriptor(args_schema={"type": "object", "properties": big_props}),
    )

    out = handler(_msg(), _cmd("show", "vault_search"))

    assert out.endswith("...")
    assert len(out) < 4096  # Telegram message ceiling


# --- enable / disable --------------------------------------------------------


def test_enable_requires_id(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    assert handler(_msg(), _cmd("enable")) == "Usage: /skills enable <id>"


def test_disable_rejects_unknown_skill(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd("disable", "ghost"))
    assert "Unknown skill" in out


def test_disable_persists_to_state(tmp_path: Path) -> None:
    handler, _, catalog, _, state = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())

    out = handler(_msg(chat_id=7), _cmd("disable", "vault_search"))

    assert "disabled" in out
    assert state.is_enabled(chat_id=7, skill_id="vault_search") is False


def test_enable_deletes_the_row(tmp_path: Path) -> None:
    handler, _, catalog, _, state = _fixture(tmp_path)
    _write_descriptor(catalog, _clean_descriptor())
    state.set_enabled(chat_id=7, skill_id="vault_search", enabled=False)

    out = handler(_msg(chat_id=7), _cmd("enable", "vault_search"))

    assert "enabled" in out
    assert state.is_enabled(chat_id=7, skill_id="vault_search") is True
    assert state.list_disabled(7) == []


# --- staged ------------------------------------------------------------------


def test_staged_empty(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd("staged"))
    assert "No staged skills" in out


def test_staged_lists_ids(tmp_path: Path) -> None:
    handler, staging, *_ = _fixture(tmp_path)
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump(_clean_descriptor()), encoding="utf-8")
    stage_skill(src, staging_dir=staging)

    out = handler(_msg(), _cmd("staged"))

    assert "vault_search" in out
    assert "confirm" in out


# --- confirm -----------------------------------------------------------------


def test_confirm_requires_id(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    assert handler(_msg(), _cmd("confirm")) == "Usage: /skills confirm <id>"


def test_confirm_not_staged(tmp_path: Path) -> None:
    handler, *_ = _fixture(tmp_path)
    out = handler(_msg(), _cmd("confirm", "ghost"))
    assert "No staged descriptor" in out


def test_confirm_happy_path_refreshes_loader(tmp_path: Path) -> None:
    # The newly-confirmed descriptor must be visible via the loader
    # immediately — i.e. /skills list right after /skills confirm shows it.
    handler, staging, _, loader, _ = _fixture(tmp_path)
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump(_clean_descriptor()), encoding="utf-8")
    stage_skill(src, staging_dir=staging)

    out = handler(_msg(), _cmd("confirm", "vault_search"))

    assert "Installed" in out
    # Loader index picks up the new file without another refresh.
    assert loader.get("vault_search") is not None


def test_confirm_rejects_if_staged_tampered(tmp_path: Path) -> None:
    handler, staging, *_ = _fixture(tmp_path)
    src = tmp_path / "src.yaml"
    src.write_text(yaml.safe_dump(_clean_descriptor()), encoding="utf-8")
    stage_skill(src, staging_dir=staging)

    # Tamper staged file with a blocking (shell-binary) finding.
    tampered = _clean_descriptor(
        tools=[{"name": "shelly", "argv_template": ["sh", "-c", "boom"]}]
    )
    (staging / "vault_search.yaml").write_text(
        yaml.safe_dump(tampered), encoding="utf-8"
    )

    out = handler(_msg(), _cmd("confirm", "vault_search"))

    assert "blocking findings" in out
