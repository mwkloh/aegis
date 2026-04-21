from __future__ import annotations

from pathlib import Path

import pytest

from runtime.skills.chat_state import ChatSkillState

pytestmark = pytest.mark.unit


def _state(tmp_path: Path) -> ChatSkillState:
    return ChatSkillState(tmp_path / "skills.db")


def test_default_is_enabled(tmp_path: Path) -> None:
    state = _state(tmp_path)
    assert state.is_enabled(chat_id=42, skill_id="echo") is True


def test_disable_then_read_returns_false(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=42, skill_id="echo", enabled=False)
    assert state.is_enabled(chat_id=42, skill_id="echo") is False


def test_re_enable_deletes_the_row(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=42, skill_id="echo", enabled=False)
    state.set_enabled(chat_id=42, skill_id="echo", enabled=True)
    assert state.is_enabled(chat_id=42, skill_id="echo") is True
    # No disabled entries remain.
    assert state.list_disabled(42) == []


def test_toggles_are_isolated_per_chat(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=1, skill_id="echo", enabled=False)
    assert state.is_enabled(chat_id=1, skill_id="echo") is False
    # Another chat with no rows still gets the default.
    assert state.is_enabled(chat_id=2, skill_id="echo") is True


def test_list_disabled_sorted_and_chat_scoped(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=1, skill_id="bravo", enabled=False)
    state.set_enabled(chat_id=1, skill_id="alpha", enabled=False)
    state.set_enabled(chat_id=2, skill_id="charlie", enabled=False)

    assert state.list_disabled(1) == ["alpha", "bravo"]
    assert state.list_disabled(2) == ["charlie"]
    # Chats with no rows get an empty list, not a None.
    assert state.list_disabled(3) == []


def test_double_disable_is_idempotent(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=1, skill_id="echo", enabled=False)
    state.set_enabled(chat_id=1, skill_id="echo", enabled=False)
    assert state.list_disabled(1) == ["echo"]


def test_double_enable_when_never_disabled_is_noop(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_enabled(chat_id=1, skill_id="echo", enabled=True)
    assert state.is_enabled(chat_id=1, skill_id="echo") is True
    assert state.list_disabled(1) == []


def test_persistence_across_instances(tmp_path: Path) -> None:
    # Durability is the whole reason we use sqlite — a second process
    # opening the same db must see prior writes.
    state1 = _state(tmp_path)
    state1.set_enabled(chat_id=99, skill_id="vault_search", enabled=False)
    del state1

    state2 = _state(tmp_path)
    assert state2.is_enabled(chat_id=99, skill_id="vault_search") is False


def test_is_enabled_fails_open_on_db_error(tmp_path: Path) -> None:
    # Point at a non-openable path. The state must not crash the dispatcher.
    state = ChatSkillState(tmp_path / "ok.db")
    # Replace the db path with a path to a directory (sqlite can't open a dir).
    bad_path = tmp_path / "notafile"
    bad_path.mkdir()
    state._db_path = bad_path  # type: ignore[attr-defined]

    # Still returns True, never raises.
    assert state.is_enabled(chat_id=1, skill_id="echo") is True
    assert state.list_disabled(1) == []
