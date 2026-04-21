"""Phase 7 step 3 — Tier 1 loader contract.

Pins the behaviour:

* `Tier1Snapshot` is frozen + `extra="forbid"`.
* Missing IDENTITY.md / USER.md / prefs.json → empty content, no
  exception. Tier 1 is always present in the turn context.
* Malformed prefs.json → `Tier1LoadError`. Silent degradation
  would hide operator mistakes.
* Per-chat isolation — chat A's prefs are invisible to chat B.
* Chat-id sanitization prevents path traversal.
* Cache refreshes only when mtime or size changes; same file read
  twice hits the cache and returns defensive copies.
* Byte counts round-trip via `total_bytes`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from runtime.chat.memory import (
    Tier1Loader,
    Tier1LoadError,
    Tier1Snapshot,
    default_workspace_root,
)

pytestmark = pytest.mark.unit


# --- Snapshot model invariants --------------------------------------------


def test_snapshot_is_frozen() -> None:
    snap = Tier1Snapshot(
        identity="id",
        user="u",
        prefs={"k": "v"},
        bytes_identity=2,
        bytes_user=1,
        bytes_prefs=9,
    )
    with pytest.raises(ValidationError):
        snap.identity = "mutated"  # type: ignore[misc]


def test_snapshot_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Tier1Snapshot(  # type: ignore[call-arg]
            identity="",
            user="",
            prefs={},
            bytes_identity=0,
            bytes_user=0,
            bytes_prefs=0,
            extra="nope",
        )


def test_snapshot_rejects_negative_byte_counts() -> None:
    with pytest.raises(ValidationError):
        Tier1Snapshot(
            identity="",
            user="",
            prefs={},
            bytes_identity=-1,
            bytes_user=0,
            bytes_prefs=0,
        )


def test_snapshot_total_bytes_is_sum() -> None:
    snap = Tier1Snapshot(
        identity="a",
        user="b",
        prefs={},
        bytes_identity=10,
        bytes_user=20,
        bytes_prefs=5,
    )
    assert snap.total_bytes == 35


# --- default root ---------------------------------------------------------


def test_default_workspace_root_is_under_home() -> None:
    root = default_workspace_root()
    assert root == Path.home() / ".aegis" / "workspace"


# --- Loader happy paths ---------------------------------------------------


def test_load_all_missing_returns_empty_snapshot(tmp_path: Path) -> None:
    loader = Tier1Loader(root=tmp_path)
    snap = loader.load("chat-123")
    assert snap.identity == ""
    assert snap.user == ""
    assert snap.prefs == {}
    assert snap.total_bytes == 0


def test_load_reads_identity_and_user(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("I am AEGIS.\n", encoding="utf-8")
    (tmp_path / "USER.md").write_text("Operator: mwk\n", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    snap = loader.load("c1")
    assert snap.identity == "I am AEGIS.\n"
    assert snap.user == "Operator: mwk\n"
    assert snap.bytes_identity == len(b"I am AEGIS.\n")
    assert snap.bytes_user == len(b"Operator: mwk\n")
    assert snap.bytes_prefs == 0
    assert snap.prefs == {}


def test_load_reads_per_chat_prefs(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    payload = {"tone": "terse", "timezone": "Asia/Singapore"}
    raw = json.dumps(payload)
    (chat_dir / "prefs.json").write_text(raw, encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    snap = loader.load("c1")
    assert snap.prefs == payload
    assert snap.bytes_prefs == len(raw.encode())


def test_load_empty_prefs_file_is_treated_as_empty_dict(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text("   \n", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    snap = loader.load("c1")
    assert snap.prefs == {}


# --- tenant isolation -----------------------------------------------------


def test_prefs_are_isolated_per_chat(tmp_path: Path) -> None:
    for chat, body in [("a", {"tone": "terse"}), ("b", {"tone": "chatty"})]:
        d = tmp_path / "chats" / chat
        d.mkdir(parents=True)
        (d / "prefs.json").write_text(json.dumps(body), encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    assert loader.load("a").prefs == {"tone": "terse"}
    assert loader.load("b").prefs == {"tone": "chatty"}


# --- input validation -----------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", "../etc", "a/b", "a b", "a.b", "a\x00b", "çhat"],
)
def test_load_rejects_dangerous_chat_ids(tmp_path: Path, bad: str) -> None:
    loader = Tier1Loader(root=tmp_path)
    with pytest.raises(ValueError, match="chat_id must match"):
        loader.load(bad)


def test_invalidate_rejects_dangerous_chat_ids(tmp_path: Path) -> None:
    loader = Tier1Loader(root=tmp_path)
    with pytest.raises(ValueError, match="chat_id must match"):
        loader.invalidate("../etc")


# --- malformed prefs ------------------------------------------------------


def test_malformed_prefs_json_raises(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text("{not json", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    with pytest.raises(Tier1LoadError, match="not valid JSON"):
        loader.load("c1")


def test_prefs_non_object_root_raises(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text("[1, 2, 3]", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    with pytest.raises(Tier1LoadError, match="must be a JSON object"):
        loader.load("c1")


def test_prefs_scalar_root_raises(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text("42", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    with pytest.raises(Tier1LoadError, match="must be a JSON object"):
        loader.load("c1")


# --- caching semantics ----------------------------------------------------


def _touch_mtime(path: Path, offset_ns: int = 2_000_000_000) -> None:
    """Shift mtime forward so the cache entry is stale on the next load."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + offset_ns))


def test_cache_reuses_parsed_content_when_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "IDENTITY.md").write_text("hello", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    first = loader.load("c1")
    # Second load of an unchanged file must NOT call `read_text`. Patch it to
    # blow up if anyone tries.
    real_read_text = Path.read_text

    def _fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == tmp_path / "IDENTITY.md":
            raise AssertionError(f"cache miss: {self}")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _fail_read_text)
    second = loader.load("c1")
    assert second.identity == first.identity == "hello"
    assert second.bytes_identity == first.bytes_identity


def test_cache_refreshes_when_mtime_changes(tmp_path: Path) -> None:
    ident = tmp_path / "IDENTITY.md"
    ident.write_text("v1", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    assert loader.load("c1").identity == "v1"
    ident.write_text("v2-longer", encoding="utf-8")
    _touch_mtime(ident)
    assert loader.load("c1").identity == "v2-longer"


def test_cache_drops_when_file_deleted_after_caching(tmp_path: Path) -> None:
    ident = tmp_path / "IDENTITY.md"
    ident.write_text("v1", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    loader.load("c1")
    ident.unlink()
    snap = loader.load("c1")
    assert snap.identity == ""
    assert snap.bytes_identity == 0


def test_load_returns_defensive_prefs_copies(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chats" / "c1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "prefs.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    first = loader.load("c1")
    first.prefs["k"] = "tampered"
    second = loader.load("c1")
    assert second.prefs == {"k": "v"}


def test_invalidate_all_clears_cache(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("v1", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    loader.load("c1")
    assert loader._cache
    loader.invalidate()
    assert not loader._cache


def test_invalidate_single_chat_only_drops_that_prefs(tmp_path: Path) -> None:
    (tmp_path / "IDENTITY.md").write_text("id", encoding="utf-8")
    for chat in ("a", "b"):
        d = tmp_path / "chats" / chat
        d.mkdir(parents=True)
        (d / "prefs.json").write_text(json.dumps({"chat": chat}), encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    loader.load("a")
    loader.load("b")
    loader.invalidate("a")
    remaining_keys = set(loader._cache.keys())
    assert str(tmp_path / "chats" / "a" / "prefs.json") not in remaining_keys
    assert str(tmp_path / "chats" / "b" / "prefs.json") in remaining_keys
    assert str(tmp_path / "IDENTITY.md") in remaining_keys


# --- root exposure --------------------------------------------------------


def test_root_is_exposed(tmp_path: Path) -> None:
    loader = Tier1Loader(root=tmp_path)
    assert loader.root == tmp_path


# --- unicode / byte counting ---------------------------------------------


def test_byte_counts_use_utf8(tmp_path: Path) -> None:
    # "héllo" is 6 bytes in UTF-8 (é = 2 bytes).
    (tmp_path / "IDENTITY.md").write_text("héllo", encoding="utf-8")
    loader = Tier1Loader(root=tmp_path)
    snap = loader.load("c1")
    assert snap.identity == "héllo"
    assert snap.bytes_identity == 6
