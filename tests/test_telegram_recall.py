"""Phase 7 §3.5 + §4.1 — `/recall verbatim` and `/recall vault:` contracts.

Pins:

* The slash supports two forms, one dispatcher slot:
    - `/recall verbatim <session_id>` → resolve ColdRef, verify
      sha256 via `ColdStorageReader`, render turns.
    - `/recall vault:<slug>` → pull body via `VaultBodyLoader`,
      render as a quoted block.
* Handler NEVER raises — per §2.8, every failure path (resolver
  error, cold storage missing/corrupt, vault loader failure)
  renders a human-readable reply.
* Missing forms, empty slugs, and unknown prefixes all fall through
  to the canonical usage string.
* `render_verbatim` header names `session_id` + `turn_range` so
  operators can quote it back into a follow-up.
* `build_read_only_handlers` only registers `/recall` when both
  resolver and reader are wired — otherwise the dispatcher reports
  `unknown_command`, not a stub reply.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.memory.cold_storage import (
    ColdStorageMismatch,
    ColdStorageMissing,
    ColdStorageReader,
)
from runtime.chat.memory.tier2 import ColdRef
from runtime.chat.memory.tier3 import Turn
from runtime.chat.session_log import ChatSessionLog
from runtime.chat.telegram import (
    ColdRefResolver,
    IncomingMessage,
    ParsedCommand,
    build_read_only_handlers,
    recall_handler,
    render_vault_note,
    render_verbatim,
)

pytestmark = pytest.mark.unit


class _FakeResolver:
    """Tiny in-memory resolver for tests."""

    def __init__(self, refs: dict[str, ColdRef] | None = None) -> None:
        self._refs = refs or {}
        self._error: Exception | None = None

    def raise_on_any(self, exc: Exception) -> None:
        self._error = exc

    def resolve(self, session_id: str) -> ColdRef | None:
        if self._error is not None:
            raise self._error
        return self._refs.get(session_id)


class _FakeVaultLoader:
    def __init__(self, bodies: dict[str, str] | None = None) -> None:
        self._bodies = bodies or {}
        self._error: Exception | None = None

    def raise_on_any(self, exc: Exception) -> None:
        self._error = exc

    def load(self, rel_path: str) -> str:
        if self._error is not None:
            raise self._error
        return self._bodies.get(rel_path, "")


def _msg(text: str = "/recall") -> IncomingMessage:
    return IncomingMessage(chat_id=111, user_id=1, text=text)


def _write_cold_session(
    tmp_path: Path,
    *,
    chat_id: str = "chat-a",
    session_id: str = "sess-01",
    texts: tuple[str, ...] = ("hello", "world"),
) -> ColdRef:
    """Write a real session JSONL + build a verified `ColdRef` for it."""
    base_dir = tmp_path / "sessions"
    log = ChatSessionLog(base_dir, chat_id=chat_id, session_id=session_id)
    for i, text in enumerate(texts):
        turn = Turn(
            chat_id=chat_id,
            turn_idx=i,
            role="user" if i % 2 == 0 else "bot",
            text=text,
            ts=datetime(2026, 4, 19, 12, i, tzinfo=UTC),
        )
        log.append(turn)
    _nbytes, sha = log.slice_sha256(0, len(texts))
    return ColdRef(
        session_id=session_id,
        jsonl_path=str(log.path),
        turn_range=(0, len(texts)),
        sha256=sha,
    )


# --- usage / prefix handling -------------------------------------------


def test_recall_usage_without_args() -> None:
    handler = recall_handler(resolver=_FakeResolver(), reader=ColdStorageReader())
    out = handler(_msg(), ParsedCommand(name="/recall", args=()))
    assert "Usage" in out
    assert "verbatim" in out
    assert "vault:" in out


def test_recall_unknown_prefix() -> None:
    handler = recall_handler(resolver=_FakeResolver(), reader=ColdStorageReader())
    out = handler(_msg(), ParsedCommand(name="/recall", args=("summary",)))
    assert "Usage" in out


# --- /recall verbatim ---------------------------------------------------


def test_verbatim_missing_session_id() -> None:
    handler = recall_handler(resolver=_FakeResolver(), reader=ColdStorageReader())
    out = handler(_msg(), ParsedCommand(name="/recall", args=("verbatim",)))
    assert "Usage: /recall verbatim <session_id>" in out


def test_verbatim_empty_session_id() -> None:
    handler = recall_handler(resolver=_FakeResolver(), reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "   ")),
    )
    assert "Usage" in out


def test_verbatim_resolver_unknown_session() -> None:
    handler = recall_handler(resolver=_FakeResolver(), reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-missing")),
    )
    assert "No cold storage" in out
    assert "'sess-missing'" in out


def test_verbatim_renders_real_cold_slice(tmp_path: Path) -> None:
    ref = _write_cold_session(
        tmp_path, session_id="sess-01", texts=("hello", "world", "again")
    )
    resolver = _FakeResolver({"sess-01": ref})
    handler = recall_handler(resolver=resolver, reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-01")),
    )
    assert "Session sess-01" in out
    assert "turns [0,3)" in out
    assert "user: hello" in out
    assert "bot: world" in out
    assert "user: again" in out


def test_verbatim_resolver_error_is_friendly() -> None:
    resolver = _FakeResolver()
    resolver.raise_on_any(RuntimeError("db down"))
    handler = recall_handler(resolver=resolver, reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-01")),
    )
    assert "Failed to look up" in out
    assert "'sess-01'" in out


def test_verbatim_cold_storage_missing(tmp_path: Path) -> None:
    # Build a ref pointing at a file that doesn't exist.
    ref = ColdRef(
        session_id="sess-gone",
        jsonl_path=str(tmp_path / "does-not-exist.jsonl"),
        turn_range=(0, 1),
        sha256="0" * 64,
    )
    resolver = _FakeResolver({"sess-gone": ref})
    handler = recall_handler(resolver=resolver, reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-gone")),
    )
    assert "aged out of cold storage" in out


def test_verbatim_cold_storage_mismatch(tmp_path: Path) -> None:
    """File exists but sha mismatches → typed failure → friendly reply."""
    ref = _write_cold_session(tmp_path, session_id="sess-bad")
    # Construct a ref with a fake sha so the reader raises Mismatch.
    bad_ref = ColdRef(
        session_id=ref.session_id,
        jsonl_path=ref.jsonl_path,
        turn_range=ref.turn_range,
        sha256="f" * 64,
    )
    resolver = _FakeResolver({"sess-bad": bad_ref})
    handler = recall_handler(resolver=resolver, reader=ColdStorageReader())
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-bad")),
    )
    assert "integrity check" in out


class _ExplodingReader:
    def read(self, ref: ColdRef) -> object:
        del ref
        raise RuntimeError("io storm")


def test_verbatim_reader_unexpected_error(tmp_path: Path) -> None:
    ref = _write_cold_session(tmp_path, session_id="sess-x")
    resolver = _FakeResolver({"sess-x": ref})
    # The real ColdStorageReader type is a Protocol — _ExplodingReader
    # matches structurally.
    handler = recall_handler(resolver=resolver, reader=_ExplodingReader())  # type: ignore[arg-type]
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-x")),
    )
    assert "Failed to read" in out


# --- /recall vault: -----------------------------------------------------


def test_vault_empty_slug() -> None:
    handler = recall_handler(
        resolver=_FakeResolver(),
        reader=ColdStorageReader(),
        vault_loader=_FakeVaultLoader(),
    )
    out = handler(_msg(), ParsedCommand(name="/recall", args=("vault:",)))
    assert "Usage: /recall vault:<slug>" in out


def test_vault_loader_missing() -> None:
    handler = recall_handler(
        resolver=_FakeResolver(),
        reader=ColdStorageReader(),
    )
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("vault:phase-6-scope",)),
    )
    assert "not configured" in out
    assert "phase-6-scope" in out


def test_vault_renders_body() -> None:
    loader = _FakeVaultLoader({"phase-6-scope": "## Phase 6\n\nbody body"})
    handler = recall_handler(
        resolver=_FakeResolver(),
        reader=ColdStorageReader(),
        vault_loader=loader,
    )
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("vault:phase-6-scope",)),
    )
    assert "Vault: phase-6-scope" in out
    assert "## Phase 6" in out


def test_vault_empty_body_is_reported() -> None:
    loader = _FakeVaultLoader({"phase-6-scope": ""})
    handler = recall_handler(
        resolver=_FakeResolver(),
        reader=ColdStorageReader(),
        vault_loader=loader,
    )
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("vault:phase-6-scope",)),
    )
    assert "empty or missing" in out


def test_vault_loader_error_is_friendly() -> None:
    loader = _FakeVaultLoader()
    loader.raise_on_any(RuntimeError("fs gone"))
    handler = recall_handler(
        resolver=_FakeResolver(),
        reader=ColdStorageReader(),
        vault_loader=loader,
    )
    out = handler(
        _msg(),
        ParsedCommand(name="/recall", args=("vault:phase-6-scope",)),
    )
    assert "Failed to load vault note" in out


# --- formatter edge cases ----------------------------------------------


def test_render_verbatim_caps_oversized(tmp_path: Path) -> None:
    ref = _write_cold_session(
        tmp_path,
        session_id="sess-big",
        texts=tuple(f"turn {i} " + ("x" * 200) for i in range(50)),
    )
    reader = ColdStorageReader()
    out = render_verbatim(reader.read(ref))
    # Ellipsis sentinel when clipped.
    assert out.endswith("…")


def test_render_vault_note_preserves_newlines() -> None:
    body = "line 1\n\nline 3"
    out = render_vault_note("my-slug", body)
    assert out.startswith("Vault: my-slug\n\n")
    assert "line 3" in out


# --- build_read_only_handlers registration ------------------------------


def test_build_read_only_handlers_skips_recall_when_unwired(tmp_path: Path) -> None:
    handlers = build_read_only_handlers(tmp_path)
    assert "/recall" not in handlers


def test_build_read_only_handlers_registers_recall(tmp_path: Path) -> None:
    ref = _write_cold_session(tmp_path, session_id="sess-01")
    resolver = _FakeResolver({"sess-01": ref})
    handlers = build_read_only_handlers(
        tmp_path,
        recall_resolver=resolver,
        recall_reader=ColdStorageReader(),
        vault_loader=_FakeVaultLoader({"my-note": "body"}),
    )
    assert "/recall" in handlers
    # Sanity-check both forms route through the composite handler.
    verbatim_out = handlers["/recall"](
        _msg(),
        ParsedCommand(name="/recall", args=("verbatim", "sess-01")),
    )
    assert "Session sess-01" in verbatim_out
    vault_out = handlers["/recall"](
        _msg(),
        ParsedCommand(name="/recall", args=("vault:my-note",)),
    )
    assert "Vault: my-note" in vault_out


# --- typed-failure smoke ------------------------------------------------


def test_exception_classes_are_importable() -> None:
    """Sanity check that the cold-storage exception taxonomy exists."""
    assert issubclass(ColdStorageMissing, Exception)
    assert issubclass(ColdStorageMismatch, Exception)
    # Protocol ColdRefResolver should be available as a symbol even though
    # runtime checks would need @runtime_checkable.
    assert ColdRefResolver.__name__ == "ColdRefResolver"
