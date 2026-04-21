"""Phase 7 step 5 dependency — `ChatSessionLog` contract.

Pins:

* JSONL lines are UTF-8 `"\\n"`-terminated (not `os.linesep`) so slice
  hashes are reproducible across OSes.
* Byte offsets are tracked per `turn_idx`; `slice_sha256` matches a
  hand-computed sha over the exact on-disk bytes.
* Chat-id / session-id regex bars path traversal.
* Append enforces matching `chat_id` and rejects duplicate `turn_idx`.
* Reopening an existing log rebuilds the offset table.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.chat.memory import Turn
from runtime.chat.session_log import ChatSessionLog

pytestmark = pytest.mark.unit


_NOW = datetime(2026, 4, 19, 12, 0, tzinfo=UTC)


def _turn(chat_id: str, idx: int, *, role: str = "user", text: str = "hi") -> Turn:
    return Turn(chat_id=chat_id, turn_idx=idx, role=role, text=text, ts=_NOW)


# --- id validation ---------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a.b", "a b", "çhat"])
def test_rejects_dangerous_chat_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="chat_id must match"):
        ChatSessionLog(tmp_path, chat_id=bad, session_id="s1")


@pytest.mark.parametrize("bad", ["", "../etc", "a/b", "a.b", "a b"])
def test_rejects_dangerous_session_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="session_id must match"):
        ChatSessionLog(tmp_path, chat_id="c1", session_id=bad)


# --- append basics ---------------------------------------------------------


def test_path_layout(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    assert log.path == tmp_path / "c1" / "s1.jsonl"
    assert log.chat_id == "c1"
    assert log.session_id == "s1"


def test_append_creates_directory_and_writes_line(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    log.append(_turn("c1", 0, text="hello"))
    data = log.path.read_bytes()
    assert data.endswith(b"\n")
    assert b"\r\n" not in data
    assert b'"text": "hello"' in data
    assert b'"turn_idx": 0' in data


def test_append_rejects_mismatched_chat_id(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    with pytest.raises(ValueError, match="does not match log"):
        log.append(_turn("other", 0))


def test_append_rejects_duplicate_turn_idx(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    log.append(_turn("c1", 0))
    with pytest.raises(ValueError, match="already written"):
        log.append(_turn("c1", 0))


def test_append_returns_byte_span(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    start1, end1 = log.append(_turn("c1", 0, text="a"))
    start2, end2 = log.append(_turn("c1", 1, text="b"))
    assert start1 == 0
    assert end1 > 0
    assert start2 == end1
    assert end2 > start2


# --- slice_sha256 ---------------------------------------------------------


def test_slice_sha256_matches_disk_bytes(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    for i in range(3):
        log.append(_turn("c1", i, text=f"body-{i}"))
    nbytes, sha = log.slice_sha256(0, 3)
    raw = log.path.read_bytes()
    assert nbytes == len(raw)
    assert sha == hashlib.sha256(raw).hexdigest()


def test_slice_sha256_partial_range(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    spans = [log.append(_turn("c1", i, text=f"body-{i}")) for i in range(4)]
    nbytes, sha = log.slice_sha256(1, 3)
    expected = log.path.read_bytes()[spans[1][0] : spans[2][1]]
    assert nbytes == len(expected)
    assert sha == hashlib.sha256(expected).hexdigest()


def test_slice_sha256_requires_positive_range(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    log.append(_turn("c1", 0))
    with pytest.raises(ValueError, match="end_turn_idx"):
        log.slice_sha256(0, 0)


def test_slice_sha256_missing_turn_raises(tmp_path: Path) -> None:
    log = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    log.append(_turn("c1", 0))
    log.append(_turn("c1", 2))  # gap at idx 1
    with pytest.raises(ValueError, match="turn_idx 1 not in log"):
        log.slice_sha256(0, 3)


# --- re-open / reload -----------------------------------------------------


def test_reopen_rebuilds_offsets(tmp_path: Path) -> None:
    first = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    for i in range(2):
        first.append(_turn("c1", i, text=f"x{i}"))
    first_sha = first.slice_sha256(0, 2)
    second = ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
    assert second.slice_sha256(0, 2) == first_sha
    # And it can still append at the next free index.
    second.append(_turn("c1", 2, text="x2"))
    assert second.slice_sha256(0, 3)[0] > first_sha[0]


def test_reopen_malformed_line_raises(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    (chat_dir / "s1.jsonl").write_bytes(b"not-json\n")
    with pytest.raises(ValueError, match="malformed JSONL"):
        ChatSessionLog(tmp_path, chat_id="c1", session_id="s1")
