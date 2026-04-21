"""Append-only JSONL writer for chat sessions with byte-offset tracking.

Phase 7 step 5 dependency. Each chat session lives in its own
`.jsonl` at `<base_dir>/<chat_id>/<session_id>.jsonl`. Every turn
is one line:

    {"turn_idx", "chat_id", "role", "text", "ts"}

The writer records the byte offset per `turn_idx` so the compressor
can compute `sha256(slice[start_byte : end_byte])` for a turn-range
and persist it as a `ColdRef` (§3.5). JSONL encoding is UTF-8 with
`"\\n"` line terminators (not `os.linesep`) so the slice hash is
reproducible across OSes.

Path-traversal guard: `chat_id` and `session_id` must match
`[A-Za-z0-9_-]+`.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

from runtime.chat.memory.tier3 import Turn

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ChatSessionLog:
    """Per-chat, per-session JSONL log with byte-offset tracking."""

    def __init__(self, base_dir: Path, *, chat_id: str, session_id: str) -> None:
        if not _ID_RE.fullmatch(chat_id):
            raise ValueError("chat_id must match [A-Za-z0-9_-]+")
        if not _ID_RE.fullmatch(session_id):
            raise ValueError("session_id must match [A-Za-z0-9_-]+")
        self._chat_id = chat_id
        self._session_id = session_id
        self._dir = base_dir / chat_id
        self._path = self._dir / f"{session_id}.jsonl"
        self._offsets: dict[int, tuple[int, int]] = {}
        self._size = 0
        if self._path.exists():
            self._rebuild_offsets()

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        return self._path

    def append(self, turn: Turn) -> tuple[int, int]:
        """Write one turn as a JSONL line. Returns `(start_byte, end_byte)`.

        Raises if `turn.chat_id` disagrees with the log's chat_id or if
        the same `turn_idx` has already been written.
        """
        if turn.chat_id != self._chat_id:
            raise ValueError(
                f"turn.chat_id={turn.chat_id!r} does not match log "
                f"chat_id={self._chat_id!r}"
            )
        if turn.turn_idx in self._offsets:
            raise ValueError(f"turn_idx {turn.turn_idx} already written")
        record = {
            "turn_idx": turn.turn_idx,
            "chat_id": turn.chat_id,
            "role": turn.role,
            "text": turn.text,
            "ts": turn.ts.isoformat(),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        payload = line.encode("utf-8")
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as fh:
            fh.write(payload)
        start = self._size
        end = start + len(payload)
        self._offsets[turn.turn_idx] = (start, end)
        self._size = end
        return start, end

    def slice_sha256(
        self, start_turn_idx: int, end_turn_idx: int
    ) -> tuple[int, str]:
        """Byte count + hex sha256 over turns `[start, end)` on disk.

        The range must be contiguous and every `turn_idx` must be in
        the log. Otherwise raises — silent skip would mask bugs that
        would later surface as sha256 mismatches on recall.
        """
        if end_turn_idx <= start_turn_idx:
            raise ValueError("end_turn_idx must be > start_turn_idx")
        spans: list[tuple[int, int]] = []
        for idx in range(start_turn_idx, end_turn_idx):
            if idx not in self._offsets:
                raise ValueError(f"turn_idx {idx} not in log")
            spans.append(self._offsets[idx])
        for prev, nxt in pairwise(spans):
            if prev[1] != nxt[0]:
                raise ValueError("turn byte-ranges are not contiguous")
        start_byte = spans[0][0]
        end_byte = spans[-1][1]
        with self._path.open("rb") as fh:
            fh.seek(start_byte)
            data = fh.read(end_byte - start_byte)
        if len(data) != end_byte - start_byte:
            raise ValueError("truncated read from session log")
        return len(data), hashlib.sha256(data).hexdigest()

    def _rebuild_offsets(self) -> None:
        pos = 0
        with self._path.open("rb") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    idx = int(record["turn_idx"])
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    raise ValueError(
                        f"malformed JSONL at byte {pos}: {exc}"
                    ) from exc
                self._offsets[idx] = (pos, pos + len(line))
                pos += len(line)
        self._size = pos


__all__ = ["ChatSessionLog"]
