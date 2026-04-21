"""Phase 7 step 6 — cold-storage reader.

Given a `ColdRef` persisted in tier 2, resolve it back to the raw
`Turn` sequence that the compressor archived. Per
`docs/PLAN_PHASE_7_TELEGRAM.md` §3.5:

* Re-hash the on-disk JSONL slice and compare to `ColdRef.sha256`.
  Any drift raises `ColdStorageMismatch` — silent acceptance would
  mask disk corruption and poison recall with fabricated history.
* Missing files raise `ColdStorageMissing` (typed, distinct from
  mismatch so the caller can decide: retention rolloff is expected
  to drop the file eventually, corruption is not).
* Gaps or duplicates in the declared `turn_range` raise
  `ColdStorageMismatch` — the sha happens to catch byte-level
  edits, but explicit structural checks make the failure mode
  legible.
* Reader is stateless; callers construct it once and reuse.

The reader uses UTF-8 / `"\\n"` byte slicing to stay consistent with
`ChatSessionLog` writes; mixing `os.linesep` or re-encoding would
change the sha.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.chat.memory.tier2 import ColdRef
from runtime.chat.memory.tier3 import Role, Turn


class ColdStorageError(RuntimeError):
    """Base for cold-storage read failures."""


class ColdStorageMissing(ColdStorageError):
    """The JSONL file referenced by the ColdRef no longer exists."""


class ColdStorageMismatch(ColdStorageError):
    """File exists but its bytes/structure disagree with the ColdRef."""


class ColdStorageRead(BaseModel):
    """Successful verified read — raw turns plus the ref they came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: ColdRef
    turns: tuple[Turn, ...] = Field(min_length=1)


class ColdStorageReader:
    """Resolve a `ColdRef` to verified `Turn` records.

    Not thread-safe in the sense that each read opens its own file
    handle; safe to share across threads if callers don't mutate the
    reader itself.
    """

    def read(self, ref: ColdRef) -> ColdStorageRead:
        """Verify + materialize the turns a `ColdRef` points to."""
        path = Path(ref.jsonl_path)
        try:
            raw_bytes = path.read_bytes()
        except FileNotFoundError as exc:
            raise ColdStorageMissing(
                f"cold JSONL not found: {ref.jsonl_path}"
            ) from exc
        start_idx, end_idx = ref.turn_range
        expected_count = end_idx - start_idx
        collected_bytes = bytearray()
        collected_records: list[tuple[int, dict[str, object], bytes]] = []
        pos = 0
        for line in raw_bytes.splitlines(keepends=True):
            try:
                record = json.loads(line)
                idx = int(cast(int, record["turn_idx"]))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                raise ColdStorageMismatch(
                    f"malformed JSONL at byte {pos} in {ref.jsonl_path}: {exc}"
                ) from exc
            if start_idx <= idx < end_idx:
                collected_bytes.extend(line)
                collected_records.append((idx, record, line))
            pos += len(line)

        if len(collected_records) != expected_count:
            raise ColdStorageMismatch(
                f"turn_range=[{start_idx},{end_idx}) expects {expected_count} "
                f"records, found {len(collected_records)} in {ref.jsonl_path}"
            )
        seen = {idx for idx, _, _ in collected_records}
        missing = [i for i in range(start_idx, end_idx) if i not in seen]
        if missing:
            raise ColdStorageMismatch(
                f"missing turn_idx {missing} in {ref.jsonl_path}"
            )
        actual_sha = hashlib.sha256(bytes(collected_bytes)).hexdigest()
        if actual_sha != ref.sha256:
            raise ColdStorageMismatch(
                f"sha256 mismatch for {ref.jsonl_path} "
                f"turn_range=[{start_idx},{end_idx}): "
                f"expected {ref.sha256}, got {actual_sha}"
            )

        turns: list[Turn] = []
        for _, record, _ in sorted(collected_records, key=lambda t: t[0]):
            try:
                turns.append(_record_to_turn(record))
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                raise ColdStorageMismatch(
                    f"record failed Turn validation in {ref.jsonl_path}: {exc}"
                ) from exc
        return ColdStorageRead(ref=ref, turns=tuple(turns))


def _record_to_turn(record: dict[str, object]) -> Turn:
    ts_raw = record["ts"]
    if not isinstance(ts_raw, str):
        raise TypeError(f"ts must be an ISO8601 string, got {type(ts_raw).__name__}")
    role = cast(Role, record["role"])
    return Turn(
        chat_id=cast(str, record["chat_id"]),
        turn_idx=int(cast(int, record["turn_idx"])),
        role=role,
        text=cast(str, record["text"]),
        ts=datetime.fromisoformat(ts_raw),
    )


__all__ = [
    "ColdStorageError",
    "ColdStorageMismatch",
    "ColdStorageMissing",
    "ColdStorageRead",
    "ColdStorageReader",
]
