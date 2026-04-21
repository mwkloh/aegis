from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.events import EventStream, EventType
from runtime.tools.record import (
    compute_argv_hash,
    load_tool_calls,
    record_tool_call,
)

pytestmark = pytest.mark.unit


def _events(tmp_path: Path, session_id: str = "sess-a") -> EventStream:
    return EventStream(tmp_path, session_id=session_id)


def _read_events(events: EventStream) -> list[dict]:
    lines = events.path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- compute_argv_hash --------------------------------------------------------


def test_compute_argv_hash_is_deterministic() -> None:
    a = compute_argv_hash(["echo", "hello"])
    b = compute_argv_hash(["echo", "hello"])
    assert a == b
    assert len(a) == 16


def test_compute_argv_hash_distinguishes_tokens() -> None:
    a = compute_argv_hash(["echo", "a", "b"])
    b = compute_argv_hash(["echo", "ab"])
    c = compute_argv_hash(["echo", "b", "a"])
    assert a != b
    assert a != c


def test_compute_argv_hash_handles_unicode() -> None:
    # Non-ASCII must still produce a stable ASCII hex hash.
    h = compute_argv_hash(["echo", "héllo", "世界"])
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# --- record_tool_call: happy path --------------------------------------------


def test_record_tool_call_emits_event(tmp_path: Path) -> None:
    events = _events(tmp_path)
    outcome = record_tool_call(
        events,
        imp_id="IMP-7f3a1c2b",
        skill="echo",
        tool="say",
        argv_hash="abcd1234abcd1234",
        verdict="verified",
        outcome_bytes=128,
    )
    assert outcome == "ok"
    entries = _read_events(events)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["type"] == EventType.TOOL_INVOKED.value
    assert entry["session_id"] == "sess-a"
    assert entry["payload"] == {
        "imp_id": "IMP-7f3a1c2b",
        "skill": "echo",
        "tool": "say",
        "argv_hash": "abcd1234abcd1234",
        "verdict": "verified",
        "outcome_bytes": 128,
    }


def test_record_tool_call_idempotent_on_composite_key(tmp_path: Path) -> None:
    events = _events(tmp_path)
    kwargs = {
        "imp_id": "IMP-7f3a1c2b",
        "skill": "echo",
        "tool": "say",
        "argv_hash": "abcd1234abcd1234",
        "verdict": "verified",
        "outcome_bytes": 10,
    }
    assert record_tool_call(events, **kwargs) == "ok"
    # Second identical call must no-op.
    assert record_tool_call(events, **kwargs) == "skipped_idempotent"
    entries = _read_events(events)
    assert len(entries) == 1


def test_record_tool_call_different_argv_not_deduped(tmp_path: Path) -> None:
    events = _events(tmp_path)
    common = {
        "imp_id": "IMP-7f3a1c2b",
        "skill": "echo",
        "tool": "say",
        "verdict": "verified",
        "outcome_bytes": 10,
    }
    assert record_tool_call(events, argv_hash="aaaa", **common) == "ok"
    assert record_tool_call(events, argv_hash="bbbb", **common) == "ok"
    entries = _read_events(events)
    assert len(entries) == 2


def test_record_tool_call_different_tool_not_deduped(tmp_path: Path) -> None:
    events = _events(tmp_path)
    common = {
        "imp_id": "IMP-7f3a1c2b",
        "skill": "echo",
        "argv_hash": "aaaa",
        "verdict": "verified",
        "outcome_bytes": 10,
    }
    assert record_tool_call(events, tool="say", **common) == "ok"
    assert record_tool_call(events, tool="whisper", **common) == "ok"
    assert len(_read_events(events)) == 2


def test_record_tool_call_different_session_not_deduped(tmp_path: Path) -> None:
    # Two sessions share the shard dir but each has its own file.
    a = _events(tmp_path, "sess-a")
    b = _events(tmp_path, "sess-b")
    common = {
        "imp_id": "IMP-7f3a1c2b",
        "skill": "echo",
        "tool": "say",
        "argv_hash": "aaaa",
        "verdict": "verified",
        "outcome_bytes": 10,
    }
    assert record_tool_call(a, **common) == "ok"
    assert record_tool_call(b, **common) == "ok"
    assert len(_read_events(a)) == 1
    assert len(_read_events(b)) == 1


# --- record_tool_call: validation ---------------------------------------------


def test_record_tool_call_rejects_empty_key_parts(tmp_path: Path) -> None:
    events = _events(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        record_tool_call(
            events,
            imp_id="",
            skill="echo",
            tool="say",
            argv_hash="aaaa",
            verdict="verified",
            outcome_bytes=0,
        )


def test_record_tool_call_rejects_negative_outcome_bytes(tmp_path: Path) -> None:
    events = _events(tmp_path)
    with pytest.raises(ValueError, match="outcome_bytes"):
        record_tool_call(
            events,
            imp_id="IMP-7f3a1c2b",
            skill="echo",
            tool="say",
            argv_hash="aaaa",
            verdict="verified",
            outcome_bytes=-1,
        )


# --- load_tool_calls ----------------------------------------------------------


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    events = _events(tmp_path)
    assert load_tool_calls(events) == []


def test_load_returns_records_in_file_order(tmp_path: Path) -> None:
    events = _events(tmp_path)
    record_tool_call(
        events,
        imp_id="IMP-7f3a1c2b",
        skill="echo",
        tool="say",
        argv_hash="aaaa",
        verdict="verified",
        outcome_bytes=10,
    )
    record_tool_call(
        events,
        imp_id="IMP-7f3a1c2b",
        skill="echo",
        tool="say",
        argv_hash="bbbb",
        verdict="exit_nonzero",
        outcome_bytes=25,
    )
    loaded = load_tool_calls(events)
    assert len(loaded) == 2
    assert [r.argv_hash for r in loaded] == ["aaaa", "bbbb"]
    assert loaded[1].verdict == "exit_nonzero"
    assert loaded[1].outcome_bytes == 25


def test_load_skips_malformed_lines(tmp_path: Path) -> None:
    events = _events(tmp_path)
    record_tool_call(
        events,
        imp_id="IMP-7f3a1c2b",
        skill="echo",
        tool="say",
        argv_hash="aaaa",
        verdict="verified",
        outcome_bytes=0,
    )
    # Append junk that shouldn't crash the loader.
    with events.path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write(json.dumps({"type": "tool.invoked", "payload": {"x": 1}}) + "\n")
    loaded = load_tool_calls(events)
    assert len(loaded) == 1
    assert loaded[0].argv_hash == "aaaa"


def test_load_ignores_legacy_thin_tool_invoked(tmp_path: Path) -> None:
    events = _events(tmp_path)
    # Thin Phase-1 payload: {"tool": ...} with no composite key.
    events.append(EventType.TOOL_INVOKED, {"tool": "legacy"})
    record_tool_call(
        events,
        imp_id="IMP-7f3a1c2b",
        skill="echo",
        tool="say",
        argv_hash="aaaa",
        verdict="verified",
        outcome_bytes=0,
    )
    loaded = load_tool_calls(events)
    assert len(loaded) == 1
    assert loaded[0].skill == "echo"


def test_load_ignores_unknown_verdict(tmp_path: Path) -> None:
    events = _events(tmp_path)
    events.append(
        EventType.TOOL_INVOKED,
        {
            "imp_id": "IMP-7f3a1c2b",
            "skill": "echo",
            "tool": "say",
            "argv_hash": "aaaa",
            "verdict": "nonsense",
            "outcome_bytes": 0,
        },
    )
    assert load_tool_calls(events) == []


def test_load_only_considers_tool_invoked_type(tmp_path: Path) -> None:
    events = _events(tmp_path)
    events.append(
        EventType.GOVERNANCE_DECISION,
        {
            "imp_id": "IMP-7f3a1c2b",
            "skill": "echo",
            "tool": "say",
            "argv_hash": "aaaa",
            "verdict": "verified",
            "outcome_bytes": 0,
        },
    )
    assert load_tool_calls(events) == []
