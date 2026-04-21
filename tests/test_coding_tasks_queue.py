"""Append-only CODING_TASKS.md queue."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.improvement.coding_tasks import load_tasks, queue_task, tasks_path
from runtime.improvement.proposal_loader import LoadedProposal, derive_imp_id

pytestmark = pytest.mark.unit


def _proposal(detector: str, change: str, affected: list[str] | None = None) -> LoadedProposal:
    aff = affected or []
    return LoadedProposal(
        imp_id=derive_imp_id(detector, aff, change),
        pattern_detector=detector,
        affected=aff,
        change=change,
        risk="low",
        rationale="r",
        source_run="2026-04-18T12:00Z",
    )


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert load_tasks(tmp_path) == []


def test_queue_assigns_monotonic_ct_numbers(tmp_path: Path) -> None:
    when = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    one = queue_task(tmp_path, _proposal("a", "do A", ["x"]), when=when)
    two = queue_task(tmp_path, _proposal("b", "do B", ["y"]), when=when)
    three = queue_task(tmp_path, _proposal("c", "do C", ["z"]), when=when)
    assert one is not None
    assert two is not None
    assert three is not None
    assert one.ct_id == "CT-001"
    assert two.ct_id == "CT-002"
    assert three.ct_id == "CT-003"
    text = tasks_path(tmp_path).read_text(encoding="utf-8")
    assert text.index("CT-001") < text.index("CT-002") < text.index("CT-003")


def test_queue_is_idempotent_per_imp_id(tmp_path: Path) -> None:
    p = _proposal("a", "do A", ["x"])
    first = queue_task(tmp_path, p)
    second = queue_task(tmp_path, p)
    assert first is not None
    assert second is None
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].imp_id == p.imp_id


def test_load_round_trips_fields(tmp_path: Path) -> None:
    when = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    p = _proposal("unknown_intent", "Add aliases.", ["intent_classifier"])
    queue_task(tmp_path, p, when=when)
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.ct_id == "CT-001"
    assert t.imp_id == p.imp_id
    assert t.scope == ["intent_classifier"]
    assert "do not modify canon" in t.constraints
    assert t.expected_output == "Add aliases."
    assert t.queued_at == when


def test_does_not_touch_canon_files(tmp_path: Path) -> None:
    canon = tmp_path / "AGENTS.md"
    canon.write_text("# canonical\n", encoding="utf-8")
    queue_task(tmp_path, _proposal("a", "do A", ["x"]))
    assert canon.read_text(encoding="utf-8") == "# canonical\n"
    assert (tmp_path / "improvement").is_dir()
