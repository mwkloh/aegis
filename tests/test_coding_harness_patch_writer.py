"""Patch-file rendering and append-only directory layout."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness.draft import Draft
from runtime.coding_harness.patch_writer import (
    diffs_dir,
    existing_drafts_for,
    patch_filename,
    write_patch,
)

pytestmark = pytest.mark.unit


def _draft(**kwargs: object) -> Draft:
    base: dict[str, object] = {
        "ct_id": "CT-001",
        "imp_id": "IMP-a86b087a",
        "model": "openrouter:minimax/minimax-m2.7",
        "summary": "Add fallback retrieval.",
        "unified_diff": "--- a/x\n+++ b/x\n@@\n-1\n+2\n",
        "test_notes": "Verify both paths in unit test.",
        "rollback": "git revert HEAD",
        "drafted_at": datetime(2026, 4, 18, 12, 7, tzinfo=UTC),
        "status": "ok",
    }
    base.update(kwargs)
    return Draft(**base)  # type: ignore[arg-type]


def test_filename_format(tmp_path: Path) -> None:
    name = patch_filename(_draft())
    assert name == "CT-001__IMP-a86b087a__2026-04-18T1207Z.patch.md"


def test_writes_under_workspace_diffs_dir(tmp_path: Path) -> None:
    target = write_patch(tmp_path, _draft())
    assert target.parent == diffs_dir(tmp_path)
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "## Unified diff" in text
    assert "```diff" in text
    assert "## Test notes" in text
    assert "## Rollback" in text


def test_force_writes_alongside_existing(tmp_path: Path) -> None:
    first = write_patch(tmp_path, _draft())
    second = write_patch(
        tmp_path, _draft(drafted_at=datetime(2026, 4, 18, 13, 30, tzinfo=UTC))
    )
    assert first != second
    drafts = existing_drafts_for(tmp_path, "CT-001")
    assert len(drafts) == 2


def test_stub_status_omits_diff_block(tmp_path: Path) -> None:
    target = write_patch(
        tmp_path,
        _draft(status="stub", unified_diff="", reason="LLM not configured"),
    )
    text = target.read_text(encoding="utf-8")
    assert "_Stub" in text
    assert "LLM not configured" in text
    assert "```diff" not in text


def test_refused_status_marks_block(tmp_path: Path) -> None:
    target = write_patch(
        tmp_path,
        _draft(status="refused", unified_diff="", reason="canon scope: SOUL.md"),
    )
    text = target.read_text(encoding="utf-8")
    assert "_Refused" in text
    assert "canon scope: SOUL.md" in text
    assert "```diff" not in text


def test_collision_in_same_minute_writes_suffixed_file(tmp_path: Path) -> None:
    first = write_patch(tmp_path, _draft())
    second = write_patch(tmp_path, _draft())  # exact same minute
    assert first != second
    assert first.exists()
    assert second.exists()
    assert second.name.endswith("-02.patch.md")
    drafts = existing_drafts_for(tmp_path, "CT-001")
    assert len(drafts) == 2


def test_does_not_touch_canon(tmp_path: Path) -> None:
    canon = tmp_path / "AGENTS.md"
    canon.write_text("# canonical\n", encoding="utf-8")
    write_patch(tmp_path, _draft())
    assert canon.read_text(encoding="utf-8") == "# canonical\n"
