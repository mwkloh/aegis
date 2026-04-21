"""Pre-call scope refusal + post-call diff refusal for canon writes."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from runtime.coding_harness.coder import (
    diff_touches_canon,
    draft_for,
    scope_touches_canon,
)
from runtime.config import get_config
from runtime.events import EventStream
from runtime.improvement.coding_tasks import CodingTask
from runtime.model_router.clients import OllamaClient

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 4, 18, 12, 7, tzinfo=UTC)


def _task(scope: list[str]) -> CodingTask:
    return CodingTask(
        ct_id="CT-001",
        imp_id="IMP-a86b087a",
        scope=scope,
        constraints="do not modify canon; produce diffs only",
        expected_output="x",
        queued_at=_NOW,
    )


def test_scope_touches_canon_detects_basename_at_root() -> None:
    assert scope_touches_canon(["AGENTS.md"]) == "AGENTS.md"
    assert scope_touches_canon(["docs/SOUL.md"]) == "docs/SOUL.md"
    assert scope_touches_canon(["coding_harness/CODING_PROMPT.md"]) == (
        "coding_harness/CODING_PROMPT.md"
    )


def test_scope_touches_canon_clean_paths() -> None:
    assert scope_touches_canon(["runtime/intent/classifier.py"]) is None
    assert scope_touches_canon([]) is None


def test_diff_touches_canon_finds_target_lines() -> None:
    diff = (
        "--- a/SOUL.md\n+++ b/SOUL.md\n@@\n-x\n+y\n"
    )
    assert diff_touches_canon(diff) == "SOUL.md"


def test_diff_touches_canon_clean() -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@\n-1\n+2\n"
    assert diff_touches_canon(diff) is None


@pytest.mark.asyncio
async def test_pre_call_refusal_skips_llm(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    out = await draft_for(
        _task(["SOUL.md"]),
        client=None,  # would normally produce a stub; canon refusal wins first
        model="m",
        events=events,
        when=_NOW,
    )
    assert out.status == "refused"
    assert "scope touches canon" in out.reason
    text = events.path.read_text(encoding="utf-8")
    assert "harness.refused" in text
    assert "scope_touches_canon" in text


@pytest.mark.asyncio
async def test_post_call_refusal_drops_diff(tmp_path: Path) -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    events = EventStream(tmp_path / "sessions")
    bad_reply = json.dumps({
        "summary": "harmless looking",
        "unified_diff": "--- a/AGENTS.md\n+++ b/AGENTS.md\n@@\n-1\n+2\n",
        "test_notes": "n/a",
        "rollback": "n/a",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": bad_reply}})
        )
        out = await draft_for(
            _task(["runtime/intent/classifier.py"]),
            client=client,
            model="gemma4:e4b",
            events=events,
            when=_NOW,
        )
    assert out.status == "refused"
    assert "diff touches canon" in out.reason
    assert out.unified_diff == ""
    text = events.path.read_text(encoding="utf-8")
    assert "diff_touches_canon" in text
