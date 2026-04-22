"""Coder LLM integration with stub-on-failure."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from runtime.coding_harness.coder import draft_for
from runtime.coding_harness.draft import Draft
from runtime.config import get_config
from runtime.events import EventStream
from runtime.improvement.coding_tasks import CodingTask
from runtime.llm.clients import OllamaClient

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 4, 18, 12, 7, tzinfo=UTC)


def _task(scope: list[str] | None = None) -> CodingTask:
    return CodingTask(
        ct_id="CT-001",
        imp_id="IMP-a86b087a",
        scope=scope or ["runtime/intent/classifier.py"],
        constraints="do not modify canon; produce diffs only",
        expected_output="Add an 'echo' alias to the intent classifier.",
        queued_at=_NOW,
    )


def _good_reply() -> str:
    return json.dumps({
        "summary": "Add 'echo' alias to intent classifier.",
        "unified_diff": (
            "--- a/runtime/intent/classifier.py\n"
            "+++ b/runtime/intent/classifier.py\n"
            "@@\n-OLD\n+NEW\n"
        ),
        "test_notes": "Verify echo intent fires for 'hi there'.",
        "rollback": "git revert HEAD",
    })


@pytest.mark.asyncio
async def test_returns_stub_when_client_is_none() -> None:
    out = await draft_for(_task(), client=None, model="m", when=_NOW)
    assert isinstance(out, Draft)
    assert out.status == "stub"
    assert "not configured" in out.reason
    assert out.unified_diff == ""


@pytest.mark.asyncio
async def test_returns_ok_on_well_formed_reply() -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": _good_reply()}})
        )
        out = await draft_for(_task(), client=client, model="gemma4:e4b", when=_NOW)
    assert out.status == "ok"
    assert "alias" in out.summary.lower()
    assert "+++ b/runtime/intent/classifier.py" in out.unified_diff


@pytest.mark.asyncio
async def test_stubs_on_malformed_json(tmp_path: Path) -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    events = EventStream(tmp_path / "sessions")
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "not json"}})
        )
        out = await draft_for(
            _task(), client=client, model="gemma4:e4b", events=events, when=_NOW
        )
    assert out.status == "stub"
    text = events.path.read_text(encoding="utf-8")
    assert '"status":"stub"' in text.replace(" ", "")


@pytest.mark.asyncio
async def test_stubs_on_extra_keys() -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    bad = json.dumps({
        "summary": "x", "unified_diff": "y",
        "test_notes": "z", "rollback": "r",
        "smuggled": "extra",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": bad}})
        )
        out = await draft_for(_task(), client=client, model="gemma4:e4b", when=_NOW)
    assert out.status == "stub"


@pytest.mark.asyncio
async def test_stubs_on_http_error() -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            side_effect=httpx.ConnectError("nope")
        )
        out = await draft_for(_task(), client=client, model="gemma4:e4b", when=_NOW)
    assert out.status == "stub"
    assert "ConnectError" in out.reason
