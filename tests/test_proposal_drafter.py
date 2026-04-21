"""Reflection proposal drafter — Ollama mocked via respx."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from runtime.config import get_config
from runtime.events import EventStream
from runtime.model_router.clients import OllamaClient
from runtime.reflection.patterns import PatternRecord
from runtime.reflection.proposals import Proposal, draft

pytestmark = pytest.mark.unit


def _pattern() -> PatternRecord:
    return PatternRecord(
        detector="unknown_intent",
        severity="medium",
        count=4,
        sample_session_ids=["s1"],
        summary="unknown 4 times",
    )


@pytest.mark.asyncio
async def test_draft_returns_stub_when_no_client() -> None:
    out = await draft([_pattern()], client=None, model="gemma4:e4b")
    assert len(out) == 1
    assert out[0].id == "P-001"
    assert "manual review" in out[0].change
    assert out[0].risk == "low"


@pytest.mark.asyncio
async def test_draft_parses_well_formed_json() -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    reply_json = json.dumps({
        "affected": ["intent_classifier"],
        "change": "Add an 'echo' keyword alias.",
        "risk": "medium",
        "rationale": "Most unknowns look like typos of echo.",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": reply_json}})
        )
        out = await draft([_pattern()], client=client, model="gemma4:e4b")

    assert len(out) == 1
    p = out[0]
    assert isinstance(p, Proposal)
    assert p.affected == ["intent_classifier"]
    assert p.change.startswith("Add an")
    assert p.risk == "medium"


@pytest.mark.asyncio
async def test_draft_falls_back_to_stub_on_malformed_json(tmp_path: Path) -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    events = EventStream(tmp_path / "sessions")
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "not json at all"}})
        )
        out = await draft([_pattern()], client=client, model="gemma4:e4b", events=events)

    assert len(out) == 1
    assert "parse failed" in out[0].change
    text = events.path.read_text(encoding="utf-8")
    assert '"pattern":"proposal_parse_failed"' in text.replace(" ", "")


@pytest.mark.asyncio
async def test_draft_rejects_extra_keys_via_pydantic() -> None:
    cfg = get_config()
    client = OllamaClient(cfg)
    bad = json.dumps({
        "affected": [], "change": "ok", "risk": "low",
        "rationale": "", "smuggled": "extra",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": bad}})
        )
        out = await draft([_pattern()], client=client, model="gemma4:e4b")

    assert len(out) == 1
    assert "parse failed" in out[0].change


@pytest.mark.asyncio
async def test_draft_propagates_per_pattern_isolation() -> None:
    """One bad pattern doesn't poison the next one in the same batch."""
    cfg = get_config()
    client = OllamaClient(cfg)
    good = json.dumps({
        "affected": ["x"], "change": "tighten X", "risk": "low",
        "rationale": "",
    })
    with respx.mock() as mock:
        route = mock.post("http://127.0.0.1:11434/api/chat")
        route.side_effect = [
            httpx.Response(200, json={"message": {"content": "garbage"}}),
            httpx.Response(200, json={"message": {"content": good}}),
        ]
        patterns = [
            _pattern(),
            PatternRecord(
                detector="tool_error", severity="high", count=3,
                sample_session_ids=["a"], summary="echo failed 3 times",
            ),
        ]
        out = await draft(patterns, client=client, model="gemma4:e4b")

    assert len(out) == 2
    assert "parse failed" in out[0].change
    assert out[1].change == "tighten X"
    assert out[1].pattern_detector == "tool_error"
