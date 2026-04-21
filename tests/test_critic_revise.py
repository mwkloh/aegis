"""Phase 5 Track B — `critic.critique_then_revise`.

Pins the bounded critique-then-revise contract:

* critique returns issues  → revise called → revised draft returned
* critique returns no issues → revise SKIPPED → original draft returned
* critique fails (HTTP/JSON/schema) → original draft returned (graceful)
* revise fails after a non-empty critique → original draft returned
* revise output that touches canon → returned as ``status="refused"``
* draft is not "ok" / has empty diff / client is None → fast no-op
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from runtime.coding_harness.critic import critique_then_revise
from runtime.coding_harness.draft import Draft
from runtime.config import get_config
from runtime.events import EventStream
from runtime.improvement.coding_tasks import CodingTask
from runtime.model_router.clients import OllamaClient

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 4, 19, 10, 0, tzinfo=UTC)
_OLLAMA = "http://127.0.0.1:11434/api/chat"


# --- factories ---------------------------------------------------------------


def _task(scope: list[str] | None = None) -> CodingTask:
    return CodingTask(
        ct_id="CT-009",
        imp_id="IMP-cafef00d",
        scope=scope or ["runtime/intent/classifier.py"],
        constraints="do not modify canon; produce diffs only",
        expected_output="Add an 'echo' alias to the intent classifier.",
        queued_at=_NOW,
    )


def _ok_draft(diff: str | None = None) -> Draft:
    return Draft(
        ct_id="CT-009",
        imp_id="IMP-cafef00d",
        model="gemma4:e4b",
        drafted_at=_NOW,
        summary="initial draft",
        unified_diff=diff if diff is not None else (
            "--- a/runtime/intent/classifier.py\n"
            "+++ b/runtime/intent/classifier.py\n"
            "@@\n-OLD\n+NEW\n"
        ),
        test_notes="run pytest",
        rollback="git revert HEAD",
        status="ok",
        reason="",
    )


def _critique(issues: list[str]) -> dict[str, Any]:
    return {"message": {"content": json.dumps({"issues": issues})}}


def _revise(
    *,
    summary: str = "revised draft",
    diff: str = (
        "--- a/runtime/intent/classifier.py\n"
        "+++ b/runtime/intent/classifier.py\n"
        "@@\n-OLD\n+REVISED\n"
    ),
    test_notes: str = "run pytest -k echo",
    rollback: str = "git branch -D aegis/CT-009-cafef00dd00d",
) -> dict[str, Any]:
    return {"message": {"content": json.dumps({
        "summary": summary,
        "unified_diff": diff,
        "test_notes": test_notes,
        "rollback": rollback,
    })}}


def _client() -> OllamaClient:
    return OllamaClient(get_config())


# --- happy paths -------------------------------------------------------------


@pytest.mark.asyncio
async def test_critique_returns_issues_triggers_revise(tmp_path: Path) -> None:
    """Critic flags issues → revise runs → revised draft replaces original."""
    events = EventStream(tmp_path / "sessions")
    responses = [
        httpx.Response(200, json=_critique(["rollback is too vague"])),
        httpx.Response(200, json=_revise()),
    ]
    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(side_effect=responses)
        out = await critique_then_revise(
            _ok_draft(), _task(), context=None,
            client=_client(), model="gemma4:e4b",
            events=events, when=_NOW,
        )
    assert out.status == "ok"
    assert out.summary == "revised draft"
    assert "+REVISED" in out.unified_diff
    text = events.path.read_text(encoding="utf-8")
    assert "harness.critique.start" in text
    assert "harness.critique.end" in text
    assert "harness.revise.start" in text
    assert "harness.revise.end" in text


@pytest.mark.asyncio
async def test_critique_no_issues_skips_revise(tmp_path: Path) -> None:
    """Empty issue list → original draft returned, NO revise call made."""
    events = EventStream(tmp_path / "sessions")
    original = _ok_draft()
    with respx.mock() as mock:
        route = mock.post(_OLLAMA).mock(
            return_value=httpx.Response(200, json=_critique([])),
        )
        out = await critique_then_revise(
            original, _task(), context=None,
            client=_client(), model="gemma4:e4b",
            events=events, when=_NOW,
        )
    assert out is original  # exact identity — no rebuild
    assert route.call_count == 1  # critique only, no revise call
    text = events.path.read_text(encoding="utf-8")
    assert "harness.revise.start" not in text
    assert '"status":"skipped"' in text.replace(" ", "")


# --- graceful degradation ----------------------------------------------------


@pytest.mark.asyncio
async def test_critique_http_failure_returns_original(tmp_path: Path) -> None:
    events = EventStream(tmp_path / "sessions")
    original = _ok_draft()
    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(side_effect=httpx.ConnectError("nope"))
        out = await critique_then_revise(
            original, _task(), context=None,
            client=_client(), model="gemma4:e4b",
            events=events, when=_NOW,
        )
    assert out is original
    text = events.path.read_text(encoding="utf-8")
    assert '"status":"failed"' in text.replace(" ", "")


@pytest.mark.asyncio
async def test_critique_malformed_json_returns_original() -> None:
    original = _ok_draft()
    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(
            return_value=httpx.Response(
                200, json={"message": {"content": "not json"}},
            ),
        )
        out = await critique_then_revise(
            original, _task(), context=None,
            client=_client(), model="gemma4:e4b",
            when=_NOW,
        )
    assert out is original


@pytest.mark.asyncio
async def test_critique_extra_keys_returns_original() -> None:
    """Pydantic extra=forbid: stray fields force graceful degradation."""
    original = _ok_draft()
    bad = json.dumps({"issues": [], "smuggled": "extra"})
    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(
            return_value=httpx.Response(200, json={"message": {"content": bad}}),
        )
        out = await critique_then_revise(
            original, _task(), context=None,
            client=_client(), model="gemma4:e4b",
            when=_NOW,
        )
    assert out is original


@pytest.mark.asyncio
async def test_revise_failure_returns_original(tmp_path: Path) -> None:
    """Critique flags issues, revise crashes → original draft preserved."""
    events = EventStream(tmp_path / "sessions")
    original = _ok_draft()
    calls: dict[str, int] = {"n": 0}

    def _side_effect(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_critique(["issue"]))
        raise httpx.ConnectError("boom")

    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(side_effect=_side_effect)
        out = await critique_then_revise(
            original, _task(), context=None,
            client=_client(), model="gemma4:e4b",
            events=events, when=_NOW,
        )
    assert out is original
    # Critique + at least one revise attempt. Ollama client retries
    # ConnectError up to 3 times via tenacity, so the total may be > 2.
    assert calls["n"] >= 2
    text = events.path.read_text(encoding="utf-8")
    assert "harness.revise.start" in text
    assert '"status":"failed"' in text.replace(" ", "")


@pytest.mark.asyncio
async def test_revise_touching_canon_is_refused(tmp_path: Path) -> None:
    """Revise that proposes a write to canon → returned as refused."""
    events = EventStream(tmp_path / "sessions")
    canon_diff = (
        "--- a/AGENTS.md\n"
        "+++ b/AGENTS.md\n"
        "@@\n-OLD\n+SNEAKY\n"
    )
    responses = [
        httpx.Response(200, json=_critique(["please clarify"])),
        httpx.Response(200, json=_revise(diff=canon_diff)),
    ]
    with respx.mock() as mock:
        mock.post(_OLLAMA).mock(side_effect=responses)
        out = await critique_then_revise(
            _ok_draft(), _task(), context=None,
            client=_client(), model="gemma4:e4b",
            events=events, when=_NOW,
        )
    assert out.status == "refused"
    assert out.unified_diff == ""
    assert "AGENTS.md" in out.reason
    text = events.path.read_text(encoding="utf-8")
    assert "harness.refused" in text


# --- fast no-op preconditions ------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_client_is_none() -> None:
    original = _ok_draft()
    out = await critique_then_revise(
        original, _task(), context=None,
        client=None, model="gemma4:e4b", when=_NOW,
    )
    assert out is original


@pytest.mark.asyncio
async def test_no_op_when_draft_status_not_ok() -> None:
    """Stub/refused drafts have nothing to critique."""
    refused = _ok_draft().model_copy(update={
        "status": "refused", "unified_diff": "", "reason": "scope hit",
    })
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(_OLLAMA).mock(
            return_value=httpx.Response(200, json=_critique(["x"])),
        )
        out = await critique_then_revise(
            refused, _task(), context=None,
            client=_client(), model="gemma4:e4b", when=_NOW,
        )
    assert out is refused
    assert route.call_count == 0  # no model calls at all


@pytest.mark.asyncio
async def test_no_op_when_diff_is_empty() -> None:
    empty = _ok_draft(diff="")
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(_OLLAMA).mock(
            return_value=httpx.Response(200, json=_critique(["x"])),
        )
        out = await critique_then_revise(
            empty, _task(), context=None,
            client=_client(), model="gemma4:e4b", when=_NOW,
        )
    assert out is empty
    assert route.call_count == 0
