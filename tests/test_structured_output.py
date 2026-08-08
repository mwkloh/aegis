"""Phase 8 Track B1 — `request_structured` wrapper contract.

Pins:

* Happy path: valid JSON on first call → `attempts=1, ok, not escalated`.
* Invalid JSON on first call, valid on retry → `attempts=2, ok`, corrective
  turn present, `llm.structured_retry` event emitted.
* Schema violation across all retries, no escalation → returns `{}` with
  `error_kind="schema_violation"` and `llm.structured_failed` event.
* Retries exhausted + escalation succeeds → `escalated=True`, `ok`,
  `llm.tier_escalated` event emitted.
* Escalation also fails → `{}` with final `error_kind`, escalated=True.
* Empty content handled as `error_kind="empty"`.
* Caller never sees a raised exception for bad model output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from runtime.events.stream import EventStream
from runtime.llm.clients.base import ChatMessage, ChatRequest, ChatResponse
from runtime.llm.structured_output import (
    EscalationTarget,
    request_structured,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeClient:
    """Replays a scripted sequence of `content` strings."""

    replies: list[str]
    seen: list[ChatRequest] = field(default_factory=list)
    model_echo: str = "fake-model"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.seen.append(request)
        idx = len(self.seen) - 1
        content = self.replies[idx] if idx < len(self.replies) else ""
        return ChatResponse(
            content=content,
            model=request.model,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
        )

    async def health(self) -> bool:
        return True


SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "confidence"],
    "properties": {
        "intent": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

BASE_MSGS = [ChatMessage(role="user", content="what's my name?")]


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def test_happy_path_first_try(tmp_path: Path) -> None:
    client = FakeClient(replies=['{"intent":"ask","confidence":0.9}'])
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        events=events,
        call_site="test.happy",
    )
    assert data == {"intent": "ask", "confidence": 0.9}
    assert outcome.attempts == 1
    assert outcome.escalated is False
    assert outcome.error_kind == "ok"
    assert outcome.final_model == "fake-7b"
    # No retry / failure events on happy path.
    assert _read_events(events.path) == []


async def test_invalid_json_then_valid_on_retry(tmp_path: Path) -> None:
    client = FakeClient(
        replies=[
            "sure thing {not json}",
            '{"intent":"ask","confidence":0.5}',
        ]
    )
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        events=events,
        call_site="test.retry",
    )
    assert data == {"intent": "ask", "confidence": 0.5}
    assert outcome.attempts == 2
    assert outcome.error_kind == "ok"

    # Second call must have an appended corrective system message
    # carrying the schema and failure reason.
    second = client.seen[1]
    assert len(second.messages) == len(BASE_MSGS) + 1
    corrective = second.messages[-1]
    assert corrective.role == "system"
    assert "json" in corrective.content.lower()
    assert "schema" in corrective.content.lower()
    assert "intent" in corrective.content  # schema leaked into prompt

    records = _read_events(events.path)
    assert len(records) == 1
    assert records[0]["type"] == "llm.structured_retry"
    assert records[0]["payload"]["error_kind"] == "invalid_json"
    assert records[0]["payload"]["call_site"] == "test.retry"


async def test_schema_violation_all_retries_no_escalation(tmp_path: Path) -> None:
    # Valid JSON but wrong shape on every attempt.
    bad = '{"intent":"ask"}'  # missing `confidence`
    client = FakeClient(replies=[bad, bad, bad])
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        max_retries=2,
        events=events,
        call_site="test.schema_fail",
    )
    assert data == {}
    assert outcome.attempts == 3  # 1 initial + 2 retries
    assert outcome.escalated is False
    assert outcome.error_kind == "schema_violation"

    records = _read_events(events.path)
    types = [r["type"] for r in records]
    assert types == [
        "llm.structured_retry",
        "llm.structured_retry",
        "llm.structured_failed",
    ]


async def test_escalation_recovers_after_primary_fails(tmp_path: Path) -> None:
    primary = FakeClient(replies=["nope", "still nope", "never"])
    smart = FakeClient(replies=['{"intent":"ask","confidence":0.7}'])
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        primary,
        BASE_MSGS,
        SCHEMA,
        model="fast-7b",
        max_retries=2,
        escalate_to=EscalationTarget(client=smart, model="smart-70b"),
        events=events,
        call_site="test.escalate",
    )
    assert data == {"intent": "ask", "confidence": 0.7}
    assert outcome.escalated is True
    assert outcome.error_kind == "ok"
    assert outcome.final_model == "smart-70b"
    assert outcome.attempts == 4  # 3 primary + 1 escalated

    records = _read_events(events.path)
    types = [r["type"] for r in records]
    assert "llm.tier_escalated" in types
    assert "llm.structured_failed" not in types
    # Escalated call must have received a corrective turn.
    escalated_req = smart.seen[0]
    assert escalated_req.model == "smart-70b"
    assert escalated_req.messages[-1].role == "system"
    # The escalated request must also carry the schema (Task 1 carry-over).
    assert escalated_req.response_schema == SCHEMA


async def test_escalation_recovers_with_repairable_content(tmp_path: Path) -> None:
    # Primary exhausts retries on genuinely broken content; the escalated
    # client's reply is a near-miss (fenced JSON) that repair_json should
    # salvage on the escalated call itself.
    primary = FakeClient(replies=["nope", "still nope", "never"])
    smart = FakeClient(
        replies=['```json\n{"intent":"ask","confidence":0.7}\n```']
    )
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        primary,
        BASE_MSGS,
        SCHEMA,
        model="fast-7b",
        max_retries=2,
        escalate_to=EscalationTarget(client=smart, model="smart-70b"),
        events=events,
        call_site="test.escalate_repair",
    )
    assert data == {"intent": "ask", "confidence": 0.7}
    assert outcome.escalated is True
    assert outcome.repaired is True
    assert outcome.error_kind == "ok"
    assert outcome.final_model == "smart-70b"

    records = _read_events(events.path)
    repaired_events = [r for r in records if r["type"] == "llm.json_repaired"]
    assert len(repaired_events) == 1
    # Locks in the deliberate attribution choice: the repaired event logs
    # the escalated model, since that's the client whose output was salvaged.
    assert repaired_events[0]["payload"] == {
        "call_site": "test.escalate_repair",
        "model": "smart-70b",
    }


async def test_escalation_also_fails(tmp_path: Path) -> None:
    primary = FakeClient(replies=["bad", "bad"])
    smart = FakeClient(replies=["still bad"])
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        primary,
        BASE_MSGS,
        SCHEMA,
        model="fast-7b",
        max_retries=1,
        escalate_to=EscalationTarget(client=smart, model="smart-70b"),
        events=events,
        call_site="test.all_fail",
    )
    assert data == {}
    assert outcome.escalated is True
    assert outcome.error_kind != "ok"
    records = _read_events(events.path)
    types = [r["type"] for r in records]
    assert types[-1] == "llm.structured_failed"
    assert "llm.tier_escalated" in types


async def test_empty_content_classified_as_empty(tmp_path: Path) -> None:
    client = FakeClient(replies=["", "   ", ""])
    events = EventStream(tmp_path)
    _, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        max_retries=2,
        events=events,
        call_site="test.empty",
    )
    assert outcome.error_kind == "empty"


async def test_request_uses_json_response_format(tmp_path: Path) -> None:
    client = FakeClient(replies=['{"intent":"ask","confidence":0.1}'])
    await request_structured(
        client, BASE_MSGS, SCHEMA, model="fake-7b", events=None
    )
    assert client.seen[0].response_format == "json"


async def test_request_threads_response_schema_to_client(tmp_path: Path) -> None:
    client = FakeClient(replies=['{"intent":"ask","confidence":0.1}'])
    await request_structured(
        client, BASE_MSGS, SCHEMA, model="fake-7b", events=None
    )
    assert client.seen[0].response_schema == SCHEMA
    # response_format stays "json" alongside the schema (fallback path).
    assert client.seen[0].response_format == "json"


async def test_no_events_when_stream_not_passed() -> None:
    # Must not raise even when all retries fail.
    client = FakeClient(replies=["bad", "bad", "bad"])
    _, outcome = await request_structured(
        client, BASE_MSGS, SCHEMA, model="fake-7b", max_retries=2
    )
    assert outcome.error_kind != "ok"


# --- Phase 11 Track A4 — deterministic JSON repair -------------------------


async def test_repairable_content_succeeds_without_retry(tmp_path: Path) -> None:
    # Fenced JSON is a near-miss the deterministic repair path should
    # salvage on the very first attempt — no retry burned.
    client = FakeClient(
        replies=['```json\n{"intent":"ask","confidence":0.9}\n```']
    )
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        events=events,
        call_site="test.repair",
    )
    assert data == {"intent": "ask", "confidence": 0.9}
    assert outcome.attempts == 1
    assert outcome.error_kind == "ok"
    assert outcome.repaired is True

    records = _read_events(events.path)
    types = [r["type"] for r in records]
    assert types == ["llm.json_repaired"]
    assert records[0]["payload"] == {"call_site": "test.repair", "model": "fake-7b"}


async def test_unrepairable_content_still_walks_retry_path(tmp_path: Path) -> None:
    # Braces present but the interior isn't valid JSON — repair_json can't
    # salvage this, so the existing retry/escalation behavior must be
    # unchanged.
    client = FakeClient(
        replies=[
            "sure thing {not json}",
            '{"intent":"ask","confidence":0.5}',
        ]
    )
    events = EventStream(tmp_path)
    data, outcome = await request_structured(
        client,
        BASE_MSGS,
        SCHEMA,
        model="fake-7b",
        events=events,
        call_site="test.unrepairable_retry",
    )
    assert data == {"intent": "ask", "confidence": 0.5}
    assert outcome.attempts == 2
    assert outcome.error_kind == "ok"
    assert outcome.repaired is False

    records = _read_events(events.path)
    types = [r["type"] for r in records]
    assert types == ["llm.structured_retry"]
    assert "llm.json_repaired" not in types


async def test_repaired_outcome_defaults_false_on_plain_success(tmp_path: Path) -> None:
    client = FakeClient(replies=['{"intent":"ask","confidence":0.9}'])
    _, outcome = await request_structured(
        client, BASE_MSGS, SCHEMA, model="fake-7b", events=None
    )
    assert outcome.repaired is False
