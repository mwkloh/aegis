"""Tests for `ModelBackedClassifier`.

Verifies:
  - rule-first short-circuit (no model call when a rule fires),
  - Ollama-fallback when no rule matches,
  - strict JSON parser rejects garbage / unknown intents,
  - graceful degradation to `unknown` on transport failure,
  - B1 corrective retry recovers when the first reply is malformed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from runtime.events.stream import EventStream
from runtime.intent import (
    ClassifierUnavailableError,
    IntentClassification,
    ModelBackedClassifier,
    parse_intent_json,
)
from runtime.llm.clients import ChatRequest, ChatResponse


class _StubClient:
    """Async stub mirroring the ModelClient surface."""

    def __init__(self, *, content: str = "", raises: Exception | None = None) -> None:
        self._content = content
        self._raises = raises
        self.calls: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        if self._raises is not None:
            raise self._raises
        return ChatResponse(content=self._content, model=request.model)

    async def health(self) -> bool:
        return True


_KNOWN = ("echo", "ping", "ask_time", "ask_question")


def _classifier(client: Any) -> ModelBackedClassifier:
    return ModelBackedClassifier(client=client, model="gemma4:e2b", known_intents=_KNOWN)


@pytest.mark.asyncio
async def test_rule_match_skips_model_call() -> None:
    stub = _StubClient(content='{"intent": "ask_time", "confidence": 0.9}')
    classifier = _classifier(stub)

    result = await classifier.classify("ping")

    assert result.intent == "ping"
    assert stub.calls == []  # model must not be called


@pytest.mark.asyncio
async def test_falls_back_to_model_when_no_rule_fires() -> None:
    stub = _StubClient(content='{"intent": "ask_time", "confidence": 0.81}')
    classifier = _classifier(stub)

    result = await classifier.classify("what time is it in Tokyo?")

    assert result == IntentClassification(intent="ask_time", confidence=0.81)
    assert len(stub.calls) == 1
    request = stub.calls[0]
    assert request.model == "gemma4:e2b"
    assert request.response_format == "json"
    assert request.messages[0].role == "system"
    assert "ask_time" in request.messages[0].content
    assert request.messages[1].content == "what time is it in Tokyo?"
    # A thinking-capable model can burn its whole max_tokens budget on a
    # hidden reasoning channel and never emit content -- confirmed live
    # against gemma4:e2b-mlx (2026-08-22). The classifier's task is trivial
    # enough to not need it, so it must be explicitly disabled.
    assert request.think is False


@pytest.mark.asyncio
async def test_oversized_input_is_truncated_before_send() -> None:
    stub = _StubClient(content='{"intent": "unknown", "confidence": 0.0}')
    classifier = _classifier(stub)

    huge = "x" * 9000
    await classifier.classify(huge)

    user_msg = stub.calls[0].messages[1].content
    assert len(user_msg) <= 4096


@pytest.mark.asyncio
async def test_intent_outside_schema_enum_raises_unavailable() -> None:
    # "rocket_launch" isn't in known_intents plus {"unknown"}, so it trips the
    # JSON-schema enum on every retry — a schema failure, not a genuine
    # model answer, so it must be distinguishable from a real "unknown".
    stub = _StubClient(content='{"intent": "rocket_launch", "confidence": 0.99}')
    classifier = _classifier(stub)

    with pytest.raises(ClassifierUnavailableError):
        await classifier.classify("anything novel here")


@pytest.mark.asyncio
async def test_model_explicit_unknown_returns_normally() -> None:
    # A genuine model-produced "unknown" is a real classification, not a
    # failure — must NOT raise, unlike the schema/transport failure cases
    # below.
    stub = _StubClient(content='{"intent": "unknown", "confidence": 0.0}')
    classifier = _classifier(stub)

    result = await classifier.classify("anything novel here")

    assert result == IntentClassification(intent="unknown", confidence=0.0)


@pytest.mark.asyncio
async def test_garbage_response_raises_unavailable() -> None:
    stub = _StubClient(content="this is not JSON at all, sorry")
    classifier = _classifier(stub)

    with pytest.raises(ClassifierUnavailableError):
        await classifier.classify("anything novel")


@pytest.mark.asyncio
async def test_transport_failure_raises_unavailable() -> None:
    stub = _StubClient(raises=httpx.ConnectError("refused"))
    classifier = _classifier(stub)

    with pytest.raises(ClassifierUnavailableError):
        await classifier.classify("anything novel")


def test_parser_rejects_extra_keys_silently() -> None:
    raw = '{"intent": "ask_time", "confidence": 0.9, "rationale": "guess"}'
    # extra fields are ignored — we only require the two we need
    result = parse_intent_json(raw, _KNOWN)
    assert result.intent == "ask_time"


def test_parser_rejects_confidence_out_of_range() -> None:
    raw = '{"intent": "ask_time", "confidence": 1.5}'
    result = parse_intent_json(raw, _KNOWN)
    assert result.intent == "unknown"


def test_parser_rejects_bool_confidence() -> None:
    raw = '{"intent": "ask_time", "confidence": true}'
    result = parse_intent_json(raw, _KNOWN)
    assert result.intent == "unknown"


def test_parser_accepts_explicit_unknown() -> None:
    raw = '{"intent": "unknown", "confidence": 0.0}'
    result = parse_intent_json(raw, _KNOWN)
    assert result.intent == "unknown"


class _ScriptedClient:
    """Async stub that replays a scripted sequence of replies."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        idx = len(self.calls)
        self.calls.append(request)
        content = self._replies[idx] if idx < len(self._replies) else ""
        return ChatResponse(content=content, model=request.model)

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_malformed_first_reply_recovers_via_corrective_retry(
    tmp_path: Path,
) -> None:
    client = _ScriptedClient(
        replies=[
            "sure thing {not json}",
            '{"intent": "ask_time", "confidence": 0.8}',
        ]
    )
    events = EventStream(tmp_path)
    classifier = ModelBackedClassifier(
        client=client,
        model="gemma4:e2b",
        known_intents=_KNOWN,
        events=events,
        max_retries=1,
    )

    result = await classifier.classify("what time is it in Tokyo?")

    assert result == IntentClassification(intent="ask_time", confidence=0.8)
    assert len(client.calls) == 2
    # Second request carries a corrective system turn appended by the wrapper.
    assert client.calls[1].messages[-1].role == "system"
    assert "schema" in client.calls[1].messages[-1].content.lower()

    records = [
        json.loads(line)
        for line in events.path.read_text().splitlines()
        if line.strip()
    ]
    assert any(r["type"] == "llm.structured_retry" for r in records)
    assert records[0]["payload"]["call_site"] == "intent.classifier"
