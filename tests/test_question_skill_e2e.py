"""End-to-end: ask_question skill needs Tier 1.

Two scenarios:
  - OPENROUTER_API_KEY present → pipeline uses the mocked Tier 1 reasoner.
  - Key absent → pipeline degrades to "tier1 unavailable" via respond tool.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from runtime.chat.cli import build_pipeline
from runtime.config import reset_config

pytestmark = pytest.mark.e2e


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_ask_question_tier1_end_to_end(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    reset_config()  # re-resolve with the test key
    pipeline = build_pipeline()

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": '{"intent": "ask_question", "confidence": 0.88}',
                    }
                },
            )
        )
        mock.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    '{"args": {"message": "Paris is the capital of France."}, '
                                    '"rationale": "fact lookup"}'
                                ),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 12},
                },
            )
        )
        reply = await pipeline.handle("What is the capital of France?")

    assert "Paris" in reply
    events = _read_jsonl(pipeline.events.path)
    skill_event = next(e for e in events if e["type"] == "skill.selected")
    assert skill_event["payload"]["skill_id"] == "ask_question"
    contract_event = next(e for e in events if e["type"] == "contract.emitted")
    assert contract_event["payload"]["tool"] == "respond"


@pytest.mark.asyncio
async def test_ask_question_without_key_degrades_gracefully(
    aegis_sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_config()
    pipeline = build_pipeline()

    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": '{"intent": "ask_question", "confidence": 0.88}',
                    }
                },
            )
        )
        # No OpenRouter mock registered — any egress would fail the test.
        reply = await pipeline.handle("What is the capital of France?")

    assert "tier1 unavailable" in reply
    events = _read_jsonl(pipeline.events.path)
    contract_event = next(e for e in events if e["type"] == "contract.emitted")
    assert contract_event["payload"]["tool"] == "respond"
