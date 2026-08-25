"""End-to-end vertical slice: CLI input → intent → skill → harness → tool → reply."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from runtime.chat.cli import build_pipeline
from runtime.config import AegisConfig, SkillsConfig, get_config
from runtime.skills.bootstrap import seed_builtin_skills

pytestmark = pytest.mark.e2e

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_CATALOG = _REPO_ROOT / "runtime" / "skills" / "catalog"
_BUNDLE = _REPO_ROOT / "runtime" / "skills" / "_bundle"


def _cfg_with_repo_catalog(tmp_path: Path) -> AegisConfig:
    """Seed the bundle + copy flat catalog into ``tmp_path`` and point config there."""
    catalog_dir = tmp_path / "skills_catalog"
    seed_builtin_skills(bundle_dir=_BUNDLE, catalog_dir=catalog_dir)
    for yaml_file in _REPO_CATALOG.glob("*.yaml"):
        target = catalog_dir / yaml_file.name
        if not target.exists():
            target.write_bytes(yaml_file.read_bytes())
    return get_config().model_copy(
        update={"skills": SkillsConfig(catalog_dir=catalog_dir)}
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_echo_round_trip_writes_full_event_chain(aegis_sandbox: Path) -> None:
    pipeline = build_pipeline(config=_cfg_with_repo_catalog(aegis_sandbox))
    # Echo matches a Tier 0 rule — no model call leaves the process.
    reply = await pipeline.handle("echo hello world")
    assert reply == "echo → hello world"

    events = _read_jsonl(pipeline.events.path)
    # build_pipeline() may emit a construction-time `pattern.observed`
    # (e.g. tier1_missing when OPENROUTER_API_KEY is absent). The handle()
    # sequence is what we assert on — slice from the first user message.
    start = next(i for i, e in enumerate(events) if e["type"] == "user.message")
    handle_events = events[start:]
    types = [e["type"] for e in handle_events]
    assert types == [
        "user.message",
        "intent.classified",
        "skill.selected",
        "contract.emitted",
        "tool.invoked",
        "tool.result",
        "assistant.reply",
    ]

    intent_event = handle_events[1]["payload"]
    assert intent_event["intent"] == "echo"
    assert float(intent_event["confidence"]) >= 0.9

    tool_result = handle_events[5]["payload"]
    assert tool_result["status"] == "ok"
    assert tool_result["payload"] == {"echoed": "hello world", "length": 11}


@pytest.mark.asyncio
async def test_unknown_intent_replies_with_help(aegis_sandbox: Path) -> None:
    pipeline = build_pipeline(config=_cfg_with_repo_catalog(aegis_sandbox))
    # "good morning" fires no rule, so the model-backed classifier calls Ollama.
    # Mock a *well-formed* reply carrying a genuine "unknown" so the pipeline
    # falls through to the help message.
    #
    # This previously mocked malformed JSON as a shortcut to the same place.
    # Since 27bc7ea that is no longer equivalent: a malformed reply is an
    # outage (ClassifierUnavailableError), not a classification. Keeping the
    # old mock here would have this test silently exercising the outage path
    # under a name that claims to cover unknown-intent — the two are covered
    # separately now.
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {"content": '{"intent": "unknown", "confidence": 0.0}'}
                },
            )
        )
        reply = await pipeline.handle("good morning")

    assert "echo hello" in reply or "ping" in reply
    types = [e["type"] for e in _read_jsonl(pipeline.events.path)]
    assert "user.message" in types
    assert "assistant.reply" in types
    assert "skill.selected" not in types


@pytest.mark.asyncio
async def test_classifier_outage_degrades_instead_of_crashing(aegis_sandbox: Path) -> None:
    """A classifier outage must not propagate out of the CLI pipeline.

    `27bc7ea` made ModelBackedClassifier raise ClassifierUnavailableError rather
    than collapsing an outage into intent="unknown" — so an outage stops failing
    open into the full-catalog planner. It wired the guard into the Telegram
    dispatcher (`except Exception: return PASS`) but not into this surface, so
    `Pipeline.handle()` had no handler and the exception reached the caller.
    """
    pipeline = build_pipeline(config=_cfg_with_repo_catalog(aegis_sandbox))
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "not json"}})
        )
        reply = await pipeline.handle("good morning")

    assert reply  # did not raise
    types = [e["type"] for e in _read_jsonl(pipeline.events.path)]
    assert "assistant.reply" in types
    # An outage is not a classification — nothing may claim an intent was found.
    assert "skill.selected" not in types
    assert "intent.classified" not in types
