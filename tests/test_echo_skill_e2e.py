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
    # "good morning" fires no rule, so the model-backed classifier would call
    # Ollama. We mock a malformed reply so it collapses to `unknown` and the
    # pipeline falls through to the help message — deterministic either way.
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "not json"}})
        )
        reply = await pipeline.handle("good morning")

    assert "echo hello" in reply or "ping" in reply
    types = [e["type"] for e in _read_jsonl(pipeline.events.path)]
    assert "user.message" in types
    assert "assistant.reply" in types
    assert "skill.selected" not in types
