"""End-to-end: model-classified intent → ask_time skill → time tool → reply.

All network is mocked with respx — the Ollama classifier sees a canned reply
that forces `ask_time`, the rest of the pipeline runs real code.
"""
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
async def test_ask_time_tokyo_end_to_end(aegis_sandbox: Path) -> None:
    pipeline = build_pipeline(config=_cfg_with_repo_catalog(aegis_sandbox))

    with respx.mock(assert_all_called=True) as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": '{"intent": "ask_time", "confidence": 0.92}',
                    },
                    "prompt_eval_count": 30,
                    "eval_count": 12,
                },
            )
        )
        reply = await pipeline.handle("what time is it in Tokyo?")

    assert "Tokyo" in reply or "Asia/Tokyo" in reply

    events = _read_jsonl(pipeline.events.path)
    types = [e["type"] for e in events]
    # Full chain present — user → intent → skill → contract → tool → result → reply.
    for expected in (
        "user.message",
        "intent.classified",
        "skill.selected",
        "contract.emitted",
        "tool.invoked",
        "tool.result",
        "assistant.reply",
    ):
        assert expected in types, f"missing event: {expected}"

    # Skill + tool wired correctly.
    skill_event = next(e for e in events if e["type"] == "skill.selected")
    assert skill_event["payload"]["skill_id"] == "ask_time"
    contract_event = next(e for e in events if e["type"] == "contract.emitted")
    assert contract_event["payload"]["tool"] == "time"
    tool_result = next(e for e in events if e["type"] == "tool.result")
    assert tool_result["payload"]["status"] == "ok"
    payload = tool_result["payload"]["payload"]
    assert payload["zone"] == "Asia/Tokyo"
