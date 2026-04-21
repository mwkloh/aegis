"""End-to-end reflection run against a fixture session directory."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from runtime.config import get_config
from runtime.reflection.cli import main as reflection_main

pytestmark = pytest.mark.e2e


def _seed_session(sessions_dir: Path) -> None:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    day_dir = sessions_dir / today
    day_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"ts": datetime.now(tz=UTC).isoformat(), "session_id": f"s{i}",
         "type": "intent.classified", "payload": {"intent": "unknown", "confidence": 0.2}}
        for i in range(4)
    ]
    (day_dir / "fixture.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8",
    )


def test_reflection_cli_writes_both_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    _seed_session(cfg.storage.sessions_dir)
    proposal_json = json.dumps({
        "affected": ["intent_classifier"],
        "change": "Add more keyword aliases for short phrases.",
        "risk": "low",
        "rationale": "Most unknowns are short greetings.",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": proposal_json}})
        )
        rc = reflection_main([])
    assert rc == 0

    out = capsys.readouterr().out
    assert "patterns: 2 detected" in out  # unknown_intent + low_confidence
    assert "proposals: 2 drafted" in out

    patterns_md = (cfg.aegis_home / "reflection" / "PATTERNS.md").read_text(encoding="utf-8")
    proposals_md = (cfg.aegis_home / "reflection" / "PROPOSALS.md").read_text(encoding="utf-8")
    assert "unknown_intent" in patterns_md
    assert "low_confidence" in patterns_md
    assert "Add more keyword aliases" in proposals_md
    assert "P-001" in proposals_md
    assert "P-002" in proposals_md


def test_reflection_cli_dry_run_writes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = get_config()
    _seed_session(cfg.storage.sessions_dir)
    proposal_json = json.dumps({
        "affected": [], "change": "noop", "risk": "low", "rationale": "",
    })
    with respx.mock() as mock:
        mock.post("http://127.0.0.1:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": proposal_json}})
        )
        rc = reflection_main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not (cfg.aegis_home / "reflection" / "PATTERNS.md").exists()
    assert not (cfg.aegis_home / "reflection" / "PROPOSALS.md").exists()
