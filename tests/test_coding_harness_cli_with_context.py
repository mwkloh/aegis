"""Phase 5 Track B — coding harness CLI ``--with-context`` flag.

Pins the contract:

* Default run (no flag) is unchanged from Phase 4 — no context-mode
  banner, no ``pattern.observed`` event with ``harness_with_context``.
* ``--with-context`` gathers the bundle, prints the banner, emits a
  ``pattern.observed`` event with ``{ct_id, imp_id, files, skills,
  total_bytes, truncated}`` and runs the bounded critique pass.
* ``--with-context`` works combined with ``--task CT-NNN``.

Tests use stub mode (no OPENROUTER, non-loopback Ollama) so the
critique-then-revise pass is short-circuited by the ``client is None``
fast no-op. The flag plumbing and the ``pattern.observed`` event
shape are what we are pinning here — model behaviour itself is
covered by ``test_critic_revise.py``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from runtime.coding_harness.cli import main as harness_main
from runtime.coding_harness.patch_writer import existing_drafts_for
from runtime.config import get_config
from runtime.improvement.cli import main as improvement_main
from runtime.improvement.proposal_loader import derive_imp_id
from runtime.reflection.event_reader import ReadStats
from runtime.reflection.proposals import Proposal
from runtime.reflection.writer import write_proposals

pytestmark = pytest.mark.unit

_STATS = ReadStats(sessions=1, events=1, skipped=0)


def _seed_eligible(workspace: Path) -> str:
    proposal = Proposal(
        id="P-001",
        pattern_detector="unknown_intent",
        affected=["intent_classifier"],
        change="Add aliases for 'hi'.",
        risk="low",
        rationale="r",
    )
    write_proposals(
        workspace, [proposal], _STATS, when=datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    )
    imp_id = derive_imp_id(
        proposal.pattern_detector, list(proposal.affected), proposal.change
    )
    improvement_main([
        "--decide", imp_id, "--verdict", "approve", "--rationale", "ok",
    ])
    return imp_id


def _events_text(cfg_home: Path) -> str:
    sessions = cfg_home / "sessions"
    out: list[str] = []
    for path in sorted(sessions.rglob("*.jsonl")):
        out.append(path.read_text(encoding="utf-8"))
    return "".join(out)


def _harness_with_context_payloads(cfg_home: Path) -> list[dict[str, object]]:
    """Return every ``pattern.observed`` payload tagged ``harness_with_context``."""
    payloads: list[dict[str, object]] = []
    for raw in _events_text(cfg_home).splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "pattern.observed":
            continue
        payload = event.get("payload") or {}
        if payload.get("pattern") == "harness_with_context":
            payloads.append(payload)
    return payloads


def test_default_run_does_not_emit_with_context_event(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 4 path: no banner, no ``harness_with_context`` event."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()

    rc = harness_main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "context-mode ON" not in out
    assert "context-mode OFF" in out
    assert _harness_with_context_payloads(cfg.aegis_home) == []


def test_with_context_flag_emits_pattern_event(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--with-context`` gathers the bundle and emits the right event shape."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")
    cfg = get_config()
    imp_id = _seed_eligible(cfg.aegis_home)

    # Provide a real in-scope file so gather_context returns a non-empty bundle.
    src = tmp_path / "runtime" / "intent"
    src.mkdir(parents=True)
    target = src / "classifier.py"
    target.write_text("# tiny in-scope file\nVALUE = 42\n", encoding="utf-8")

    capsys.readouterr()
    rc = harness_main(["--with-context", "--repo-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "context-mode ON" in out
    assert "gathered" in out
    assert "in-scope file" in out

    payloads = _harness_with_context_payloads(cfg.aegis_home)
    assert len(payloads) == 1
    p = payloads[0]
    assert p["pattern"] == "harness_with_context"
    assert p["ct_id"] == "CT-001"
    assert p["imp_id"] == imp_id
    assert isinstance(p["files"], int)
    assert isinstance(p["skills"], int)
    assert isinstance(p["total_bytes"], int)
    assert isinstance(p["truncated"], bool)

    # Stub mode → still writes a draft so the run completes cleanly.
    drafts = existing_drafts_for(cfg.aegis_home, "CT-001")
    assert len(drafts) == 1


def test_with_context_combined_with_task_flag(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--with-context --task CT-001`` targets a single CT and emits one event."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()

    rc = harness_main([
        "--task", "CT-001", "--with-context", "--repo-root", str(tmp_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 eligible CT(s)" in out
    assert "context-mode ON" in out

    payloads = _harness_with_context_payloads(cfg.aegis_home)
    assert len(payloads) == 1
    assert payloads[0]["ct_id"] == "CT-001"


def test_with_context_unknown_task_returns_one(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--with-context`` does not bypass the eligibility check."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1")
    cfg = get_config()
    _seed_eligible(cfg.aegis_home)
    capsys.readouterr()

    rc = harness_main(["--task", "CT-999", "--with-context"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "CT-999" in err
    assert "not eligible" in err
    assert _harness_with_context_payloads(cfg.aegis_home) == []
