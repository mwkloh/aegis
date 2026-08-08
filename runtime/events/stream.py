"""Append-only JSONL event stream.

Sessions land under `<sessions_dir>/YYYY-MM-DD/<session_id>.jsonl`.
Every entry is a JSON object with `ts`, `session_id`, `type`, `payload`.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    SESSION_START = "session.start"
    USER_MESSAGE = "user.message"
    INTENT_CLASSIFIED = "intent.classified"
    SKILL_SELECTED = "skill.selected"
    CONTRACT_EMITTED = "contract.emitted"
    TOOL_INVOKED = "tool.invoked"
    TOOL_RESULT = "tool.result"
    ASSISTANT_REPLY = "assistant.reply"
    SESSION_END = "session.end"
    ERROR = "error"
    # Phase 1 — model-call instrumentation.
    MODEL_CALL_START = "model.call.start"
    MODEL_CALL_END = "model.call.end"
    # Structured signals for the Reflection plane to cluster later.
    PATTERN_OBSERVED = "pattern.observed"
    # Phase 3 — Improvement plane governance audit trail.
    GOVERNANCE_DECISION = "governance.decision"
    # Phase 4 — Coding harness draft lifecycle.
    HARNESS_DRAFT_START = "harness.draft.start"
    HARNESS_DRAFT_END = "harness.draft.end"
    HARNESS_REFUSED = "harness.refused"
    # Phase 5 Track B — critique-then-revise lifecycle.
    HARNESS_CRITIQUE_START = "harness.critique.start"
    HARNESS_CRITIQUE_END = "harness.critique.end"
    HARNESS_REVISE_START = "harness.revise.start"
    HARNESS_REVISE_END = "harness.revise.end"
    # Phase 7 — Telegram chat turn telemetry (structural counts only, no bodies).
    CHAT_TURN_CONTEXT = "chat.turn.context"
    CHAT_TURN_REPLY = "chat.turn.reply"
    # Phase 8 Track B — structured LLM output reliability.
    LLM_STRUCTURED_RETRY = "llm.structured_retry"
    LLM_TIER_ESCALATED = "llm.tier_escalated"
    LLM_STRUCTURED_FAILED = "llm.structured_failed"
    # Phase 11 Track A — deterministic JSON repair salvage.
    LLM_JSON_REPAIRED = "llm.json_repaired"
    # Phase 10 — Autonomous scheduler lifecycle.
    SCHEDULER_TICK = "scheduler.tick"
    SCHEDULER_JOB_STARTED = "scheduler.job_started"
    SCHEDULER_JOB_SUCCEEDED = "scheduler.job_succeeded"
    SCHEDULER_JOB_FAILED = "scheduler.job_failed"
    SCHEDULER_SKIPPED_BUSY = "scheduler.skipped_busy"
    SCHEDULER_SKIPPED_STALE = "scheduler.skipped_stale"
    # Phase 11 Track C — task_complete completion gate.
    HARNESS_COMPLETION_GATED = "harness.completion_gated"


class EventStream:
    """Append-only writer. Single open() per append — Phase 0 is not throughput-critical."""

    def __init__(self, sessions_dir: Path, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        date = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        self.dir = Path(sessions_dir) / date
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.session_id}.jsonl"

    def append(self, event_type: EventType, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "session_id": self.session_id,
            "type": event_type.value,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + os.linesep)
