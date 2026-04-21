# AEGIS — End-to-End Improvement Walkthrough

This document demonstrates **one complete, concrete pass** through the AEGIS closed-loop improvement system, from runtime failure to merged improvement.

---

## Step 1 — Runtime Execution

A user issues the following request via Telegram:

> "Capture that decision we just made and add it to my notes."

The system selects the **update_personal_notes** skill and invokes the **search_obsidian** tool.

---

## Step 2 — Failure Occurs

The Obsidian search returns no results.

The system:
- Does NOT hallucinate context
- Does NOT silently fail
- Gracefully reports the issue to the user

---

## Step 3 — Event Recorded (EVENTS.md)

```json
{
  "timestamp": "2026-04-17T09:32:10+12:00",
  "event_type": "tool_failure",
  "skill": "update_personal_notes",
  "tool": "search_obsidian",
  "details": "No results returned for meeting note query",
  "context_hash": "sess_2026_04_17_001"
}
```

This entry is factual, append-only, and contains no reasoning.

---

## Step 4 — Pattern Detection (PATTERNS.md)

The reflection process later identifies repeated instances of this failure.

```json
{
  "pattern_id": "P-001",
  "description": "Obsidian search frequently fails when relevant information exists only in recent session memory",
  "related_events": ["evt_2026_04_17_001", "evt_2026_04_15_004"],
  "frequency": 4,
  "affected_components": ["skill:update_personal_notes", "memory:retrieval"],
  "severity": "low"
}
```

---

## Step 5 — Improvement Proposal (PROPOSALS.md)

```json
{
  "proposal_id": "IMP-001",
  "pattern_id": "P-001",
  "summary": "Fallback to recent session memory when Obsidian search returns no results",
  "proposed_change": "Modify retrieval logic to query recent session notes before declaring failure",
  "risk_level": "low",
  "requires_human_approval": true
}
```

---

## Step 6 — Human Approval (DECISIONS.md)

```json
{
  "proposal_id": "IMP-001",
  "decision": "approved",
  "rationale": "Low-risk reliability improvement that preserves skill intent",
  "timestamp": "2026-04-17T09:45:00+12:00"
}
```

---

## Step 7 — Coding Task Creation (CODING_TASKS.md)

The approved proposal is translated into **CT-001**, authorizing draft-only implementation.

---

## Step 8 — Coding Harness Execution

A coding agent is invoked with **CODING_PROMPT.md** and CT-001.

The agent:
- Reads only the approved scope
- Generates a unified diff
- Writes no files outside the allowed directory

---

## Step 9 — Human Review & Merge

A human reviews the diff, validates correctness, and manually merges the change.

No automated deployment occurs.

---

## Step 10 — Loop Closure

Subsequent executions show:
- Fewer tool failures
- Reduced EVENT frequency for this pattern
- Improved task completion rate

The system has improved **without autonomy creep**.

---

## Key Takeaway

AEGIS improves by discipline:

- Observation → Reflection → Proposal → Approval → Draft → Review

At no point does the system modify itself without explicit human intent.
