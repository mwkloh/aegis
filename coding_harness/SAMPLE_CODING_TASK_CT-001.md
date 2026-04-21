# AEGIS — Sample Coding Task (CT-001)

This document represents a **realistic, approved coding task** produced by the AEGIS closed-loop improvement system.

---

## Task Metadata

```json
{
  "task_id": "CT-001",
  "proposal_id": "IMP-001",
  "risk_level": "low",
  "approval_required": true
}
```

---

## Problem Statement

When executing the **update_personal_notes** skill, searches against the Obsidian knowledge base sometimes return no results even though the relevant information exists in **recent session memory**.

This leads to unnecessary task failure or follow-up clarification.

---

## Approved Scope

The coding task is strictly limited to the following areas:

```json
[
  "runtime/skills/update_personal_notes",
  "memory/semantic/retrieval"
]
```

---

## Constraints (Non-Negotiable)

- Do NOT modify AGENTS.md
- Do NOT modify USER.md, IDENTITY.md, or SOUL.md
- Do NOT change skill intent or expand authority
- Do NOT alter OpenHarness execution semantics
- Produce diffs only (no commits, no deployment)

---

## Task Description

Add a **fallback retrieval mechanism**:

1. Attempt Obsidian search as the primary retrieval mechanism
2. If no results are returned:
   - Query recent session memory (last N sessions)
   - Use those results as context if available
3. Only declare failure if both strategies fail

---

## Expected Output

The coding harness must produce:

- ✅ Unified diff implementing fallback logic
- ✅ Minimal validation notes or test stub (if applicable)
- ✅ Rollback instructions explaining how to revert the change

---

## Notes

This task:
- Is additive and reversible
- Does not increase model authority
- Improves reliability without increasing context window usage
- Is safe for local-model-first execution
