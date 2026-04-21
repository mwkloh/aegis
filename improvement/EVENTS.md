# AEGIS — Improvement Events

Append-only log of execution signals.

---

## Event Schema

```json
{
  "timestamp": "ISO8601",
  "event_type": "tool_failure | retry | user_correction | skill_abort",
  "skill": "string",
  "tool": "string | null",
  "details": "string",
  "context_hash": "string"
}
```

---

## Rules

- Events are factual only.
- No reasoning or summarization.
- Never edited or deleted.
