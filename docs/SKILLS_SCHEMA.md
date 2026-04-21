# AEGIS — Skills Registry Schema

Defines how skills are declared.

---

```yaml
skill_id: string
description: string
trigger_intents: []
allowed_tools: []
memory_read: []
constraints:
  max_steps: int
  risk_level: low|medium|high
```

---

Skills restrict reasoning scope and improve reliability for local models.
