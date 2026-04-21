# AEGIS — Model Routing Policy

Rules for selecting models per task.

---

## Policy

- Intent classification → smallest local model
- Skill reasoning → local reasoning model
- Complex planning → optional frontier model
- Execution → no LLM (harness only)

---

## Constraint

Frontier models must never be required for correctness.
