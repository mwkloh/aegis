# `coding_harness/` — Plane 3 (Coding Harness)

> The coding harness translates approved proposals into **draft** code diffs. It MAY read repo context, produce diffs into `coding_harness/diffs/`, and write tests/notes. It MUST NOT commit, deploy, or mutate runtime during execution.

See:

- [`CODING_PROMPT.md`](CODING_PROMPT.md) — canonical prompt for the coding agent
- [`SAMPLE_CODING_TASK_CT-001.md`](SAMPLE_CODING_TASK_CT-001.md) — example task format
- [`../docs/CLOSE_LOOP_IMPROVEMENT_ARCHITECTURE.md`](../docs/CLOSE_LOOP_IMPROVEMENT_ARCHITECTURE.md) §8

## Output policy

- Diffs land in `coding_harness/diffs/` (gitignored — drafts only).
- Each diff ships with: summary, unified diff, test notes, rollback instructions.
- A human merges manually, never the harness.
