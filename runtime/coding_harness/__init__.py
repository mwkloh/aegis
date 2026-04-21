"""Plane 3 — Coding Harness (draft only).

Reads approved `CT-NNN` rows from
`<workspace>/improvement/CODING_TASKS.md`, asks a coding model for a
unified diff + summary + tests + rollback, and writes one
`.patch.md` draft per CT under `<workspace>/coding_harness/diffs/`.

**Never** applies a diff, commits, runs tests, or mutates canon.
A human merges manually. Failure modes degrade to a structural stub
so the workflow is always replayable.
"""
