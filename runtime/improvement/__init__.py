"""Plane 3 — Improvement (human-gated governance).

Walks the proposals drafted by the Reflection plane, lets a human record
`approve / reject / defer` verdicts, and queues approved items as coding
tasks for the future Phase 4 coding harness. **Never** executes a
proposal, calls an LLM, or mutates canon. Deterministic, human-driven
only.
"""
