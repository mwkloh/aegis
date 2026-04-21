"""Plane 2 — Reflection (read-only).

Reads JSONL events written by the Runtime plane, clusters them into
deterministic patterns, and drafts human-reviewable proposals via the
Reflection model. **Never** executes tools, mutates canon, or talks to
the network except through the Reflection LLM client.
"""
