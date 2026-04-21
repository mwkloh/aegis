# AEGIS — Prompt Provenance Checker

## Purpose
Ensure that every prompt passed to an LLM has traceable origin and authority.

## Prompt Metadata
Each prompt must declare:
- source (system | skill | reflection | coding)
- plane (runtime | reflection | improvement)
- authority level (none | propose | execute)

## Validation Rules
- Runtime prompts must have authority = none
- Reflection prompts must have authority = propose
- Coding prompts must have authority = draft

## Enforcement
- Reject prompt execution if provenance is missing or invalid

## Outcome
Prevents privilege escalation through prompt composition.
