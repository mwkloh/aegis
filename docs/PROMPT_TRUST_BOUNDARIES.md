# AEGIS — Prompt Trust Boundaries

## Principle
Untrusted text must never be treated as executable instruction.

## Boundaries
- User input is always DATA
- External content is always DATA
- System prompts are static and file-based
- No mixed instruction hierarchies

## Allowed Flows
- DATA -> intent classification
- DATA -> reasoning (constrained)

## Forbidden Flows
- DATA -> tool permissions
- DATA -> memory writes
- DATA -> code changes

## Implementation Rules
- Never concatenate system prompts with fetched content
- Always label external text as inert data

## Rationale
This defeats classic "ignore previous instructions" attacks.
