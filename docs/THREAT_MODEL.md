# AEGIS — Threat Model (Prompt Injection & Agent Misuse)

## Scope
This document enumerates realistic threat classes affecting AEGIS, with emphasis on prompt injection, privilege escalation, and self-modifying behavior.

## Assets
- Execution authority (tools via OpenHarness)
- Canonical memory (USER.md, IDENTITY.md, SOUL.md)
- Skills registry
- Codebase integrity

## Trust Boundaries
- User input
- External content (web, notes, APIs)
- Reflection outputs
- Coding harness outputs

## Threat Classes (STRIDE mapping)
- **Spoofing**: Impersonation of system instructions
- **Tampering**: Attempts to alter memory or skills via prompts
- **Repudiation**: Ambiguous or unaudited changes
- **Information Disclosure**: Leakage via tool calls
- **Denial of Service**: Prompt flooding / context exhaustion
- **Elevation of Privilege**: Acquiring tool or code authority via language

## Architectural Mitigations
- Capability separation by plane
- Skills scoping of tools
- Proposal-only mutation
- Human-gated code changes

## Residual Risks
- Human approval error
- Poisoned trusted content

## Conclusion
AEGIS reduces prompt injection risk structurally by preventing language from conferring authority.
