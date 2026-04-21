AEGIS
Local‑First Guarded Cognitive System
Closed‑Loop Improvement Architecture (v1)
Invariant:
No existing file is renamed, repurposed, or semantically altered.
All new capability is additive and layered.
0. What This Blueprint Does (In One Sentence)
It extends your current Aegis system into a self‑reflective, human‑governed, locally runnable cognitive system with a separate improvement plane and coding harness, without allowing runtime self‑modification.
1. Canonical System Planes
Aegis explicitly operates across three planes:
PLANE 1 — RUNTIME (immutable during execution)
PLANE 2 — REFLECTION (read-only analysis)
PLANE 3 — IMPROVEMENT (human-gated change)
Each plane has different permissions.
2. Existing Files & Their Preserved Meaning
These remain authoritative and unchanged:
File
Canonical Role
AGENTS.md
Defines agent roles, boundaries, and authority
MEMORY.md
Execution & session memory mechanics
USER.md
User preferences
IDENTITY.md
Long-term identity facts
SOUL.md
Values, tone, personality invariants
HEARTBEAT.md
Periodic/background behaviors
Session notes
Raw conversational recall
All additions respect this canon.
3. Target Repo Structure (Additive Only)
Plain Text
aegis/
├── AGENTS.md
├── MEMORY.md
├── USER.md
├── IDENTITY.md
├── SOUL.md
├── HEARTBEAT.md
│
├── runtime/ # PLANE 1
│ ├── chat/
│ ├── intent/
│ ├── skills/
│ ├── harness/
│ └── model_router/
│
├── memory/
│ ├── semantic/
│ ├── episodic/
│ ├── indexing/
│ └── sessions/
│
├── improvement/ # PLANE 2 + 3 (NEW)
│ ├── EVENTS.md
│ ├── PATTERNS.md
│ ├── PROPOSALS.md
│ ├── DECISIONS.md
│ ├── CODING_TASKS.md
│ └── IMPROVEMENT.md
│
├── coding_harness/ # PLANE 3 (NEW)
│ ├── PROMPT.md
│ ├── CONTEXT.md
│ ├── OUTPUT_POLICY.md
│ └── diffs/
│
└── docs/
└── ARCHITECTURE.md
Show more lines
4. Runtime Plane (PLANE 1)
Responsibilities
Chat ingestion (Telegram, future channels)
Intent classification
Skill selection
Tool execution (OpenHarness)
Progress streaming
Prohibitions
No self‑modification
No semantic memory writes
No improvement reasoning
5. Skills Registry (Preserved + Extended)
Skills remain declarative scaffolds, not executable logic.
Location (recommended):
runtime/skills/
Each skill:
Restricts allowed tools
Declares memory access
Limits reasoning scope
Produces a tool‑intent contract
This structure is critical for local‑model reliability.
6. Model Routing (Local‑First)
Tiered Agent Model
Tier
Purpose
Preferred Model
Tier 0
Intent classification
Small local
Tier 1
Skill reasoning
Local reasoning
Tier 2
Complex planning (optional)
Frontier
Tier 3
Execution
OpenHarness
Frontier models are never required for correctness.
Model routing logic lives in:
runtime/model_router/
7. Closed‑Loop Improvement System
This is new, but strictly out‑of‑band.
7.1 Improvement Philosophy (IMPROVEMENT.md)
Defines:
Why failures are captured
What improvement means
What must never change automatically
This acts as a governance document.
7.2 Event Capture (EVENTS.md)
Captured during runtime, but without reasoning.
Events include:
Tool failures
Retries
User corrections
Skill aborts
Confusion signals
Append‑only. No summarization.
7.3 Pattern Detection (PATTERNS.md)
Generated offline or heartbeat‑driven.
Clusters repeated events into issues:
“Skill A frequently misses intent X”
“Tool B often fails after step Y”
Local models can do this using summaries.
7.4 Proposal Generation (PROPOSALS.md)
Each proposal is:
Scoped
Non‑executing
Human‑reviewable
Example proposal fields:
Observed pattern
Affected skill/tool/memory
Proposed change
Risk level
7.5 Human Decisions (DECISIONS.md)
Records:
Approved
Rejected
Deferred
This ensures auditability and prevents drift.
8. Coding Harness (PLANE 3)
Purpose
Translate approved proposals into code diffs, nothing more.
8.1 Coding Harness Rules
The coding harness: ✅ Reads repo context
✅ Produces diffs
✅ Writes to /coding_harness/diffs/
✅ Generates tests and notes
It cannot: ❌ Commit
❌ Deploy
❌ Modify runtime during execution
8.2 Coding Agent Canonical Prompt (PROMPT.md)
This is what you give the coding agent:
You are a coding assistant operating in the AEGIS Improvement Plane.

Constraints:
- You may only implement tasks explicitly listed in CODING_TASKS.md.
- You must not modify runtime behavior beyond the approved scope.
- You must not change AGENTS.md, USER.md, IDENTITY.md, SOUL.md.
- You must output changes as unified diffs only.
- You must include rollback notes and tests if applicable.

Objective:
Produce safe, minimal, reviewable code improvements.

Output format:
- Summary
- Unified diff
- Test notes
- Rollback instructions

This prompt prevents dangerous autonomy.
9. Implementation Plan (Phased, Safe)
Phase 0 — Documentation & Boundaries
Finalize ARCHITECTURE.md
Write IMPROVEMENT.md ✅ No behavior change
Phase 1 — Instrumentation Only
Add event logging
Populate EVENTS.md ✅ Observe, don’t act
Phase 2 — Reflection (Read‑Only)
Pattern clustering
Proposal drafting ✅ No execution effects
Phase 3 — Human Approval Loop
Telegram / CLI approval
Decision logging ✅ Governance established
Phase 4 — Coding Harness (Draft Only)
Generate diffs
No merges ✅ Human still in control
Phase 5 — Optional Acceleration
Frontier models for coding only
More automated suggestions ✅ Runtime remains local‑safe
10. Why This Design Is Stable
Runtime cannot mutate itself
Improvement is mediated and delayed
Frontier models are sandboxed
Local models remain sufficient
Memory integrity is preserved
This avoids every known failure mode of self‑modifying agents.
