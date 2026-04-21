# AEGIS — System Architecture

**Version:** v1  
**Status:** Canonical  
**Scope:** Runtime, Memory, Skills, Improvement, and Coding Harness  
**File Preservation:** Strict (no renaming or semantic alteration of existing files)
**OS Agnostic:** Should be able to install on Linux/MacOS/Windows (WSL2)
**Installed Environment:** Separate default "~/.aegis/" folder that stores data, skills, logs, configurable data and system required data, from application codebase folder.

---

## 1. Purpose

AEGIS is a **local-first, guarded cognitive system** designed to:

- Preserve long-term personal context
- Execute real-world actions reliably
- Operate effectively with **short-context local models**
- Improve itself over time **without autonomous self-modification**
- Remain understandable, auditable, and human-governed

This document defines the **authoritative system architecture**, the separation of responsibilities, and the trust boundaries between subsystems.

---

## 2. Core Architectural Principles

AEGIS is built on the following non-negotiable principles:

1. **Separation of Concerns**
   - Conversation, reasoning, execution, memory, and improvement are distinct layers.

2. **Local-First Sufficiency**
   - The system must function correctly using only local models.
   - Frontier models are optional accelerators, not dependencies.

3. **Harnessed Execution**
   - Language models do not self-report success.
   - All real actions are executed and verified by a harness layer.

4. **Governed Self-Improvement**
   - The system may observe failures and propose changes.
   - No component may modify runtime behavior without human approval.

5. **Strict File Canon**
   - Existing canonical files (`AGENTS.md`, `MEMORY.md`, etc.) retain original meaning.
   - New capabilities are strictly additive.

---

## 3. Architectural Planes

AEGIS operates across **three planes**, each with different permissions.

### 3.1 Plane 1 — Runtime (Production)

**Characteristics**
- Executes user-facing tasks
- Immutable during execution
- Must be deterministic and safe

**Includes**
- Chat interfaces (e.g. Telegram)
- Intent classification
- Skill selection and reasoning
- Tool execution via harness
- Progress and result streaming

**Must NOT**
- Modify its own code
- Modify skills or memory schemas
- Apply improvements

---

### 3.2 Plane 2 — Reflection (Read-Only Analysis)

**Characteristics**
- Observes and analyzes runtime behavior
- Never executes real-world actions
- Runs asynchronously or on a schedule

**Includes**
- Failure and friction signal aggregation
- Pattern detection
- Improvement proposal drafting

**Must NOT**
- Execute tools
- Modify runtime logic
- Write directly to canonical memory or skills

---

### 3.3 Plane 3 — Improvement (Human-Governed)

**Characteristics**
- Translates approved proposals into changes
- Human-in-the-loop required
- Changes are reviewable and reversible

**Includes**
- Coding harness (frontier model allowed)
- Draft diffs and migration notes
- Explicit human approval and merge

---

## 4. Canonical Files and Their Authority

The following files are **authoritative and preserved**:

| File | Canonical Responsibility |
|----|--------------------------|
| `AGENTS.md` | Agent roles, authority, and boundaries |
| `MEMORY.md` | Execution and session memory rules |
| `USER.md` | Preferences and personal defaults |
| `IDENTITY.md` | Long-term identity facts |
| `SOUL.md` | Values, tone, and persona invariants |
| `HEARTBEAT.md` | Periodic/background behavior definitions |
| Session notes | Raw conversational recall |

No automation may overwrite or repurpose these files.

---

## 5. High-Level System Flow

```
User Input
   ↓
Chat Interface (Telegram, CLI, etc.)
   ↓
Intent Classification Agent
   ↓
Skills Registry Lookup
   ↓
Skill-Scoped Reasoning Agent
   ↓
Tool-Intent Contract
   ↓
Execution Harness (OpenHarness)
   ↓
Results & Progress Streamed Back
```

Memory is accessed **orthogonally** at controlled points.

---

## 6. Skills Architecture

### 6.1 Definition

A **skill** is a declarative orchestration scaffold that:

- Narrows reasoning scope
- Constrains allowed tools
- Declares memory access
- Produces a tool-intent contract

Skills are **not tools**, **not agents**, and **not code**.

---

### 6.2 Skill Responsibilities

A skill may:
- Select from a constrained toolset
- Enforce execution limits
- Declare risk and approval requirements

A skill may not:
- Execute tools directly
- Modify memory
- Persist changes autonomously

---

## 7. Execution Harness

AEGIS relies on an execution harness (e.g. OpenHarness) to:

- Execute tools deterministically
- Enforce permissions
- Handle retries and failures
- Emit structured execution events

The harness is the **only component allowed to touch the external world**.

Language models never assume execution success.

---

## 8. Memory Architecture

AEGIS uses a **tiered memory system**, external to language model context.

### 8.1 Memory Tiers

| Tier | Purpose | Writable by Agent |
|----|--------|------------------|
| Preferences | Habits, defaults | No (proposal only) |
| Identity | Long-term facts | No |
| Episodic | Experiences and outcomes | No |
| External Knowledge | Obsidian vault, notes | No |
| Execution Memory | Task state | Yes (harness only) |

All long-term memory writes require governance.

---

## 9. Closed-Loop Improvement System

### 9.1 Overview

AEGIS improves by **observing**, **reflecting**, and **proposing**, never by mutating itself at runtime.

Improvement follows this lifecycle:

```
Execution Signals
   → Pattern Detection
       → Improvement Proposal
           → Human Review
               → Coding Harness (draft only)
                   → Human Merge
```

---

### 9.2 Improvement Artifacts

The improvement plane introduces dedicated artifacts:

- `EVENTS.md` — raw failure and friction signals
- `PATTERNS.md` — clustered recurring issues
- `PROPOSALS.md` — candidate improvements
- `DECISIONS.md` — approved / rejected decisions
- `CODING_TASKS.md` — approved implementation tasks

These files are **never injected into runtime prompts**.

---

## 10. Coding Harness

### 10.1 Purpose

The coding harness translates **approved improvement proposals** into **reviewable code changes**.

### 10.2 Constraints

The coding harness:

✅ May generate diffs  
✅ May generate tests and notes  
✅ May use frontier models  

❌ May not commit  
❌ May not deploy  
❌ May not run during execution  
❌ May not bypass human approval  

It behaves as a **draft-only engineering assistant**.

---

## 11. Model Strategy

AEGIS uses **tiered agents and model routing**:

| Agent Tier | Role | Model Preference |
|----------|-----|------------------|
| Tier 0 | Intent classification | Small local |
| Tier 1 | Skill reasoning | Local reasoning |
| Tier 2 | Complex planning | Frontier optional |
| Tier 3 | Execution | Harness (no LLM) |

Local models must be sufficient for correctness.

---

## 12. Safety and Stability Guarantees

This architecture guarantees:

- No autonomous self-modification
- No silent behavioral drift
- No memory corruption
- No hallucinated execution
- Clear audit trails
- Human sovereignty over change

---

## 13. Architectural Summary

AEGIS is:

- **Guarded**, not autonomous
- **Structured**, not prompt-driven
- **Local-first**, not cloud-dependent
- **Improving**, not self-rewriting
- **Explainable**, not opaque


---

## 14. Trust Boundaries

AEGIS enforces explicit **trust boundaries** to prevent prompt injection, privilege escalation, and unintended authority transfer.

### 14.1 Trust Levels

All inputs and artifacts are classified into one of the following trust levels:

- **Untrusted Data**
  - User messages
  - External content (web pages, notes, emails, APIs)

- **Constrained Instructions**
  - Skill prompts
  - Reflection prompts
  - Coding harness prompts

- **Authoritative Configuration**
  - Canonical markdown files (AGENTS.md, MEMORY.md, USER.md, IDENTITY.md, SOUL.md)
  - Architecture and governance documents

Untrusted data is NEVER treated as executable instruction.

---

### 14.2 Explicit Boundaries

The following flows are **permitted**:

- Untrusted data → intent classification
- Untrusted data → constrained reasoning within a skill
- Tool results → inert context for summarization or analysis

The following flows are **forbidden**:

- Untrusted data → tool permission changes
- Untrusted data → memory mutation
- Untrusted data → code or configuration changes
- External content → system prompt escalation

---

### 14.3 Plane Isolation

Trust boundaries align with architectural planes:

- **Runtime Plane**
  - Processes untrusted data
  - Holds no mutation authority

- **Reflection Plane**
  - Processes historical evidence only
  - May propose changes, but cannot enact them

- **Improvement Plane**
  - Receives only human-approved inputs
  - Produces draft-only artifacts

No plane may assume the authority of another.

---

### 14.4 Design Guarantees

By enforcing trust boundaries structurally, AEGIS guarantees:

- Prompt injection cannot escalate privileges
- Instructions cannot be overridden by user-provided text
- Memory and identity cannot be corrupted by language alone
- All lasting changes require explicit human intent

These guarantees are enforced by architecture, not by prompt discipline.


This document is the **canonical architectural reference**.
