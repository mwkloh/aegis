Robust Local‑First AI Assistant
Design Blueprint (v1)
1. Design Goals (Non‑Negotiable)
Local‑model first
Entire system must be usable with only local models (7B–13B class).
Frontier models are optional accelerators, not dependencies.
Structural reliability over raw intelligence
Tool correctness comes from architecture, not model capability.
LLMs never self‑report tool success.
Strict separation of concerns
Chat ≠ reasoning ≠ execution ≠ memory.
Model‑agnostic
Supports local (Ollama/llama.cpp) and frontier models via routing.
No architectural coupling to a specific vendor.
2. High‑Level System Architecture
┌───────────────────────────────────────────┐
│               Chat Interface               │
│          (Telegram / CLI / Web)            │
└───────────────┬───────────────────────────┘
                │ user input
                ▼
┌───────────────────────────────────────────┐
│          Intent Classification Agent       │
│   (small, cheap, short‑context model)      │
└───────────────┬───────────────────────────┘
                │ intent + confidence
                ▼
┌───────────────────────────────────────────┐
│             Skills Registry                │
│   (global, declarative, read‑only catalog)│
└───────────────┬───────────────────────────┘
                │ selected skill descriptor
                ▼
┌───────────────────────────────────────────┐
│          Skill‑Scoped Reasoning Agent      │
│ (local model OK; frontier optional)        │
│ - constrained context                      │
│ - allowed tools only                       │
└───────────────┬───────────────────────────┘
                │ tool‑intent contract
                ▼
┌───────────────────────────────────────────┐
│                OpenHarness                 │
│   (execution, retries, permissions)        │
└───────────────┬───────────────────────────┘
                │ progress + results
                ▼
┌───────────────────────────────────────────┐
│        Progress / Result Streaming         │
│            (back to chat UI)               │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│             Memory System                  │
│  (4‑tier semantic + episodic memory)       │
│  - embeddings, chunking, vector search     │
│  - Obsidian vault integration              │
└───────────────────────────────────────────┘
3. Core Principle
LLMs decide intent and planning.
Harnesses decide execution.
Memory systems decide persistence.
No component violates this.
4. Memory Architecture (External, Authoritative)
4.1 Memory Tiers
Tier
Purpose
Writable by Agent
Tier 1 – Preferences
Style, habits, defaults
❌ (proposal only)
Tier 2 – Identity
Long‑term facts about user
❌
Tier 3 – Episodic
Experiences, outcomes
❌
Tier 4 – External Knowledge
Obsidian vault, notes
❌
Execution Memory
Current task state
✅ (OpenHarness)
Rule:
LLMs can propose memory updates.
Only the memory layer commits.
4.2 Memory Access Pattern
Memory is never dumped wholesale
Retrieved per skill
Aggressively summarized
Token‑bounded
5. Skills System (Critical Layer)
5.1 What a Skill Is (Canonical Definition)
A skill is:
A constraint on reasoning
A selector for tools
A gatekeeper for memory access
A producer of a tool‑intent contract
A skill is not a tool and not a subagent.
5.2 Skills Registry (Global)
The skills registry is:
Declarative (YAML / JSON / Markdown)
Global
Read‑only at runtime
Not injected wholesale into prompts
Example Skill Descriptor
YAML
skill_id: update_personal_notes

description: >
Capture new information and append it to existing
personal notes without overwriting content.

trigger_intents:
- persist_information
- update_notes
- capture_insight

allowed_tools:
- search_obsidian
- summarize_text
- draft_note

memory_read:
- preferences
- episodic
- obsidian

memory_write_policy:
proposal_only: true
requires_review: true

constraints:
overwrite_existing: false
max_steps: 5
risk_level: low

progress_events:
- searching_notes
- summarizing
- drafting_update
Show more lines
6. Tiered Agent & Model Selection Strategy
This is how you make local models sufficient.
6.1 Agent Tiers
Tier
Role
Model Requirements
Tier 0
Intent Classification
Very small local model
Tier 1
Skill Reasoning
Local model preferred
Tier 2
Complex Planning (optional)
Frontier model optional
Tier 3
Execution
OpenHarness (no LLM)
6.2 Model Routing Logic
Plain Text
If task = intent classification:
use smallest local model

If task = skill‑scoped reasoning:
use local reasoning model

If task = long horizon / ambiguous:
optionally escalate to frontier model

Never:
require frontier model for correctness
Show more lines
Frontier models improve quality, not capability.
6.3 Why This Works for Local Models
Context is tightly bounded by skill
Tools are not reasoned about abstractly
Memory is externalized
Execution is deterministic
Failures are structural, not conversational
7. OpenHarness Integration
OpenHarness handles:
Tool execution
Permission enforcement
Retries & backoff
Parallel execution (if allowed)
Execution state
OpenHarness never:
Selects tools
Modifies memory
Decides intent
7.1 Tool‑Intent Contract (Interface)
The agent emits:
JSON
{
"skill": "update_personal_notes",
"steps": [
{
"tool": "search_obsidian",
"args": { "query": "meeting decision" }
},
{
"tool": "summarize_text",
"input_from": "search_obsidian"
},
{
"tool": "draft_note",
"args": { "mode": "append" }
}
]
}
Show more lines
OpenHarness executes only this.
8. Chat Interface Responsibilities (Telegram)
Telegram (or any chat UI):
✅ Handles:
Message ingestion
User identity
Progress rendering
Approval prompts
❌ Does not:
Execute tools
Query semantic memory directly
Manage retries
Infer skills
Telegram is just a transport + display surface.
9. Progress & Feedback Loop
Execution emits structured events:
JSON
{ "event": "searching_notes" }
{ "event": "summarizing" }
{ "event": "draft_ready", "requires_approval": true }
Show more lines
These are streamed back to Telegram.
This reduces:
User re‑prompting
Agent verbosity
Context drift
Token waste
10. Failure Handling (Local‑Model Safe)
Failures are states, not text.
Tool failure → structured error
Skill policy decides retry / fallback
Agent does not hallucinate recovery
User sees progress transparently
11. Minimal Prompting Strategy (Local‑Model Optimized)
No tool lists in prompts
No full skills registry in prompts
No long chat history
No chain‑of‑thought dumping
Use:
Structured outputs
Short, task‑specific prompts
Hard token caps
12. Implementation Order (Recommended)
Define memory API (read/propose/write)
Define skills registry schema
Implement intent classifier
Implement skill selector
Integrate OpenHarness
Add Telegram adapter
Add model router
Harden failure states
13. What This Blueprint Guarantees
✅ Works with only local models
✅ Frontier models optional
✅ High tool reliability
✅ Low hallucination rate
✅ Clear audit trail
✅ Extensible to new channels
✅ Scales without context explosion
14. How to Prompt a Coding Agent
You can literally say:
“Build the system described in this blueprint.
Follow the component boundaries strictly.
Do not collapse memory, skills, or execution layers.
Optimize for local models with short context windows.”
Final Note
You are designing a structural intelligence system, not just an agent.
This architecture is:
Current (2026‑grade)
Local‑model friendly
Production‑ready
Future‑proof
