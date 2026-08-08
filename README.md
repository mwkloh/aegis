# AEGIS

> **Local-first, guarded cognitive system.** A personal AI assistant whose tool execution is correct *by architecture*, not by model capability — built to run usefully on small open-weights models (2B–13B; tested down to Gemma4:e2B) with frontier APIs as optional accelerators.

> ⚠️ **Status: experimental, single-operator R&D.** This repo is one person's running notebook for an idea — that the reliability of AI assistants comes from structure (gates, planes, contracts) more than from raw model intelligence. It is **not** a packaged product. APIs change without notice, the test suite assumes a specific operator workflow, and the canonical state lives outside the repo at `~/.aegis/`. Read it as a design exploration, not a drop-in tool.

---

## Why AEGIS exists

Most "agentic" wrappers around an LLM trust the model to report on its own tool use. When the model is GPT-5 / Claude Sonnet that mostly works. When it's a 4B local model running on a laptop, it doesn't — the model will happily reply *"I've deleted the duplicate files for you 🦊"* without ever emitting a tool call.

AEGIS treats every LLM reply as a **claim**, and uses a separate, deterministic layer to decide whether the claim is actually backed by an executed tool call. The layered defenses are the point of the project:

| Layer | What it stops |
|---|---|
| **Skills registry** | LLM cannot reach for tools it wasn't granted on this turn |
| **Multi-step loop with bounded steps** | LLM cannot loop forever, cannot escalate scope mid-chain |
| **Destructive guard** | Mutating tools (`files_delete`, `files_move`, `files_write`) at step 2+ are intercepted with a deterministic confirmation prompt — *no LLM rephrasing* |
| **Verdict gate** | Regex detects first-person action claims ("I deleted X", "I created Y") and prepends `⚠️ unverified tool claim — not executed` if the implied tool didn't actually run |
| **Tool-result audit trail** | Every executed tool emits a `tool.invoked` record; the verdict gate reads from this, not from the LLM's self-report |

The design principle, lifted from `AEGIS_BLUEPRINT.md`:

> LLMs decide **intent and planning**. Harnesses decide **execution**. Memory systems decide **persistence**. No component violates this.

---

## High-level architecture

```
┌──────────────────────────────────────────────┐
│              Chat Interface                  │
│             (Telegram / CLI)                 │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│        Intent Classifier (Tier 0)            │
│        small local model, cheap              │
└────────────────────┬─────────────────────────┘
                     │ {intent, confidence}
                     ▼
┌──────────────────────────────────────────────┐
│           Skills Registry                    │
│   declarative YAML, read-only at runtime     │
└────────────────────┬─────────────────────────┘
                     │ selected skill descriptor
                     ▼
┌──────────────────────────────────────────────┐
│   Multi-Step Planner + Skill-Scoped Reasoner │
│   • bounded step count (max 5 by default)    │
│   • allowed-tools list per skill             │
│   • destructive-guard interception           │
└────────────────────┬─────────────────────────┘
                     │ tool-intent contracts
                     ▼
┌──────────────────────────────────────────────┐
│              Harness Adapter                 │
│   the only path to side effects              │
│   (tool registry: files_*, time, echo, …)    │
└────────────────────┬─────────────────────────┘
                     │ tool results + audit trail
                     ▼
┌──────────────────────────────────────────────┐
│   Reply Synthesizer  →  Verdict Gate         │
│   model writes prose → gate annotates if     │
│   the prose claims actions that didn't run   │
└──────────────────────────────────────────────┘
```

The codebase is split into **three planes**, enforced by import-graph tests:

- **Runtime (`runtime/`)** — classify, plan, execute tools, render replies. May not modify its own source or canonical memory files.
- **Reflection (`runtime/reflection/`)** — read-only sweep over today's events (`reflect` make target).
- **Improvement (`improvement/`, `coding_harness/`)** — drafts code patches as `.patch.md` files for human review (`harness` / `apply` make targets). Plane 1 may not import from this plane.

---

## Key features

### Bounded multi-step tool loop
The planner can chain tools (e.g. `files_search` → `files_read` → respond) but is hard-capped at `cfg.harness.max_steps` (default 5). Termination conditions: planner returns `kind=respond`, step cap hit, registry empty, or destructive-guard interception. See `runtime/chat/telegram/harness_dispatcher.py:_run_multi_step`.

### Destructive guard + confirmation flow
Mutating tools are allowed at step 1 (the operator's explicit opening request) but intercepted at step 2+ with a deterministic message that surfaces the exact tool id and args — no LLM paraphrase between the model and the operator. Implemented as a `frozenset` lookup, not a prompt instruction, so it can't be jailbroken by clever phrasing. On interception the harness holds the pending intent and executes it only on an **exact-match** affirmative follow-up ("yes", "confirm", "go ahead") within a 120s TTL — no model in the confirmation loop, and the intent is armed only *after* the prompt is delivered, so a failed send can't leave an invisible pending action. See `runtime/chat/telegram/harness_dispatcher.py`.

### Evidence ledger
Every tool the harness executes leaves a structural proof record — skill, tool, an argv hash, a byte count, and a classified verdict (`verified` / `exit_nonzero` / `tool_error` / …) — appended to the session event shard, scoped to the turn. No argv contents or output bodies are stored. This is the substrate the completion gate reads. See `runtime/tools/record.py`.

### Completion gate
When the planner declares it's done (`kind=task_complete`), the harness checks the claimed summary against the turn's *verified* ledger evidence rather than trusting the model. A tool that failed (including a soft failure like a non-zero `run_command` exit) never counts as verified, and an unrecovered failure appends an honest "⚠️ this did not complete successfully" note to the reply. Annotate-not-block for now — the `harness.completion_gated` events measure how often a hard block would fire before one is turned on. This is the direct counter to a small model lying about completion.

### Schema-constrained decoding
For local models via Ollama, tool-call and plan JSON is constrained in the decoder itself (a JSON schema passed as Ollama's `format`, compiled to a GBNF grammar) so malformed output is impossible rather than merely discouraged — the single biggest reliability lever for 2B-class models. A deterministic `repair_json()` pass salvages wrapper noise (markdown fences, prose, trailing commas) before spending a corrective retry. Client-side schema validation stays the trust boundary; the decoder constraint is an optimization that can never weaken it.

### Guarded file writes and command runner
`files_write` writes text inside the configured sandbox roots (atomic tmp-then-rename, 256 KiB cap, symlink-escape rejected) and rides the confirmation flow above. `run_command` runs a read-only inspection command as an **argv list** — no shell, ever — against an operator-defined binary allowlist (`ls`, `cat`, `grep`, … by default; `find` deliberately excluded); path arguments are validated against the same sandbox roots as `files_read`, so it can't read the bot's own secrets. Both are opt-in and off by default.

### Reply verdict gate
Pure-function regex check against ~15 first-person claim patterns ("I've deleted", "I ran", "I created"), suppressed by negation/offer patterns ("I would", "I can"). Tool-pinned: "I deleted X" only fires the gate if `files_delete` is *not* in the verified-tools set for this turn. Fully unit-tested; conservative by design (prefers false negatives over annotation noise).

### Skills system
Each skill is a YAML descriptor with `trigger_intents`, `allowed_tools`, and an args schema. The reasoner sees only the tools its skill grants — broad capabilities don't leak into narrow tasks. Skills live in `~/.aegis/workspace/skills/`, seeded from `runtime/skills/_bundle/` on first boot.

### Local-model first
The default model stack assumes Ollama or LM Studio is running locally. OpenRouter / Anthropic / OpenAI are wired in but are accelerators, not requirements. See `runtime/llm/router.py` for tier routing.

### Memory tiering
Four-tier memory (preferences / identity / episodic / external knowledge) plus per-task execution memory. Memory is retrieved per-skill, summarized, never wholesale-injected. SQLite + sqlite-vec for embeddings. Optional Obsidian vault integration.

---

## Installation

### Prerequisites

- **Python 3.11+** (project tested on 3.11–3.13)
- **Ollama** running locally (`brew install ollama && ollama serve`) — required for the harness dispatcher to enable
- A Telegram bot token if you want the chat interface (`@BotFather`)
- Optional: OpenRouter / Anthropic / OpenAI keys for frontier-tier routing

### First-time setup

```bash
git clone https://github.com/<you>/aegis.git
cd aegis

# Create venv, install runtime + dev deps, provision ~/.aegis/, run doctor
make setup

# One-shot health check (re-runnable)
make doctor
```

`make setup` runs three steps you can also invoke individually:

| Step | What it does |
|---|---|
| `make install` | Creates `.venv`, installs runtime + dev dependencies via pip |
| `make bootstrap` | Idempotently provisions `~/.aegis/workspace/` (memory, sessions, skills, canonical .md files), copies a starter `config.json` and `.env` if absent |
| `make doctor` | Verifies Ollama is reachable, required models are pulled, `~/.aegis/` layout is correct, optional API keys are valid |

### Configure secrets and models

Edit `~/.aegis/.env` (created by bootstrap):

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOW_FROM=<your-numeric-tg-user-id>
OPENROUTER_API_KEY=...        # optional
ANTHROPIC_API_KEY=...         # optional
OPENAI_API_KEY=...            # optional
```

Edit `~/.aegis/config.json` to pick model defaults (see the `models:` and `modelAliases:` sections). To enable the multi-step tool loop:

```json
{
  "harness": {
    "multi_step": true,
    "max_steps": 5
  }
}
```

The flag can also be set per-process via `HARNESS_MULTI_STEP=1`.

### Run

```bash
# CLI walking skeleton — no Telegram needed
make run

# Production Telegram bot (requires TELEGRAM_BOT_TOKEN + Ollama up)
.venv/bin/python -m runtime.serve

# Recommended: run inside tmux to survive macOS TCC prompts
tmux new -d -s aegis '.venv/bin/python -m runtime.serve 2>&1 | tee -a ~/.aegis/logs/aegis-tmux.log'
```

### Test

```bash
make test           # full gate: ruff + mypy + unit + e2e
make test-unit      # fast loop
make security       # bandit + semgrep
```

---

## Maintaining and releasing

The repeatable change → verify → PR → release loop is documented in
[`docs/MAINTAINING.md`](docs/MAINTAINING.md); notable changes per version are in
[`CHANGELOG.md`](CHANGELOG.md). The short version: branch, change it test-first,
`make test` must be green, PR it, and tag a release from a clean `main`.

---

## Repository layout

```
runtime/             Plane 1 — chat, intent, skills, reasoning, harness, llm
  chat/              CLI + Telegram surfaces; reply_verdict.py lives here
  intent/            Tier 0 classifier (small local model)
  skills/            Skill registry + bundled descriptors
  reasoning/         Tier 1 skill-scoped planner; prompts/ holds the chain templates
  harness/           Tool registry + execution adapter (the only path to side effects)
  llm/               Ollama / OpenRouter / OpenAI / Anthropic clients + tier router
  reflection/        Read-only event sweep (Plane 2)
  improvement/       Patch-draft pipeline (Plane 3, draft-only)
  scheduler/         Cron-style job runner (UTC internally)
  events/            Append-only JSONL audit trail (~/.aegis/workspace/sessions/)
coding_harness/      Plane 3 patch-application harness
memory/              SQLite + sqlite-vec memory store
scripts/             aegis-doctor, aegis-bootstrap, aegis-telegram-smoke, …
docs/                ADRs and design plans
deploy/              launchd plists for macOS supervised runs
tests/               1350+ unit/integration/e2e tests
AEGIS_BLUEPRINT.md   Original design document
```

Canonical *operator* state lives outside the repo:

- `~/.aegis/workspace/` — agent workspace (memory, sessions, identity, skills)
- `~/.aegis/.env` — secrets
- `~/.aegis/config.json` — runtime config
- `~/.aegis/logs/` — log files

---

## Caveats and limitations

- **Experimental and unstable.** Schemas, tool ids, gate patterns, and configuration shape change as the design is iterated. There is no semver contract.
- **Single-operator assumptions.** Authentication is a numeric Telegram user-id allowlist. There is no multi-tenant isolation, no RBAC, no audit-export tooling for compliance contexts. Do not point this at strangers.
- **macOS-shaped.** The deploy story (`launchd`, tmux, TCC), the file-tool sandbox, and the doctor checks assume macOS. Linux probably works for the runtime but is untested.
- **The gates are conservative, not infallible.** The verdict-gate regex set is curated to lean toward false negatives (missed flags) over false positives (annotation noise). The destructive guard only catches the listed tool ids — adding a new mutating tool requires adding it to `DESTRUCTIVE_TOOLS` *and* writing a unit test. The completion gate annotates rather than blocks (see open questions in `docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md`). Read `runtime/chat/reply_verdict.py` and `runtime/chat/telegram/harness_dispatcher.py` before trusting any layer in a new context.
- **The confirmation flow and multi-step tools require `harness.multi_step`.** The destructive guard, the confirmation flow, and chained tool use only run when the multi-step loop is enabled (`harness.multi_step=true` or `HARNESS_MULTI_STEP=1`). In the default single-shot mode a directly-requested `files_write` executes as the operator's stated step-1 intent without confirmation.
- **`run_command` containment is an allowlist plus path-sandboxing, not a jail.** It refuses non-allowlisted binaries and validates absolute/`~` path arguments against the sandbox roots, but it is not a full sandbox: a bare relative token with no separator resolves against the process working directory. Extend `commands.allowed_binaries` deliberately — adding `find`, `rm`, or an interpreter re-opens execution vectors the defaults exclude.
- **Local-model dependency.** With Ollama down, the harness dispatcher logs `harness_dispatcher.disabled` and the bot falls back to plain chat. Multi-step tool use requires a working local classifier.
- **Solo project, solo issue tracker.** Issues live as markdown files under `.scratch/<feature>/` in this repo, not on GitHub Issues. Triage labels and conventions are documented in `docs/agents/`.

## License

[MIT](LICENSE) — see the `LICENSE` file. Do whatever you want with it; just keep the copyright and disclaimer.

## Acknowledgements

The design is heavily influenced by Anthropic's structured-tool-use patterns and by the broader "small model + good scaffolding > big model + thin wrapper" line of thinking. The blueprint document (`AEGIS_BLUEPRINT.md`) preserves the original v1 reasoning.
