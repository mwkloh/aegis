# Phase 8 — Web UI Operator Console (Tailscale-gated)

> Status: **Stub drafted 2026-04-18**. Full plan deferred until
> Phase 7 Telegram bot lands and surfaces real operator pain points
> that justify a third interface. This document captures the
> **load-bearing design principles** so the eventual build cannot
> drift away from them.

## 1. Goal

A single browser surface — bound to the Tailscale interface only,
no public exposure — that gives the operator a 1-stop-shop for
everything AEGIS does:

- Tuning knobs (`AUTO_RECALL_CONFIDENCE`, byte budgets, model tier
  defaults, vault source priorities).
- Event timeline browser (Plane 1 events grouped by session/imp_id).
- Session detail viewer with cold-storage replay (verbatim raw
  turns rendered inline; same `cold_storage.py` slice reader the bot
  uses).
- Pattern explorer (Plane 2 `PatternRecord`s with drill-down to the
  events that produced them).
- Proposal review with diff preview (read-only render of the
  patches the harness drafted).
- Apply-verdict history (Plane 3 outcomes with `applied_clean` /
  `applied_test_failed` / `apply_conflict` / `reverted` filters).
- Vault source management (list/add/remove sources from §5.1,
  trigger reindex).

Telegram complements this for **mobile + governance moments**.
CLI complements this for **scriptable + power-user workflows**.
The web UI is the **default operator console for everything else**.

## 2. Non-negotiables (the load-bearing principles)

### 2.1 No reinvention of any plane

**The web UI is a thin renderer over the same APIs the CLI and
Telegram bot already call.** Concretely:

| Surface action       | API the UI must call                            |
|:---------------------|:------------------------------------------------|
| Approve a proposal   | `governance.record_decision(...)` — same path   |
| Run apply            | `coding_harness.apply_cli.run(...)` — same path |
| Trigger reflection   | `reflection.detect_all(events)` — same path     |
| Recall verbatim      | `chat.memory.cold_storage.read_slice(...)`      |
| Reindex vault        | `chat.memory.vault_indexer.reindex(...)`        |

The web UI **never** opens a new write path to canon files. It
**never** ships its own copy of `record_decision`. If a feature
requires server-side work that doesn't already exist as an API,
that API is built in the appropriate plane *first*, then the UI
calls it.

> **Why this matters.** Three implementations of `record_decision`
> drift apart. One drifts behind on validation, one ships a new
> field the others don't know about, one races on a write boundary.
> By the time the bug surfaces it has happened in production and
> the other two surfaces appear to work. The single-substrate
> rule prevents the entire class.

### 2.2 Web surfaces == union of (CLI commands ∪ Telegram slashes)

For Phase 8 v1, the web UI is **scope-locked** to the union of
existing CLI commands and Telegram slashes. New capabilities go to
CLI/Telegram first, where the API contract gets exercised. The web
UI gets it next as a renderer. This prevents Phase 8 from
swallowing six months of work designing UX for capabilities that
don't exist yet.

### 2.3 Tailscale-only binding

The web server binds to the Tailscale tailnet interface, not
`0.0.0.0` and not `127.0.0.1` exposed via tunnel. Identity is
established by the Tailscale node identity headers
(`Tailscale-User-Login`, `Tailscale-User-Profile-Picture`); no
separate auth layer. Allow-list of node identities lives in the
same config file as the Telegram `chat_id` allow-list.

### 2.4 Read-mostly with explicit-action writes

Every page renders state read-only by default. Mutating actions
(approve, reindex, apply) require an explicit click on a labeled
button — never a side-effect of viewing. Same trust posture as
Phase 7 slashes.

### 2.5 No PII in events (carry-forward)

Same as Phase 7 §2.7. The web UI may *display* message bodies
loaded via the cold-storage reader (already in operator's
session), but it MUST NOT cause those bodies to be written into
new Plane 1 events. The event payloads stay structural.

## 3. Out of scope for Phase 8 v1

- Multi-user collaboration (comments, mentions, @-replies on
  proposals). Single operator only.
- Real-time push (server-sent events, WebSockets). Polling at a
  reasonable interval is sufficient for v1.
- Authoring UI (writing new proposals, editing patches). Operator
  edits the underlying files via their normal editor; the web UI
  re-renders on next poll.
- Vault content editing. Mount stays read-only, same as Phase 7.
- Public exposure / SaaS-ification.

## 4. Build order (placeholder — to be expanded post-Phase-7)

1. Tailscale binding + node identity allow-list.
2. Read-only event timeline + session detail viewer.
3. Cold-storage replay rendering (reuses `cold_storage.py`).
4. Pattern + proposal browsers.
5. Approve/reject/defer write actions (calls existing
   `record_decision`).
6. Vault source management (calls existing indexer APIs).
7. Tuning-knob editor (writes to the same YAML config files the
   CLI reads).
8. Apply / harness trigger buttons (calls existing CLI entrypoints
   in-process).

## 5. Exit criteria (high-level)

- Web UI demonstrably implements every CLI command and every
  Telegram slash, and no more.
- A grep for `record_decision`, `apply_patch`, `detect_all` shows
  the web UI calling them but never re-implementing them.
- Operator can complete a full approve → apply → review cycle
  without leaving the browser, and the resulting events,
  decisions, and verdicts are byte-identical to the CLI/Telegram
  paths.
- Tailscale-only binding verified: a request from a non-tailnet IP
  fails to connect.
