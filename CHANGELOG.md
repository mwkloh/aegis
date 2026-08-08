# Changelog

All notable changes to AEGIS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); AEGIS is pre-`1.0`
and pre-contract (see the README caveats), so `0.0.x` bumps do not imply
stability guarantees.

## [Unreleased]

## [0.0.2] - 2026-08-08

Phase 11 — capability floor + evidence ledger. Lifts AEGIS from "chatbot with
bounded skills" to "small agent with verified actions" without loosening any
reliability guarantee. Design and open questions in
`docs/PLAN_PHASE_11_CAPABILITY_FLOOR.md`.

### Added
- **Schema-constrained decoding** — `ChatRequest.response_schema` is passed to
  Ollama's `format` so tool-call/plan JSON is grammar-constrained in the
  decoder (GBNF). Deterministic `repair_json()` salvages wrapper noise before
  spending a corrective retry.
- **Evidence ledger** — the harness records every tool execution as a
  structural proof record (skill, tool, argv hash, byte count, verdict), scoped
  by turn, via `runtime/tools/record.py`.
- **`task_complete` plan kind + completion gate** — the model's "I'm done"
  claim is checked against verified ledger evidence; unrecovered failures get an
  honest warning. Annotate-not-block (measured via `harness.completion_gated`
  events).
- **Confirmation flow** for the destructive guard — pending intents are held and
  executed only on an exact-match affirmative within a 120s TTL, armed only
  after the prompt is delivered.
- **`files_write` tool** — atomic tmp-then-rename, sandbox-root containment,
  256 KiB cap, symlink-escape rejection; guarded and off by default.
- **`run_command` tool** — argv-only (no shell) against an operator binary
  allowlist, with path arguments sandboxed to the same roots as `files_read`;
  off by default. `CommandsConfig` is operator-configurable via `config.json`.

### Changed
- `verdict_for_result` honors a tool's own payload verdict, so a soft failure
  (e.g. a non-zero `run_command` exit) is recorded and gated as a failure
  rather than counted verified.

### Notes
- The confirmation flow and multi-step tools require `harness.multi_step`.
- Open questions (deferred, tracked in the phase plan): the ledger idempotency
  key must gain the verdict before the completion gate can move from annotate to
  hard-block.

## [0.0.1] - 2026-08-07

Initial public release. Bounded multi-step tool loop, destructive guard, reply
verdict gate, skills system, local-model-first routing, tiered memory, scheduler
with recurring system jobs, Telegram + CLI surfaces. MIT licensed.

[Unreleased]: https://github.com/mwkloh/aegis/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/mwkloh/aegis/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/mwkloh/aegis/releases/tag/v0.0.1
