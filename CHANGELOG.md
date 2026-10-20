# Changelog

All notable changes to AEGIS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); AEGIS is pre-`1.0`
and pre-contract (see the README caveats), so `0.0.x` bumps do not imply
stability guarantees.

## [Unreleased]

### Added
- **Fresh-install path that works** — `make setup` on a clean clone now ends green. New
  `templates/workspace/` supplies the identity/memory files bootstrap copies into
  `~/.aegis/workspace/`; bootstrap also seeds the bundled skills and prints the
  `ollama pull` commands to run next. README gains a copy-paste Quickstart.
- `AEGIS_ROOT` is honoured by bootstrap and doctor, as it already was by the runtime.
- Doctor warns (`skills:example_paths`) while an installed skill's example paths still
  contain the `YOUR_USERNAME` placeholder, and prints the `sed` that fixes it.

### Fixed
- **Doctor** no longer demands `OPENROUTER_API_KEY` for an all-local setup (required
  only when `MODEL_SMART_PROVIDER=openrouter`), checks the smart-tier local model, gives
  an `ollama pull` hint for a missing one, and no longer reports `gemma4:e4b` as
  installed because `gemma4:e2b` is.
- **`make run`** on a fresh install answered every intent with "I don't know how to
  handle"; the CLI now seeds the bundled skills the way the Telegram bot does.
- `.env.example` no longer pins `AEGIS_HOME=~/.aegis/workspace`. Loaded at startup, it
  overrode a relocated `AEGIS_ROOT` and sent the doctor and the bot back to the default
  workspace.
- The starter `config.json` no longer carries a `models` block the runtime ignores.
- `.gitignore`'s `workspace/` rule no longer swallows `templates/workspace/`.
- Operator-specific paths and an email address removed from tracked docs and the newsyslog
  template (now `__USER__` / `/Users/alice` placeholders). The bundled skill descriptors'
  example paths use a `/Users/YOUR_USERNAME` placeholder; the README explains how to
  fill it in.
- README documents bringing your own local model (`MODEL_SMART_LOCAL`, a `modelProfiles`
  entry, and how to measure it with the eval harness).

### Changed
- **Lint debt cleared** — `ruff check .` and `bandit` (via `[tool.bandit]` in
  `pyproject.toml`) now pass clean. Genuine issues were fixed (line length,
  one-line statements, magic-value constants, a docstring en-dash); deliberate
  patterns were centralized in config with justification (argv-no-shell
  subprocess, the `PASS` enum, opinionated complexity thresholds, test-only
  conventions). `make lint` is green.
- **CI added** — `.github/workflows/ci.yml` runs `make lint` + `make test-unit`
  + `make test-e2e` as blocking checks on every PR and push to `main`; `make
  type` (mypy --strict) runs non-blocking until its ~7 pre-existing errors are
  cleared. See `docs/MAINTAINING.md`.

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
