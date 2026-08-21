# Session Handoff — 2026-08-21

For a fresh Claude Code session picking this up. Read this first, then follow the pointers below rather than re-deriving context — the real detail lives in the specs/plans/memory files this points at, not duplicated here.

## What this session did, in order

1. **Memory sizing / eval sanity checks** on the Mac Mini (16GB RAM) via Ollama over Tailscale — resident-memory behavior of `gemma4:e2b-mlx`/`e4b-mlx` under sustained multi-step load, then reverted.
2. **Follow-up Medium article** — extended `2026-08-20-aegis-medium-article-benchmark-results.md` (repo root) across several rounds as new findings landed. **Not yet finalized or published** — this is explicit unfinished business per the user.
3. **Cross-family model benchmarking** against AEGIS's own eval harness (`runtime/eval/`, `eval/tasks/`): Qwen3-VL (a `ReadTimeout` confound, not a real result — documented as such), Llama 3.2 3B (local) and 3.1 8B (OpenRouter), LFM2.5:8B (local), Qwen3.5:4b-mlx (local — a genuine model non-determinism finding: one run showed clean zero-engagement, an immediate re-probe of the identical input showed real tool-call success followed by a *different* timeout at the reply-synthesis step, not the planning step), Qwen3.5-9b (OpenRouter). Full results and root-cause traces are in the article draft and in memory (see below) — not repeated here.
4. **Three merged features**, each via brainstorm → spec → plan → subagent-driven-development → merge into `chore/ruff-zero-and-ci`:
   - **Multi-step plan lock** (commits `59bba8a`, `c77b877`) — locks the planner's tool choice to a cursor derived from the model's own first successful step, once execution has started. Spec: `docs/superpowers/specs/2026-08-21-multi-step-plan-lock-design.md`.
   - **Classifier wiring fix** (`f1c8246`, bounded, no spec doc) — `build_harness_dispatcher` was wiring the intent classifier to `cfg.models.smart_local` (the SMART-tier planning model) instead of `cfg.models.fast`, `ModelConfig`'s own documented Tier-0-classifier field. This meant every cross-family eval comparison this session never actually held classification constant. Fixed; classification is now genuinely decoupled from whatever's under test.
   - **Classification fallback to the planner** (`355207a`, `33cb177`) — when intent classification finds no matching skill, `dispatch()` now falls through to the multi-step planner with the full catalog instead of silently doing nothing. Root-caused a real, previously-silent failure class (compound "do X then Y" requests). Spec: `docs/superpowers/specs/2026-08-21-classification-fallback-design.md`. **Has a real, explicitly out-of-scope follow-up** — see "Open threads" below.
5. Two prompt-only fix attempts, both measured to fail (consistent with the session's repeated finding that prompt nudges don't reliably fix small-model behavior where schema/structural constraints do): the `remaining` field (`97eb1da`, earlier session — improved aggregate numbers, didn't fix the target failure) and a path-copy-verbatim instruction (`ab8a49c`) for the `search_then_read` path-guessing problem, which changed nothing.
6. Added `VariantResult.actual_calls` to the eval harness (`78f84e0`) so failed variants show what the model actually did, not just pass/fail.

## Current repo/infra state (verified, not assumed, at end of session)

- On `chore/ruff-zero-and-ci`, **40 commits ahead of `origin/chore/ruff-zero-and-ci`, never pushed**. All of this session's work is local-only.
- Working tree: only `CLAUDE.md` shows as modified, and it was already modified before this session started (confirmed multiple times) — not this session's doing, leave it alone unless the user says otherwise. ~640 untracked files, almost all `graphify-out/` cache output and pre-existing scaffolding from before this session — not session-generated, not worth auditing.
- No open worktrees (all three used this session — plan-lock, classification-fallback, and one earlier — were cleaned up after merging).
- `~/.aegis/.env`'s model pins are back at their pre-session baseline: `MODEL_SMART_PROVIDER=ollama`, `MODEL_SMART_LOCAL=gemma4:e2b-mlx`, `MODEL_SMART=deepseek-v4-pro:0813-cloud`. No model currently resident in Ollama.
- Two background processes from earlier in this session are **still running** and were not touched: `ollama-tailscale-proxy.py` (PID 1208028, forwards 127.0.0.1:11434 → the Mac Mini's real Ollama over Tailscale) and `scripts.telegram_smoke` (PID 2132293, the live bot). Check `ps aux` before assuming either is or isn't still alive by the time you read this.

## Open threads — real, named, not yet started

1. **The classifier's fail-open-on-outage gap.** `ModelBackedClassifier.classify()` swallows its own transport/schema failures into a returned `intent="unknown"` rather than raising — indistinguishable from a genuine "the model doesn't know" classification. Before the classification-fallback merge, both cases meant "touch nothing" (safe). Now both reach the full-catalog planner; a classifier *outage* specifically fails open instead of safe. This is documented as an explicit, named decision in `docs/superpowers/specs/2026-08-21-classification-fallback-design.md`'s Safety section — not silently accepted — but fixing it properly requires changing `runtime/intent/classifier.py`'s own return contract, which was out of scope for that branch. This is real, scoped-out follow-up work, not a forgotten bug.
2. **`search_then_read`'s path-guessing problem.** The model constructs a wrong `files_read` path after a successful `files_search`, even though the search result contains the correct path (confirmed via code trace — not a truncation bug). A prompt-only fix failed. The next real candidate, not yet built: have the harness auto-fill `path` from the immediately-prior search match deterministically, rather than asking the model to retype it.
3. **The article draft** (`2026-08-20-aegis-medium-article-benchmark-results.md`) is content-complete through this session's findings but not finalized or published. Whether/how to close it out is the user's call.
4. **Never pushed to origin.** 40 local commits on `chore/ruff-zero-and-ci`. Whether/when to push or open a PR wasn't discussed this session.
5. Testing bigger models across the remaining families (a bigger LFM, a mid-size Gemma like 12B) was raised as a bonus option but never pursued — open if the user wants more cross-family data points.

## Where to look for real detail

- **Memory** (persists across sessions automatically): `project_eval_harness_motivation.md`, `project_multi_step_plan_lock.md`, `user_collaboration_style.md`, `feedback_eval_ledger_args.md` — indexed in `MEMORY.md`. These already carry the "why" and the measured numbers; don't re-derive them from git log.
- **Specs**: `docs/superpowers/specs/2026-08-21-*.md` (three from this session).
- **Plans**: `docs/superpowers/plans/2026-08-21-*.md` (two from this session; the classifier wiring fix was bounded, no plan doc).
- **Eval results**: `eval/results/*.json` (gitignored, local only — re-run `make eval` or `.venv/bin/python -m runtime.eval.cli --yes` if you need fresh numbers; they won't persist across a fresh clone).
