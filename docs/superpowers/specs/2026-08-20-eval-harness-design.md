# Aegis Eval Harness Design

## Goal

Aegis's whole premise is that a harness built around small/local models (gemma4:e2b/e4b, llama3.1:8b — routed via the `smart_provider` pin in `runtime/llm/router.py`) can reliably decompose and execute multi-step tool calls, with the harness's own guardrails (destructive-tool confirmation, the evidence ledger, `_gate_completion`'s claim-vs-evidence check) compensating for the model's own unreliability. That claim has never been measured — only argued architecturally. This design builds a benchmark that actually measures it: does `HarnessDispatcher`'s multi-step planner (`runtime/chat/telegram/harness_dispatcher.py`) complete real tasks against Aegis's real skill catalog, using whichever model is actually pinned in the operator's live config.

Borrowed framing (from IBM Research's ALTK-Evolve article, reviewed earlier this session) — two metrics, not one:
- **TGC** (lenient): fraction of individual task-variant runs that pass.
- **SGC** (strict): fraction of *tasks* where *every* variant passes. A task that passes 2 of 3 phrasings counts as failed for SGC — this is the metric that actually answers "is this reliable," not just "does it sometimes work."

## Non-Goals

- No LLM-judge grading. Matches the harness's own existing philosophy (`_gate_completion` trusts structural evidence, not self-reported text) — grading is a structural check against the evidence ledger, nothing else.
- No record-and-replay mode. Live models only for v1; replay is a plausible future layer once real recorded baselines exist, not built now.
- No repeated sampling per variant (each variant runs once per eval invocation). Statistical confidence via N repeats is a documented future extension (`--repeat` flag), not v1 — repeating would multiply live-model cost/time by N with no design payoff yet.
- No CI integration. This is a slow, live-model, cost-incurring manual tool (`make eval`), never part of `pytest -m unit` or any automated pipeline.
- No forcing `cfg.harness.multi_step=True`. The eval runs against whatever the operator's live `~/.aegis/config.json` actually has configured — if multi-step is off, the multi-step chain tasks will honestly show as failing, which is itself a meaningful signal about the live deployment, not a bug to work around.

## Architecture

New top-level `eval/` directory for human-authored content (mirrors the existing `coding_harness/` top-level docs directory):
```
eval/
  tasks/*.yaml       — task definitions
  results/*.json     — timestamped output artifacts (gitignored)
```

New `runtime/eval/` package for runner code (mirrors `runtime/coding_harness/`):
```
runtime/eval/
  __init__.py
  tasks.py      — EvalTask/ExpectedCall/Fixture Pydantic models + YAML loader
  runner.py     — per-(task, variant) execution + grading
  report.py     — EvalReport model, TGC/SGC computation, console + JSON rendering
  cli.py        — entrypoint: python -m runtime.eval.cli
```

The runner reuses `build_harness_dispatcher` from `runtime/chat/telegram/bot.py` directly — the real classifier, real `Tier1Reasoner`, real synthesizer, whatever `smart_provider`/model the operator's `~/.aegis/.env` currently pins. Running the eval against a different model is changing that config and re-running; no separate eval-specific model config surface exists.

## Task Definition Format

YAML, matching `SkillDescriptor`'s own `safe_load`-only convention (`runtime/skills/registry.py`):

```yaml
id: search_then_read
description: "Search for files matching a pattern, then read the first result."
fixture:
  files:
    - path: "notes/CT-001-notes.md"
      content: "Some notes about CT-001."
    - path: "notes/other.md"
      content: "Unrelated notes."
variants:
  - "find files about CT-001 in {sandbox}/notes and read the first one"
  - "search {sandbox}/notes for CT-001 and open the top match"
expected_calls:
  - tool: files_search
    args_match: {glob: "*CT-001*"}
  - tool: files_read
    args_match: {}
```

- `fixture.files`: seeded into a fresh sandbox directory before each variant run — see Fixture Isolation below.
- `variants`: each is literal `user_text` sent to the harness, with `{sandbox}` substituted for that run's actual sandbox path immediately before dispatch. A task with no fixture needs (e.g. a pure `time_query` task) omits `fixture` and `{sandbox}` from its variants entirely.
- `expected_calls`: an ordered list. `args_match` is a **partial** dict — only listed keys are checked, and `{sandbox}` is substituted into `args_match` string values the same way as into variant text before comparison. String values compare via **substring containment**, not exact equality (a resolved/expanded filesystem path from `FilesClient` — e.g. after `~`-expansion or `.resolve()` — won't byte-for-byte match the literal fixture path string, but will contain it); non-string values (numbers, bools) compare via exact equality.

## Fixture Isolation (safety-relevant, non-negotiable)

Every `(task, variant)` run gets its own fresh temp sandbox directory (`tempfile.mkdtemp()`-style), seeded per `fixture.files`, and that run's `FilesClient` is constructed with `allowed_roots=[sandbox]` — **never** the operator's real home directory or any of the default `FilesConfig` roots. Task variant text references `{sandbox}`, never a literal path like `~/Downloads`. A live-model eval harness must not be able to touch real files, full stop — this is the one hard constraint in this design.

## Run Isolation

Per `(task, variant)` run, constructed fresh: `Tier3Store()`, `EventStream` (pointed at a per-run temp sessions dir), sandboxed `FilesClient`, and the `HarnessDispatcher` itself (built via `build_harness_dispatcher`, which needs these plus the shared `SkillRegistry`/`Tier1Loader`). `SkillRegistry` and `Tier1Loader` are built once and shared read-only across all runs — no isolation need, no run-to-run state.

## Grading

After each run, the runner reads back `load_tool_calls(events)` for that run's `EventStream`, filters to `verdict_for_result(...) == "verified"` entries (reusing the existing helper from `runtime/tools/record.py`), and walks `expected_calls` as an ordered **subsequence** match against the actual verified calls — each expected call must match some actual call at or after the position of the previous match. This tolerates incidental extra tool calls (e.g. a preliminary lookup) without failing the task, while still requiring the expected calls to happen, with matching args, in the right relative order. A variant that never gets far enough to make an expected call fails at the point of the missing match.

## Metrics & Reporting

```
$ make eval
Pinned: ollama / gemma4:e2b-mlx  |  8 tasks x 3 variants = 24 runs

TGC (per-run):          62.5% (15/24)
SGC (per-task, strict):  37.5% (3/8)

list_downloads       3/3 variants  PASS
search_then_read      2/3 variants  PARTIAL  (variant 2: files_read never called)
...

Written: eval/results/2026-08-20T14-30-00Z-ollama-gemma4-e2b-mlx.json
```

The JSON artifact captures per-run results (task id, variant text, pass/fail, which expected call failed if any, timing) so results from different models/dates can be diffed later — not built in v1, but the artifact format is designed to make that comparison possible without re-running anything.

## Error Handling & Cost Safety

Any run that errors (model unreachable, timeout, malformed response) is recorded as a structured FAIL with the exception message, never crashes the batch — matches the "never raise" convention already used throughout `runtime/coding_harness/`, `runtime/reflection/`, and `runtime/chat/pipeline.py`. Before running, the CLI prints the pinned provider/model and the total call count (`tasks × variants`) and requires confirmation before proceeding — relevant cost signal especially when `smart_provider=openrouter` is the live pin.

## V1 Task Scope

~6-10 task templates: most single-tool (one per representative skill: `list_files`, `read_file`, `search_files`, `time_query`), 2-3 deliberately chaining 2+ tools to exercise `_run_multi_step` specifically (the actual thing this benchmark exists to measure), each with 2-3 phrasing variants. Skills with side effects beyond the sandbox (`morning_brief` writes to the vault, `tier2_compress`, `vault_reindex`, `reflection_sweep`) are out of v1 scope — they need real vault/memory-db fixtures this design doesn't build yet, not just a sandboxed folder. `ask_question`/`echo` are trivial enough that a v1 task for each isn't high-value.

## Testing

The runner code itself (`runtime/eval/tasks.py`'s YAML loader, `runtime/eval/runner.py`'s subsequence-match grading logic, `runtime/eval/report.py`'s TGC/SGC computation) gets normal unit tests under `tests/` with fakes — no live model calls in the unit-test suite. Only `runtime/eval/cli.py`'s actual end-to-end invocation requires a live model and stays outside `pytest -m unit` entirely, run manually.
