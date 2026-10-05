# Harness ablation: which mechanism earns its latency (design)

**Date:** 2026-10-06 · **Status:** approved in brainstorming, awaiting spec review
**Context:** `docs/superpowers/plans/2026-10-04-mac-mini-rerun-and-bare-arm.md`, section 18
(evidence review).

## 1. Goal and decision

The goal is a harness that supports small-language-model (SLM) delivery reliably on this
Mac Mini (M4, 16 GB, Ollama). The evidence so far (section 18 of the plan doc) shows the
full AEGIS harness helps 3-4B models after the three reply fixes, costs 3-6x latency for
no correctness gain at 9B, and breaks `lfm2.5:8b` entirely. The harness bundles several
mechanisms, and we do not know which of them earns its cost.

**Decision this study informs: an adaptive harness.** For each model (or family), which
harness configuration should ship? Success is a per-model recommended configuration,
backed by data, written as `ModelProfile` values.

## 2. Mechanisms in the full harness

| # | Mechanism | Today | Model calls | Portable to native loop |
|---|---|---|---|---|
| M1 | JSON planner protocol (grammar-constrained, `remaining` first) | `Tier1Reasoner.plan_next` | one per step + a finishing call | n/a (it is the protocol) |
| M2 | Plan-lock (offers only the planned next tool) | `_run_multi_step` (`plan_ids`, `cursor`) | none | no (needs `remaining`) |
| M3 | Path autofill from the previous search result | `_autofill_path_from_prior_search` | none | yes |
| M4 | Completion gate + claim annotation | `_gate_completion`, `annotate_unverified_claim` | none | yes |
| M5 | Reply construction: synthesis call, or summary + evidence block | `_synthesize_chain`, `_with_evidence` | synthesis: +1 | yes |
| M6 | Evidence ledger (event records) | `_record_tool_call` | none | bookkeeping only; not ablated |
| M7 | Destructive-tool guard (step >= 2) | `_run_multi_step` | none | must be ported; not ablated |

All latency comes from M1 and the synthesis part of M5. M2-M4 affect correctness only.

## 3. Architecture

Approach: **build up from bare.** The guardrails and the native tool-calling loop become
production code, and every eval arm is a preset of one runtime configuration, so the
winning configuration can ship without reimplementation.

```
runtime/harness/native_loop.py   moved from runtime/eval/bare.py
    run_native_loop, native_chat, build_tool_specs, parse_tool_calls
    history recorded as (ToolIntent, ToolResult) pairs; a hook before each tool runs
runtime/harness/guardrails.py    new; pure functions, no model calls
    autofill_path(intent, history)        <- _autofill_path_from_prior_search
    gate_claims(reply, history)           <- annotate_unverified_claim + verified tools
    with_evidence(reply, history, limits) <- _with_evidence / _render_chain
runtime/chat/telegram/native_turn.py   new; the native-mode turn (section 4)
HarnessDispatcher
    harness_mode = planner -> _run_multi_step (+ plan_lock switch)
    harness_mode = native  -> native_turn
```

**Configuration.** `ModelProfile` gains:
- `harness_mode`: `planner` | `native` (unset = `planner`, today's behaviour)
- `guardrails`: set of `autofill`, `claim_gate`, `evidence`, `synthesis`, `plan_lock`.
  Unset means the mode's default: in `planner` mode, today's behaviour (autofill,
  claim_gate, evidence and plan_lock on; synthesis only on the planner's `respond`
  path); in `native` mode, the `native+guard` preset (autofill, claim_gate, evidence
  on; synthesis off).

Invalid values are rejected when the config loads, including `plan_lock` in `native`
mode (it needs the planner's `remaining` list, so it cannot apply). An unset profile
changes nothing.

**Continuity control.** `runtime/eval/bare.py` keeps the existing `bare` arm exactly as it
is (importing the moved loop unchanged), so three nights of bare-arm data stay
comparable.

## 4. A native-mode turn

1. **Prompt and tools.** The same system prompt (`BARE_SYSTEM_PROMPT`) and tool specs as
   `bare`, so `native` differs from `bare` only by code path and guardrails.
2. **Loop.** The model requests tools; `autofill_path` corrects a guessed path (if on);
   the tool runs through the same `HarnessAdapter` (same allowlists) as the planner; the
   call is recorded in the ledger; the result goes back to the model clipped to the
   model's `tool_result_chars` (default 3,500; `bare` uses a flat 8,000, which does not
   matter for the easy and hard suites' short outputs).
3. **Reply order:** optional synthesis, then the claim gate, then the evidence block. The
   gate checks only the model's own words, never the appended evidence.
4. **Edge cases:**
   - Empty final message: the reply is the evidence block (evidence on) or empty
     (evidence off, as `bare`).
   - Step cap reached: "Stopped after N steps." plus the evidence (evidence on), or no
     reply (evidence off, as `bare`). Cap = `max_steps` (5), same as the planner.
5. **Destructive-tool guard, always on in native mode:** a `files_delete`, `files_move` or
   `files_write` call at step 2 or later is intercepted and routed to the existing
   confirmation flow. No eval task uses these tools; this is production safety, not an
   ablated variable.
6. **Errors:** a transport or model failure is logged and the turn returns PASS (the
   chat path answers), matching how the planner path degrades. An unknown tool name
   returns an error result to the model. Garbled arguments become `{}` and the tool's own
   validation reports the error.

**To verify during planning:** how the eval runner enters `dispatch()`, and that the
intent classifier still routes first in native mode (native mode then offers the full
tool catalogue, as `bare` does).

## 5. Arms

| Arm | Mode | autofill | claim_gate | evidence | synthesis | plan_lock | Role |
|---|---|---|---|---|---|---|---|
| `bare` | eval-only control | - | - | - | - | - | continuity |
| `native` | native | off | off | off | off | - | parity check only |
| `native+guard` | native | on | on | on | off | - | candidate for capable models |
| `native+guard+synth` | native | on | on | on | on | - | is a second call worth it? |
| `full` | planner | on | on | on | on `respond` | on | today's harness |
| `full-lock` | planner | on | on | on | on `respond` | off | is plan-lock worth it? |

`--arm` accepts these six names; every result file records its arm.

## 6. Run plan

**Models (5, one per observed pattern):** `llama3.2:3b` and `qwen3.5:4b-mlx` (planner
helps), `gemma4:e4b-mlx` (bare wins), `lfm2.5:8b` (planner protocol breaks it),
`qwen3.5:9b` (9B latency cost).

**Suites:** easy (`--repeat 3`, n=45) and hard (`--repeat 2`, n=30) on all arms except
`native`; honesty (`--repeat 2`, n=30) on `bare`, `native+guard`, `full`.

**Parity check (before any night):** `native` vs `bare`, `gemma4:e4b-mlx`, easy suite,
~10 min. Both run at temperature 0 with the same prompt and tools. Pass: reply
correctness within 10 points. Fail: no night starts until the difference is explained.

**Schedule** (same pre-flight, per-config wall caps and 07:55 deadline as the
2026-10-05 night run; ordered by value):

| Night | Arms | Estimate |
|---|---|---|
| 1 | `bare`, `native+guard`, `native+guard+synth` on easy + hard; honesty for `bare`, `native+guard` | ~7.5 h |
| 2 | `full` on hard; `full-lock` on easy + hard | ~8 h |
| 3 (shared) | `full` on honesty; stage 6 remainder for the other two SLMs (`gemma4:e2b-mlx`, `qwen3-vl:4b`) | - |

`full` on easy is **reused** from `eval/results/fixed/easy` (2026-10-05 night) only if
build step 1 is proven behaviour-preserving by the existing dispatcher tests; otherwise
it is added to night 2. Results go to `eval/results/ablation/<suite>/`.

## 7. Metrics and decision rule

**Primary: correct within budget.** Share of variants whose reply contains the task's
`answer_contains` text AND whose every model call finished in 30 s or less without a
timeout. A task without `answer_contains` counts a tool-call pass as correct.

**Veto: hallucination.** An arm is ruled out if its grader flag rate exceeds the cheaper
arm's by 10 points or more AND hand-reading confirms at least half of the extra flags are
real (the grader is lexical and has produced false positives).

**Reported alongside:** median turn time, model calls, tokens, TGC.

**Per-model decision:**
1. Choose the non-vetoed arm with the highest correct-within-budget score among the
   shippable arms `native+guard`, `native+guard+synth`, `full-lock`, `full`.
2. If a cheaper arm is within 15 points (about the 95% interval at n=30-45), ship the
   cheaper arm. Cheapness order (model calls per turn): `native+guard` <
   `native+guard+synth` < `full-lock` <= `full`.
3. `bare` is the reference, not a default candidate: the deterministic guardrails cost no
   model calls, so on a tie the safety layer stays. Ship guardrails-off native mode only
   if `bare` beats `native+guard` by 15 points or more (the guardrails actively hurt).
4. Keep plan-lock only if `full` beats `full-lock` by 15 points or more for at least one
   model; otherwise remove it.
5. Unmeasured models inherit by family, marked "inferred" until run: `gemma4:e2b-mlx` <-
   `gemma4:e4b-mlx`; `llama3.1:8b` <- `llama3.2:3b`; `qwen3-vl:4b` <- `qwen3.5:4b-mlx`.

## 8. Build order and tests

Each step is test-first and ends with the unit suite and `ruff check runtime tests` green.

1. Extract M3, M4, M5-evidence into `runtime/harness/guardrails.py` with no behaviour
   change; the existing dispatcher tests are the proof.
2. Plan-lock switch in `_run_multi_step` (default on). Test: with it off, step 2 sees the
   full catalogue.
3. Move the native loop to `runtime/harness/native_loop.py`; `runtime/eval/bare.py`
   imports it unchanged. Existing `tests/test_eval_bare.py` must pass untouched.
4. `native_turn.py`. Tests with a scripted fake model: autofill fixes a bad path; the
   gate annotates an unverified claim and runs before the evidence; an empty final
   message becomes the evidence; the step cap gives the note plus evidence; synthesis
   makes exactly one extra call and falls back when empty; a destructive tool at step >= 2
   is intercepted into the confirmation flow; an unknown tool returns an error result.
5. `ModelProfile.harness_mode` and `guardrails`. Tests: unset = today's behaviour;
   invalid values rejected.
6. Dispatcher wiring for `harness_mode = native` (after verifying the eval entry point).
7. Eval arms: `--arm` presets; arm recorded in results. Test: each name maps to its
   preset; `bare` unchanged.
8. Live: smoke-test each new arm on one model, one variant; then the parity check.

## 9. Out of scope (future work)

- A production persona/identity prompt for native mode (the ablation keeps the `bare`
  prompt so the comparison stays clean).
- Multi-turn tasks, tools beyond the file tools, the large-context suite.
- Run-to-run variance beyond the existing repeats (section 18.4, item 3).
- Hand-labelled calibration of the hallucination grader beyond the veto check (18.4,
  item 5).
