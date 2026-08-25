# Eval Measurement Confounds: Findings and Instrumentation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eval suite able to distinguish *"the model could not do it"* from *"the harness cut it off"*. Today those two outcomes are indistinguishable in `eval/results/*.json`, and that single ambiguity produces most of the outliers in the 2026-08-23 rerun.

**Non-goal (this plan):** changing planner behaviour. Thinking mode stays enabled for the SMART tier; we measure its cost first and decide separately. Do not "fix" a model's score as part of this work.

**Context:** 2026-08-23 rerun of 9 configs (`eval/results/2026-08-23T*.json`) vs the 2026-08-20/21 baseline. Findings below are measured, not inferred.

---

## Findings (evidence)

Ruled out first: **network** (Tailscale RTT to `100.64.170.84` measured at 1.24 ms avg) and **remote-host degradation** (warm throughput normal — see F3).

Environment note, not itself a confound but required to read any of this: `runtime/llm/clients/ollama_client.py:39` `_validate_local()` enforces a loopback host, but `127.0.0.1:11434` is served by `~/.aegis/ollama-tailscale-proxy.py`, a raw TCP forward to a remote Mac. Every "local" call crosses a machine boundary. The loopback guard does not mean what it appears to mean.

### F1 — Cold model load consumes 78% of the timeout budget

Measured on `qwen3-vl:4b`, cold: `load_duration` **23.48 s** against `read=30.0 s` (`runtime/llm/clients/ollama_client.py:29`). `/api/ps` reported no resident models. Running 9 configs back-to-back forces evictions, so this tax lands *inside* the measured window, unpredictably.

### F2 — Thinking mode is uncapped on the planner tier

`2537913` disabled thinking for Tier-0 classification only. Measured on `qwen3-vl:4b` warm, `num_predict=512`: all 512 tokens went to `thinking`, `content` empty, `done_reason:"length"`, `eval_duration` 33.5 s. That alone exceeds the 30 s read timeout.

### F3 — Warm throughput varies ~8x across models

| Model | warm tok/s |
|---|---|
| `gemma4:e2b-mlx` | 117.8 |
| `lfm2.5:8b` | 80.5 |
| `gemma4:e4b-mlx` | 78.3 |
| `qwen3-vl:4b` | **15.0** |

A single fixed 30 s timeout is therefore a **different effective token budget per model**. The suite is not measuring the same thing across the table.

### F4 — Retry amplification manufactures clean-looking failures

`stop_after_attempt(3)` with `wait_exponential(multiplier=0.2, min=0.2, max=2.0)` (`ollama_client.py:131-134`) → 3 × 30 s + backoff ≈ **90.6 s**. Observed `qwen3-vl:4b` durations: 91.7–97.4 s across all 15 variants, a band far too tight to be inference variance. Every call exhausted all three retries. In the JSON this is indistinguishable from engagement-then-wrong-answer; `reason` only ever says "expected call to X never found".

### F5 — Two visually identical 0% scores have different mechanisms

- `qwen3-vl:4b` — 0% is **entirely timeout-driven** (F2 + F4). Zero tool calls because no response ever arrived.
- `lfm2.5:8b` — 0% is a **genuine decision**. Warm throughput 80.5 tok/s, median run 17.4 s, no timeouts. It returns well-formed output choosing `respond` over `tool_call`.

Reporting both as "0% TGC" is the clearest instance of the core problem.

### F6 — The classification-fallback fix changed what is being measured

Zero-call variants dropped after the fix: `gemma4:e2b-mlx` 2→0, `llama-3.1-8b-instruct` 2→0, `qwen3.5:4b-mlx` 15→0; total call volume rose. Requests that used to die cheaply at the Tier-0 gate now run the full planner. Intended behaviour — but it means old-vs-new comparisons are not apples-to-apples, because the new runs do strictly more work per task.

### F7 — Provenance gap in the baseline

`actual_calls` was added in `78f84e0`. The 2026-08-20 result files predate it, so their zero-call readings are *field absent*, not observed behaviour. Any comparison using `actual_calls` must start from 2026-08-21.

### F8 — n=1 cannot separate flaky from regressed

`openrouter/qwen3.5-9b` run A: `time_check` 0/2, `search_files` 3/3. Run B, same config minutes later: `time_check` 2/2, `search_files` 2/3. Different task failing each run.

---

## Global Constraints

- **Do not change planner or model behaviour in this plan.** No thinking-mode changes, no prompt edits, no timeout changes to the production path. Instrumentation and eval-harness only.
- `runtime/llm/clients/ollama_client.py`'s production `_TIMEOUT` and retry policy stay as they are — the eval harness gets its own configurable budget, so measurement changes cannot silently alter runtime behaviour.
- New fields on `VariantResult` must be optional with defaults, so existing `eval/results/*.json` still parse (`extra="forbid"`, `frozen=True` on all report models — `runtime/eval/report.py:29`).
- Preserve the existing `tgc`/`sgc` definitions exactly. New metrics are added alongside, never redefining the old ones.
- Never delete or rewrite historical result files.

---

### Task 1: Capture per-call telemetry so failures are self-diagnosing

**Files:**
- Modify: `runtime/llm/clients/ollama_client.py` (surface Ollama's own timing fields + retry/timeout outcome)
- Modify: `runtime/eval/report.py:29-37` (`VariantResult`, new optional telemetry fields)
- Modify: `runtime/eval/runner.py:58-145` (`run_variant` — thread telemetry through)
- Test: `tests/test_eval_report.py`, `tests/test_eval_grading.py`

**Interfaces:**
- Ollama's non-streaming `/api/chat` response already returns `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, `done_reason`. These are free — currently discarded.
- Produces: new optional fields on `VariantResult`; no change to `tgc`/`sgc`.

- [x] **Step 1: Confirm current shapes before editing**

Run: `sed -n '25,40p;100,150p' runtime/llm/clients/ollama_client.py` and `sed -n '29,50p' runtime/eval/report.py`

Confirm `_TIMEOUT` is at line 29, the `AsyncRetrying` block at 131-134, and that `VariantResult` has exactly `task_id`/`variant_text`/`passed`/`reason`/`duration_s`/`actual_calls`. If the file has diverged, stop and report.

- [x] **Step 2: Write failing tests first**

Tests to add:
- A `VariantResult` carrying the new telemetry fields round-trips through `model_dump_json` and back.
- A `VariantResult` constructed *without* the new fields still validates (back-compat with existing result files).
- A run whose telemetry shows retry exhaustion is classified `timeout_exhausted`, not `no_tool_call`.

- [x] **Step 3: Record timing + termination on the client**

Capture from each Ollama response: `load_duration`, `eval_count`, `eval_duration`, `prompt_eval_duration`, `done_reason`, plus whether the call hit a read timeout and how many retry attempts were consumed. Derive `thinking_token_share` (F2's measurement — needed for the deferred thinking-mode decision).

- [x] **Step 4: Aggregate onto `VariantResult`**

Per variant, record: total model calls, summed load time, summed eval time, max single-call latency, retry-exhausted count, and whether any call hit the timeout.

**Task 1 verified live (2026-08-24).** Two single-task runs against real models:

- `gemma4:e2b-mlx`, passing: `{"model_calls": 5, "load_ms_total": 4535, "max_call_wall_ms": 6927, "timed_out_calls": 0, "max_thinking_token_share": 0.939}`
- `qwen3-vl:4b`, failing: `{"model_calls": 2, "max_call_wall_ms": 91232, "timed_out_calls": 1}`

Both failing `qwen3-vl:4b` variants carry the *identical* `reason` string that a wrong-answer produces — `timed_out_calls` is now the only thing separating them. That is the goal of this plan, working end to end.

**Incidental finding, feeds Task 5 Step 3:** `gemma4:e2b-mlx` shows `max_thinking_token_share` ≈ **0.94 on runs that pass**. Thinking is consuming almost the entire generation budget even where it is not fatal, so the thinking-mode question is not confined to `qwen3-vl:4b`. Also visible: the cold-load tax, `load_ms_total` 4535 ms on the first variant vs 697 ms on the second.

---

### Task 2: A failure taxonomy, so 0% stops being one bucket

**Files:**
- Modify: `runtime/eval/grading.py` (classify failures)
- Modify: `runtime/eval/report.py` (carry the label)
- Test: `tests/test_eval_grading.py`

**Interfaces:** consumes Task 1's telemetry. Produces a `failure_kind` on failing variants; passing variants carry none.

- [x] **Step 1: Write failing tests for each category**

Minimum categories, each justified by a finding above:
- `timeout_exhausted` — all retries consumed, no response (F4; `qwen3-vl:4b`)
- `declined_tool` — well-formed response, chose `respond` over `tool_call` (F5; `lfm2.5:8b`)
- `wrong_tool` — called a tool, not the expected one
- `incomplete_chain` — first step succeeded, later step never issued
- `repeated_step` — same successful call issued repeatedly until budget exhausted (the new `llama-3.1-8b-instruct` shape)
- `zero_engagement` — responded, no tool call, not a timeout

- [x] **Step 2: Implement classification from telemetry, not from `reason` strings**

`reason` is a human-readable grading message and must not become a parsed interface.

- [x] **Step 3: Surface counts in `render_console` and the JSON**

---

**Task 2 verified on real recorded runs (2026-08-24).** The same two models the
article reported as an undifferentiated "0% TGC", reclassified from their own
telemetry:

```
qwen3-vl:4b  dur=  92.0s  -> timeout_exhausted
qwen3-vl:4b  dur=  96.3s  -> timeout_exhausted
lfm2.5:8b    dur=  15.5s  -> no_tool_call
lfm2.5:8b    dur=  15.2s  -> no_tool_call
```

`render_console` now prints a "Failures by kind" block plus an explicit warning
when any variant hit a retry-exhausted timeout, so a run that measured the
harness budget rather than the model says so on its own.

**Scope correction against this plan's own Task 2 step 1:** the planned split
between `declined_tool` and `zero_engagement` was dropped. Both reach the
grader as *no observed calls, no timeout* -- the planner's `kind: "respond"`
choice is not visible at that layer, so they collapse into `NO_TOOL_CALL`.
Separating them needs planner-level instrumentation; asserting the distinction
from this evidence would have been guessing. Documented on `FailureKind`.

### Task 3: Separate capability from the product budget

Decision taken 2026-08-24: report **both** numbers from a single run rather than picking one.

**Files:**
- Modify: `runtime/eval/runner.py` (pre-warm; eval-local timeout)
- Modify: `runtime/eval/report.py` (second metric)
- Modify: `runtime/eval/cli.py` (flags)
- Test: `tests/test_eval_report.py`

- [x] **Step 1: Pre-warm the model before the measured window**

Issue a trivial call per config before the first task and discard it, so F1's 23.5 s cold load is not inside any measured run. Record the observed load time in the report header — it is a real hardware finding worth keeping, just not inside a task's duration.

- [x] **Step 2: Give the eval harness its own generous timeout**

Configurable, defaulting well above the production 30 s, so a model is measured on what it can do rather than on whether it fits the client budget. **Must not** alter `ollama_client._TIMEOUT` for the runtime path.

- [x] **Step 3: Record the product-budget verdict alongside**

For each variant, record whether the slowest single model call would have breached the 30 s production timeout. Report a second headline metric next to TGC/SGC — tasks that succeeded *and* would have fit in the real budget.

- [x] **Step 4: Verify both metrics on a fast, already-passing config**

`gemma4:e4b-mlx` should show near-identical TGC under both, confirming the change is measurement-only.

---

**Task 3 verified live (2026-08-24)** on `gemma4:e4b-mlx`, the already-passing config:

```
Pre-warming gemma4:e4b-mlx...
  resident after 8.3s (load 7.5s) -- excluded from results
TGC (per-run):          100.0%
TGC within 30s budget: 100.0%
```

Both metrics identical on a fast model, which is the check that this change is
measurement-only. The 7.5 s load it absorbed would previously have landed
inside whichever variant ran first.

Implementation notes:
- `read_timeout_override` (`ollama_client.py`) is a `ContextVar`, mirroring the
  telemetry collector. Only `read` widens -- `connect`/`write`/`pool` guard a
  wedged socket rather than a slow model. With no override active the shipped
  `PRODUCTION_READ_TIMEOUT_S` applies, so the runtime is untouched.
- `PRODUCTION_READ_TIMEOUT_S` now lives in `runtime/llm/telemetry.py` and is the
  single source for both the client's `_TIMEOUT` and the report's budget check.
  Two copies would let the benchmark grade against a budget the product no
  longer has.
- Eval default read timeout: 300 s (`--read-timeout`).

### Task 4: Repeat runs, so variance is reported not guessed

**Files:** `runtime/eval/cli.py`, `runtime/eval/report.py`, `tests/test_eval_report.py`

- [x] **Step 1: Add a repeat count (`--repeat N`, default 1)**
- [x] **Step 2: Report per-task pass *rate* across repeats, plus min/max**

F8 is the motivating case: `time_check` flipping 0/2 → 2/2 between consecutive runs must show as variance, not as a regression.

- [x] **Step 3: Keep n=1 the default** — the suite is slow and costs real tokens; opt in per investigation.

---

**Task 4 note:** `TaskResult.pass_rate` and `.is_flaky` are additive.
`all_passed` -- and therefore SGC -- keeps its strict definition; softening it
to accommodate repeats would silently redefine a published metric.

### Interlude: `qwen3-vl:4b` answered ahead of Task 5

Task 5 Step 2 asks whether `qwen3-vl:4b` clears 0% once F1 (cold load) and F4
(retry amplification) are out of the measurement. A single-task probe on
2026-08-24, pre-warmed and with a 300 s read timeout, answers it: **no — and
for a third reason, not either of the two already catalogued.**

```
qwen3-vl (old, 30s cap)      92.0s -> timeout_exhausted
qwen3-vl (new, 300s)        223.9s -> thinking_budget_exhausted
lfm2.5:8b                    15.5s -> no_tool_call
```

Telemetry for the new `qwen3-vl` run: `timed_out_calls: 0` (the timeout really
is gone), `eval_ms_total: 161467`, `max_thinking_token_share: 1.0`,
`truncated_calls: 2`, `actual_calls: []`. It generates for 161 s, spends every
single token on `thinking`, hits the token ceiling, emits no content, and so
never decodes a tool call.

**This reverses part of Task 2's scope correction, on new evidence.** Task 2
merged "declined a tool" and "never engaged" because the grader could not tell
them apart from observed calls alone — correct at the time. Task 1's telemetry
supplies the missing discriminator: `truncated_calls`. Both models show a 1.0
thinking share, so that is *not* the signal; truncation is. `qwen3-vl` is cut
off mid-reasoning, `lfm2.5` finishes cleanly in 15 s and still declines. Added
as `FailureKind.THINKING_BUDGET_EXHAUSTED`.

**Consequence for Task 5 Step 3:** the thinking-mode decision now has a
concrete failing case that is *not* a timeout and *not* a model choice — it is
`max_tokens` being consumed by a hidden reasoning channel. Capping
`num_predict` would not help here (truncation is already the symptom);
disabling thinking on the SMART tier is the change this evidence points at.

---

### Task 5 (local subset, 2026-08-24): results

Seven Ollama-provider configs, full six-task suite, instrumented. OpenRouter
held back.

| Model | TGC old | TGC new | SGC new | within 30 s budget |
|---|---|---|---|---|
| `gemma4:cloud` | 100.0% | 100.0% | 100.0% | 100.0% |
| `gemma4:e4b-mlx` | 93.3% | 93.3% | 83.3% | 93.3% |
| `qwen3.5:4b-mlx` | 86.7% | **100.0%** | 100.0% | **73.3%** |
| `gemma4:e2b-mlx` | 66.7% | 60.0% | 33.3% | 60.0% |
| `qwen3-vl:4b` | 0.0% | **40.0%** | 0.0% | **0.0%** |
| `llama3.2:3b` | 26.7% | 20.0% | 0.0% | 20.0% |
| `lfm2.5:8b` | 0.0% | 6.7% | 0.0% | 6.7% |

**The interlude above was wrong, and this run corrects it.** That conclusion --
"`qwen3-vl:4b` does not clear 0%" -- was drawn from a single-task probe on
`time_check`. Across the full suite it reaches **40% TGC**: 2/3 on
`list_downloads`, 2/3 on `read_file`, 2/3 on `search_files`. `time_check` is
one of the three tasks it still fails outright, so the probe sampled the
unrepresentative case and generalised from it. The model is substantially more
capable than either the published 0% or that probe implied.

**But its in-budget score is 0.0%.** Every passing run breached the shipped
timeout -- slowest calls per task run 103-162 s. Capable and unusable are both
true, and only the dual metric states both. A single number would have
mis-told this either way: 0% understates the model, 40% overstates the product.

`qwen3.5:4b-mlx` shows the same split more mildly: 100% TGC, 73.3% in budget --
four runs succeeded too slowly to have shipped.

**Failure taxonomy and thinking cost:**

| Model | max thinking share | truncated calls | failure kinds |
|---|---|---|---|
| `gemma4:cloud` | 0.00 | 0 | — |
| `gemma4:e4b-mlx` | 1.00 | 1 | incomplete_chain=1 |
| `qwen3.5:4b-mlx` | 1.00 | 4 | — |
| `gemma4:e2b-mlx` | 1.00 | 5 | incomplete_chain=2, repeated_step=2, wrong_tool=2 |
| `qwen3-vl:4b` | 1.00 | **22** | thinking_budget_exhausted=9 |
| `llama3.2:3b` | 0.00 | 0 | no_tool_call=6, wrong_tool=4, repeated_step=2 |
| `lfm2.5:8b` | 1.00 | 3 | no_tool_call=11, thinking_budget_exhausted=3 |

Five of seven models spend **100%** of some call's output budget on `thinking`.
`lfm2.5:8b` is now firmly characterised: 11 of 14 failures are genuine
declines, not artifacts -- the published reading of that model survives.

**Not treated as regressions — now settled by `--repeat 3` (n=45 each):**

| Model | n=1 (08-23) | n=1 (08-24) | **n=45** | verdict |
|---|---|---|---|---|
| `gemma4:e2b-mlx` | 66.7% | 60.0% | **64.4%** | noise — true value sits between the two single reads |
| `llama3.2:3b` | 26.7% | 20.0% | **24.4%** | noise — same |

Neither drop was real. Both single-run readings bracket the repeated measure,
which is exactly what sampling noise looks like.

The repeat pass also produced a finding the aggregates hide: **most tasks are
individually flaky for these two models.** `gemma4:e2b-mlx` — `list_downloads`
5/9, `list_then_read` 3/6, `search_files` 6/9. `llama3.2:3b` — four of six
tasks split, including `read_file` at 1/9. These models do not have a per-task
pass/fail state at all; they have a pass *rate*, and any n=1 reading of them
(including every number in the published article) carries error bars wide
enough to swallow the differences it reports.

`llama3.2:3b` also shows 24.4% TGC against 22.2% in budget — one run succeeded
too slowly to have shipped.

**Thinking-mode decision input (Task 5 Step 3):** `qwen3-vl:4b` has 22 truncated
calls and 9 `thinking_budget_exhausted` failures -- the single largest
identified failure bucket in this run, and one that a `num_predict` cap cannot
help because truncation is already the symptom. Disabling thinking on the SMART
tier is the change the evidence supports, as its own spec'd before/after.

---

### Task 5 (complete): all 9 configs, instrumented

| Model | TGC | SGC | within 30 s budget |
|---|---|---|---|
| `gemma4:cloud` | 100.0% | 100.0% | 100.0% |
| `qwen3.5:4b-mlx` | **100.0%** | 100.0% | **73.3%** |
| `gemma4:e4b-mlx` | 93.3% | 83.3% | 93.3% |
| `meta-llama/llama-3.1-8b-instruct` | 93.3% | 83.3% | 93.3% |
| `qwen/qwen3.5-9b` | 93.3% | 83.3% | **80.0%** |
| `gemma4:e2b-mlx` | 64.4% (n=45) | 33.3% | 64.4% |
| `qwen3-vl:4b` | 40.0% | 0.0% | **0.0%** |
| `llama3.2:3b` | 24.4% (n=45) | 0.0% | 22.2% |
| `lfm2.5:8b` | 6.7% | 0.0% | 6.7% |

**The budget gap is not a local-hardware artifact.** `qwen/qwen3.5-9b` — hosted,
nothing to do with this Raspberry Pi or the Mac behind the Tailscale proxy —
scores 93.3% TGC but only **80.0% in budget**, with three individual calls at
38.7 s, 32.1 s and 39.5 s. A hosted provider breaches the shipped 30 s timeout
too. `meta-llama/llama-3.1-8b-instruct` at the same 93.3% TGC stays fully
inside it (93.3%), so this separates two models that a single TGC column
reports as identical.

**A local 4B beats both hosted 8-9B models on capability.** `qwen3.5:4b-mlx`
reaches 100% TGC / 100% SGC against 93.3% / 83.3% for `qwen3.5-9b` and
`llama-3.1-8b-instruct` — but at 73.3% in budget versus 80.0% and 93.3%. That
is the project's own thesis in one row: the local small model can do the work,
and the thing it actually lacks is speed, not capability.

**Nothing regressed.** Every config is at or above its previous number once
measurement artifacts are removed. The three that moved up —
`qwen3.5:4b-mlx` 86.7→100%, `qwen3-vl:4b` 0→40%, `lfm2.5:8b` 0→6.7% — moved
because the old measurement was wrong, not because anything was fixed.

---

### Follow-up (2026-08-25): thinking mode measured both ways — it is a knob, not a default

Task 5 Step 3 pointed at disabling thinking on the SMART tier as the only
evidence-backed lever. Implemented as `think=False` on both Tier-1 calls, then
re-run across the 7 local configs. **It is not a global win, and the change did
not ship as a hardcoded default.** (`think` is Ollama-only; the OpenRouter
client never sends it, so those two configs are unaffected and were not re-run.)

| Model | TGC think | TGC no-think | in-budget think | in-budget no-think | calls | truncated |
|---|---|---|---|---|---|---|
| `qwen3-vl:4b` | 40.0% | **80.0%** | 0.0% | 0.0% | 46→77 | **22→4** |
| `qwen3.5:4b-mlx` | 100.0% | **86.7%** | 73.3% | **6.7%** | 57→**89** | 4→**12** |
| `lfm2.5:8b` | 6.7% | 0.0% | 6.7% | 0.0% | 43→31 | 3→0 |
| `gemma4:cloud` | 100.0% | 100.0% | 100.0% | 100.0% | 59→58 | 0→0 |
| `gemma4:e4b-mlx` | 93.3% | 93.3% | 93.3% | 93.3% | 54→50 | 1→0 |
| `gemma4:e2b-mlx` | 60.0% | 60.0% | 60.0% | 60.0% | 69→63 | 5→0 |
| `llama3.2:3b` | 20.0% | 26.7% | 20.0% | 20.0% | 70→70 | 0→0 |

**`qwen3-vl:4b` doubles.** TGC 40%→80%, `thinking_budget_exhausted` eliminated
entirely, truncated calls 22→4. The prediction held exactly.

**`qwen3.5:4b-mlx` is damaged, via a mechanism worth naming.** Its in-budget
score collapses 73.3%→6.7% and its model-call count *rises* 57→89 with
truncations tripling 4→12. Disabling thinking did not make it faster — it made
it produce worse-formed JSON, which sent `request_structured` into its retry
loop. Its total eval time went 6.2→20.3 minutes. For this model the reasoning
channel was not competing with the answer; it was how the answer got well
formed on the first attempt.

**`llama3.2:3b`'s 20.0%→26.7% is not evidence of anything** — its n=45 baseline
is 24.4%, and `think` never changes its behaviour (thinking share 0.00, call
count identical at 70). Its eval-time swing 3.5→13.3 min with identical call
counts is host variance, a reminder that wall-clock on this setup is noisy even
when behaviour is not.

**Shipped instead:** `Tier1Reasoner(think=...)`, tri-state, wired to
`MODEL_SMART_THINK`. Default `None` — leave each model's own behaviour alone —
because the measurement shows no setting is right for every model. An operator
running `qwen3-vl:4b` sets it false and roughly doubles their success rate; one
running `qwen3.5:4b-mlx` must not.

The generalisable finding: **thinking mode is a per-model property, not a tier
property.** The Tier-0 fix (`2537913`) disabled it for classification and was
right to; extending the same reasoning to Tier-1 by argument alone would have
broken the best-performing local model in the suite. The argument was sound and
the outcome was still model-specific — which is the case for measuring rather
than reasoning about a change, one more time.

---

### Task 5: Re-run and re-interpret

- [x] **Step 1: Re-run all 9 configs with instrumentation on**
- [x] **Step 2: Re-classify every historical outlier against the taxonomy**

Specifically resolve: does `qwen3-vl:4b` score above 0% once F1 and F4 are removed from the measurement? If yes, its line in the published article is a harness artifact, exactly as `qwen3.5:4b-mlx`'s turned out to be.

- [x] **Step 3: Produce the thinking-mode decision input**

With `thinking_token_share` per model, decide whether to disable thinking on the SMART tier — as a separate, spec'd change with its own before/after.

- [x] **Step 4: Correct the published article's affected claims**

At minimum `qwen3.5:4b-mlx` (already known to be a classifier-wiring artifact) and whatever Step 2 resolves for `qwen3-vl:4b`.

---

## What this plan deliberately does not conclude

`llama3.2:3b` (26.7%, unchanged) and `lfm2.5:8b` (0% by choice, F5) show no evidence of being harness artifacts — their throughput is healthy and they do not time out. Those look like genuine model behaviour and should be left alone until the instrumentation says otherwise.
