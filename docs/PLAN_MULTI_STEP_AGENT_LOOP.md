# Plan — Multi-Step Agent Loop in HarnessDispatcher

Track this as a follow-up to the production-hardening branch work. Today's
HarnessDispatcher is single-shot: classify → one tool → synthesise. That can't
satisfy operator requests like *"find X then open it"* or *"list these and
delete the duplicates"* — the synthesiser was even hallucinating follow-up
actions until commit `deed46e` tightened `tool_synthesis.txt` to forbid that.
The clean fix is a bounded multi-step loop.

## Goal

Eva can chain tool calls inside one Telegram turn, with auditable history and
an honest synthesis that only reports actions that actually ran.

## Architecture change

`HarnessDispatcher.dispatch()` becomes a bounded loop:

```
classify
loop step in 1..MAX_STEPS:
    plan = await runner.plan_next(history, last_result)
    if plan.kind == "respond":       → break
    if plan.kind == "tool_call":     → execute, append (call, result) to history
    if step == MAX_STEPS - 1 and not respond: → force respond on next step
synthesize(history)  # one synthesis at the end, sees the full chain
```

`history` is `list[(ToolIntent, ToolResult)]` carried through the loop and
threaded into both planner and synthesiser prompts.

## Concrete modules to touch

| File | Change |
|---|---|
| `runtime/chat/telegram/harness_dispatcher.py` | `dispatch()` becomes loop. New `_MAX_TOOL_STEPS = 5`. |
| `runtime/reasoning/skill_runner.py` | Add `plan_next(...)` that builds the next step or signals done. Existing `build()` stays for callers that want one-shot. |
| `runtime/reasoning/tier1_reasoner.py` | New code path that takes prior `(call, result)` pairs and returns either `{kind: "tool_call", tool, args}` or `{kind: "respond"}`. Schema-validated reply, same `request_structured` retry path. |
| `runtime/reasoning/prompts/tier1_planner.txt` | New prompt: shows the operator's request + history of calls + their results, asks "next tool call (with args) or are we done?". |
| `runtime/reasoning/prompts/tool_synthesis.txt` | Update to "you ran *these* tools" (plural) using the chain history. Keep the no-claim-extra-actions clause from `deed46e` — the **set** of tools is now what's verified. |
| `runtime/chat/reply_verdict.py` | Refactor `annotate_unverified_claim` to accept `verified_tools: set[str]` rather than `int`. Match each claim phrase against the action it implies; flag if the verb maps to a tool not in `verified_tools`. |
| `runtime/chat/telegram/bot.py` | Typing-indicator task already keeps firing through awaits — should keep working. Optionally surface a progress edit between steps ("step 1/3: searching…") for long chains. |

## Decisions to make before coding

1. **Step cap — hard or soft?** 5 is a sensible default. Soft (warn at 5,
   abort at 10) gives wiggle room but complicates audit. Recommend hard cap = 5
   with a settable `cfg.harness.max_steps`.
2. **One synthesis or per-step?** One at the end is cheaper and gives a
   coherent narrative; per-step gives the operator visibility but multiplies
   cost and risks contradictions. Recommend one synthesis at the end; emit a
   "running step N…" typing-action between steps for UX.
3. **Tier-1 model for planning vs synthesis.** Currently both are
   `cfg.models.smart` (Grok-4.1-fast). For planning a small fast model is
   fine; for synthesis you want the same. Keep a single model unless cost
   data suggests splitting.
4. **Loop termination conditions.** Three terminations: `kind == "respond"`,
   step cap reached, or tool error with `policy="abort"`. Per-tool
   retry-on-error policy is out of scope for v1.
5. **Idempotency / safety on destructive tools.** Any destructive tool
   (`files_delete`, `files_move`, `files_write`) inside a chain — do we need
   explicit operator confirmation per destructive step? Recommend yes: the
   chain auto-aborts if a destructive intent appears beyond step 1, and the
   synthesiser is told "the operator must confirm the destructive step." This
   is a reasonable belt for a personal-trust system.

## Test plan (rough)

- **Happy chain**: search → open. Both tools fire, synthesis sees both
  results, no hallucination flagged.
- **Single-step still works**: list_files → respond. No regression in the
  existing test suite.
- **Step cap**: a planner that always returns `tool_call` hits the cap, gets
  force-respond, no infinite loop.
- **Mid-chain error**: search returns ok, open raises; chain aborts,
  synthesis explains the partial result.
- **Destructive guard**: search → delete; second step is intercepted,
  operator asked to confirm.
- **Verdict gate**: synthesis claims "moved" when only "search" + "open"
  ran; gate flags it.

## Cost & latency back-of-envelope

Today: 1× classifier (Ollama, ~1 s warm) + 1× planner (OpenRouter, ~1 s) + 1×
synthesis (OpenRouter, ~2 s) ≈ 4–5 s per turn.

After: same baseline + per-extra-step ~1 s OpenRouter planner each. A 3-step
chain ≈ 7 s. Acceptable for Telegram. Token cost: synthesis prompt grows
linearly in chain length, but `max_tokens=2048` (commit `f12d511`) already
covers ~5 steps' worth of output comfortably.

## Sequencing

1. **Foundations (no behaviour change)** — refactor `dispatch()` to call
   `plan_next` once, behind a feature flag `cfg.harness.multi_step=False`
   (defaults off). Land on main, all existing tests still pass.
2. **Loop body** — implement the loop, step cap, history threading, with
   tests for the happy chain.
3. **Verdict gate refactor** — per-tool match, backwards-compatible for the
   count-based call site in `pipeline.py`.
4. **Destructive guard** — explicit list of destructive tools, mid-chain
   interceptor + confirm flow.
5. **Flip the flag** after a smoke session in tmux. Ship.

## Out of scope for v1

- Parallel tool calls (only sequential).
- Cross-turn memory of partial chains (each turn is its own atomic chain).
- Model-driven retry on tool error (planner can request retry on the next
  step if it wants, but no automatic policy).
- Streaming progress updates to the Telegram message (only typing
  indicator and optional between-step status messages).
