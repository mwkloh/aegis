# Classification Fallback to the Multi-Step Planner — Design Spec

## Context and motivation

A live diagnostic investigation this session (prompted by `eval/tasks/list_then_read.yaml` failing across every model tested — Gemma 2B, Qwen 3.5 9B, Llama 3.1 8B — while passing at Gemma 4B and 32B, a pattern that didn't track parameter count the way every other result this session did) traced the failure to `HarnessDispatcher.dispatch()`'s intent-classification gate, not the multi-step planner.

`dispatch()` (`runtime/chat/telegram/harness_dispatcher.py:450-465`) classifies `user_text` before the planner ever sees it. When `ModelBackedClassifier.classify()` returns `intent="unknown"` (confidence 0.0), `registry.for_intent("unknown")` returns `None`, and `dispatch()` returns `DispatchOutcome.PASS` immediately — zero tool calls, no reply, and the multi-step planner is never invoked at all.

Direct probes confirmed the classifier genuinely fails on this phrasing shape: "list files in {sandbox} and then read report.txt" classifies as `unknown`/0.0, while the structurally similar "find files about CT-001 in {sandbox}/notes and read the first one" classifies correctly as `find_files`/0.95. But a separate direct probe of `Tier1Reasoner.plan_next` with the *same* failing phrasing, given the full skill catalog directly (bypassing classification entirely), returned the correct plan (`tool_call`, `files_list`, `remaining=["files_list","files_read"]`). The planner can handle this request. The classifier's gate never lets it try.

A related, now-fixed bug (commit f1c8246, same session) compounded this: `build_harness_dispatcher` wired the classifier to `cfg.models.smart_local` — the SMART-tier planning model — instead of `cfg.models.fast`, `ModelConfig`'s own documented Tier-0-classifier field. That coupling meant every cross-family eval comparison this session never actually held classification constant the way the comparison assumed to. That fix is out of scope here; it's already merged.

This spec covers the second, larger fix: what `dispatch()` does when classification fails, so a real category of request (compound "do X then Y" phrasing) stops being silently, completely unhandled.

## Scope

**In scope:** `dispatch()`'s behavior on `descriptor is None` — the case where `registry.for_intent(classification.intent)` finds no matching skill. Fallback to the multi-step planner with the full skill catalog, when `self._multi_step` is enabled.

**Explicitly out of scope:**
- Improving the classifier's own accuracy (prompt or schema changes to `ModelBackedClassifier`/`runtime/intent/classifier.py`). Rejected as the primary fix during brainstorming — this session already confirmed twice that prompt-only fixes for small-model gaps don't reliably work, where a structural fix (this one) does.
- The low-confidence-but-classified case (`confidence < HARNESS_CONFIDENCE_THRESHOLD` with a real `descriptor`). That path already has defined, reasonable behavior (`DispatchOutcome.CLARIFY`) and is untouched by this spec.
- The single-shot (`self._multi_step is False`) dispatch path. `runner.build(descriptor, user_text)` requires a real, specific descriptor to reason about (a concrete `tool`/`args_schema`); there is no sensible fallback for it. When multi-step is disabled, `descriptor is None` keeps today's `PASS` behavior unchanged.

## Design

### Fallback mechanism

`_run_multi_step`'s current signature takes a `SkillDescriptor`, but the descriptor does not scope which tools the planner sees — `available = list(self._registry.all())` already pulls the full registry unconditionally. The descriptor is used only as a label: `ToolIntent.skill_id` and log `extra` fields. No signature change is needed to `_run_multi_step` itself.

**Control-flow placement matters and is not incidental.** `dispatch()` runs two checks after resolving a descriptor that only make sense for a real, classified skill: `self._harness.has_tool(descriptor.tool)` (`harness_dispatcher.py:467`) and the confidence threshold (`harness_dispatcher.py:473`, routing to `CLARIFY`). A synthetic fallback descriptor has no real tool to check availability for and no classification confidence to be uncertain about — routing it through either check would either false-negative on `has_tool` (returning `PASS`, silently defeating the fallback) or misrepresent an unclassified request as a low-confidence classified one.

The fix gates those two checks behind an `is_fallback` flag, set only when `descriptor is None`, so the fallback path skips straight past them to the existing `if self._multi_step:` block (`harness_dispatcher.py:484` onward) — reusing its guarded_intent / completion_summary / history / chain-synthesis handling exactly as-is, not duplicating it:

```python
descriptor = self._registry.for_intent(intent)
is_fallback = False
if descriptor is None:
    logger.info("harness_dispatcher.no_descriptor", extra={"intent": intent})
    # The fallback is for classification MISSES ("unknown") only -- a
    # confident classification for a real intent that simply has no
    # registered skill (a catalog/config mismatch, not a miss) must still
    # PASS, not get swept into the fallback planner too.
    if not self._multi_step or intent != "unknown":
        return DispatchOutcome.PASS
    descriptor = _UNCLASSIFIED_DESCRIPTOR
    is_fallback = True
    logger.info(
        "harness_dispatcher.classification_fallback_start",
        extra={"chat_id": chat_id, "intent": intent},
    )

if not is_fallback:
    if not self._harness.has_tool(descriptor.tool):
        logger.info("harness_dispatcher.no_tool", extra={"tool": descriptor.tool})
        return DispatchOutcome.PASS
    if confidence < HARNESS_CONFIDENCE_THRESHOLD:
        question = _clarify_question(descriptor)
        await _send(question)
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", question)
        return DispatchOutcome.CLARIFY

recent = self._recent_turns(chat_id)
if self._multi_step:
    chain = await self._run_multi_step(
        descriptor=descriptor,
        user_text=user_text,
        recent=recent,
        turn_id=turn_id,
        guard_min_step=1 if is_fallback else _GUARD_MIN_STEP,
    )
    # everything from here down is the EXISTING guarded_intent /
    # completion_summary / history / chain-synthesis block, unchanged --
    # is_fallback being True only ever reached this point via the
    # multi_step branch, so the else (single-shot runner.build) branch
    # below it is never reached with a synthetic descriptor.
    ...
```

with the module-level sentinel:

```python
_UNCLASSIFIED_DESCRIPTOR = SkillDescriptor(
    id="unclassified_fallback",
    description="Classification miss — multi-step planner chose from the full catalog.",
    intents=[],
    tool="_unclassified",  # sentinel label only -- never checked against
                            # has_tool() or passed to harness.execute(); the
                            # planner's own plan.tool is what actually runs.
    args_schema={},
    requires_tier1=True,
)
```

`tool` cannot be an empty string — `SkillDescriptor.tool` requires `min_length=1` (`runtime/skills/registry.py:86`) — so it's a non-empty sentinel string instead, valid only because this descriptor's `tool` field is never read by anything except `_run_multi_step`'s label-only usages (`descriptor.id` in `ToolIntent.skill_id` and logging). It exists purely so those existing references have a clear, distinguishable label rather than requiring `descriptor: SkillDescriptor | None` throughout `_run_multi_step` and every caller.

When `self._multi_step` is `False`, the existing `PASS` behavior is unchanged — there is no descriptor-free path through the single-shot reasoner.

### Safety: destructive-guard tightening for this path only

Normal multi-step dispatch (`_GUARD_MIN_STEP = 2`, `harness_dispatcher.py:82`) allows a destructive tool at step 1, because classification already narrowed the request to a specific skill matching stated intent — a real trust signal. The fallback path has none: the planner is choosing from the full catalog with no external validation that this is even the right category of request.

`_run_multi_step` gains an optional parameter:

```python
async def _run_multi_step(
    self,
    *,
    descriptor: SkillDescriptor,
    user_text: str,
    recent: tuple[tuple[str, str], ...],
    turn_id: str,
    guard_min_step: int = _GUARD_MIN_STEP,
) -> _ChainResult:
```

The destructive-guard check (`harness_dispatcher.py:734`) changes from `step_no >= _GUARD_MIN_STEP` to `step_no >= guard_min_step`. Every existing call site is unaffected (the default preserves current behavior exactly). The new fallback call site in `dispatch()` passes `guard_min_step=1` — a destructive tool is guarded at every step, including the first, when there was no classification signal at all.

Cost: a genuinely destructive first action cannot complete in one step through the fallback path, even if it's exactly what the user wanted. They see a confirmation prompt on step 2 instead. Given zero classification signal, that's the correct default — not overcaution.

### Safety: the fallback's trigger condition is broader than compound requests

`dispatch()` gates the fallback on `descriptor is None and intent == "unknown"` (the `intent != "unknown"` half of that guard was added during merge-blocking review to keep a confidently-classified-but-unregistered intent from being swept into the fallback too — see Testing, `test_classified_but_unregistered_intent_still_returns_pass_in_multi_step`). In production this guard is a no-op: `ModelBackedClassifier`'s own JSON schema (`_build_schema` in `runtime/intent/classifier.py`) constrains the model's `intent` output to `known_intents ∪ {"unknown"}`, so `for_intent(intent) is None` and `intent == "unknown"` are structurally equivalent whenever the real classifier is in the loop — the guard only matters for classifiers (or test doubles) that don't respect that invariant. That equivalence is exactly why the trigger condition is still broader than the compound-request case that motivates this branch:

1. **Every unclassifiable message reaches the fallback, not just compound requests.** Chit-chat, greetings, typos, and any other input the classifier can't place all resolve to `intent="unknown"` — the same value a genuinely-compound "do X then Y" request produces on a classification miss. `dispatch()` has no way to tell "this needs the planner" from "this needs nothing at all" using `intent`/`confidence` alone, so a plain conversational message now pays one extra SMART-model planner round-trip before falling through to chat. This is bounded, not open-ended: if the planner's first move is `respond` with empty history — the expected outcome when there's nothing to plan — `dispatch()`'s existing PASS-on-empty-history handling still returns `PASS`, and the conversation is preserved exactly as before this branch. The cost is one extra round-trip's latency, not a functional regression.

2. **A classifier failure now fails open instead of failing safe.** `ModelBackedClassifier.classify()` returns `IntentClassification(intent="unknown", confidence=0.0)` for two situations that are indistinguishable from the return value alone: a genuine "I can't classify this," and an internal transport/schema failure it catches and converts into a value rather than raising (`runtime/intent/classifier.py`'s `except (httpx.HTTPError, ValueError, RuntimeError)` branch and the two schema/validation-failure branches in `_ask_model`, all three collapsing to the same `unknown`/`0.0` result). Both look identical to `dispatch()`. Before this branch, either case meant "touch nothing" — safe regardless of which one happened. After this branch, both reach the full multi-step planner with the entire catalog and zero classification signal: a genuine classifier *outage* now fails open (routes everything to the fallback planner) instead of failing safe (routing everything to `PASS`, as it did before this branch).

3. **Fixing (2) is a real follow-up, not something this branch treats as acceptable.** Distinguishing "genuinely unclassifiable" from "the classifier itself failed" requires changing `ModelBackedClassifier`/`runtime/intent/classifier.py`'s own return contract — e.g. raising on transport/schema failure instead of degrading to a value, or returning a distinct sentinel for infra failures so `dispatch()` can tell the two apart. That interface change is explicitly out of scope for this branch (see Scope, above — ruled out during brainstorming). This section exists so that ruling is a stated, deliberate tradeoff instead of an undocumented side effect discovered later.

### Observability

`dispatch()` logs a distinct event when the fallback triggers, before calling `_run_multi_step` — shown already in the "Fallback mechanism" code above (`harness_dispatcher.classification_fallback_start`, carrying `chat_id` and `intent`). Deliberately excludes raw `user_text` from the log payload, matching this codebase's existing practice elsewhere in this file (tool args are logged, but the evidence ledger stores only `argv_hash`, never real argument values — see `runtime/tools/record.py`). This is a new, previously-unreachable code path; being able to find every turn that took it in production logs is worth the one extra log line, but logging raw user text is a precedent this spec should not set.

`available_skills` is deliberately omitted from this log line (added during merge-blocking review): `_run_multi_step`'s own `multi_step_start` log, fired immediately next in the same turn, already reports it from `available`, which that method has to materialize anyway to run the loop. Computing `len(self._registry.all())` a second time here would only build a throwaway list for this one field.

## Testing

**Unit** (`tests/test_harness_dispatcher.py`, extending existing stub-classifier/stub-planner patterns):
- A stub classifier returning `intent="unknown"` with `multi_step=True` reaches `_run_multi_step` (a stub-planner call happens) instead of returning `PASS` directly.
- The same scenario with `multi_step=False` (or its default) still returns `PASS` — confirms the single-shot path is untouched.
- A tool call recorded via the fallback path carries `skill_id="unclassified_fallback"` in its `ToolIntent`.
- A destructive tool chosen as the planner's very first step through the fallback path trips the guard (new behavior — `guard_min_step=1`).
- A destructive tool chosen as the very first step through *normal*, classified multi-step dispatch (existing call site, no `guard_min_step` override) still does **not** trip the guard at step 1 — a regression guard confirming today's behavior at every unmodified call site is unchanged.
- A confidently classified intent (`confidence=0.95`, a real intent string, not `"unknown"`) with no matching skill still returns `PASS` under `multi_step=True` — confirms the fallback triggers on classification *misses* specifically, not on every `descriptor is None` case (`test_classified_but_unregistered_intent_still_returns_pass_in_multi_step`, added during merge-blocking review alongside the `intent != "unknown"` guard in "Safety: the fallback's trigger condition is broader than compound requests").

**Live validation:** re-run `runtime/eval/cli.py` against `gemma4:e2b-mlx`. Success: `list_then_read`'s `actual_calls` show a real `files_list` → `files_read` chain (whether or not the task's specific grading passes — this fix's job is reaching the planner, not guaranteeing task completion), and `search_then_read` and every currently-passing task show no regression.

## Non-goals

- Classifier prompt/schema improvements — rejected, see Scope.
- Touching the `CLARIFY` path or its confidence threshold.
- Touching the single-shot dispatch path.
- Persisting or exposing `_UNCLASSIFIED_DESCRIPTOR` outside `dispatch()`/`_run_multi_step` — it is a call-site-local labeling construct, not a new registered skill.
