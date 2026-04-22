# Telegram Harness Dispatcher Design

## Goal

Wire Eva's existing `HarnessAdapter` (with file tools) into the Telegram free-form chat path so she can autonomously call tools during conversation, using real data in her replies instead of hallucinating.

## Architecture

`route_chat` currently has three layers (in order):

```
1. Slash command handlers         (/board, /files, etc.)
2. IntentRouter short-circuit     (morning_brief, vault_reindex — subprocess)
3. ChatPipeline fallthrough       (pure conversational LLM — no tools)
```

This feature adds a fourth layer between 2 and 3:

```
1. Slash command handlers         (unchanged)
2. IntentRouter short-circuit     (unchanged)
3. HarnessDispatcher              ← NEW
4. ChatPipeline fallthrough       (unchanged)
```

`HarnessDispatcher` is injected as an optional `harness_dispatcher: HarnessDispatcher | None` parameter into `route_chat`. When `None`, the layer is transparent and layer 4 fires as before. `ChatPipeline` is not touched.

---

## Dispatch Outcomes

`HarnessDispatcher.dispatch()` returns one of three outcomes:

| Outcome | Who replies | What's sent |
|---|---|---|
| `FIRED` | Dispatcher | Synthesized reply using real tool data |
| `CLARIFY` | Dispatcher | Clarifying question ("Which folder should I list?") |
| `PASS` | ChatPipeline | Normal conversational reply (status quo) |

Dispatch table:

```
intent → harness tool, confidence >= 0.7  → FIRED   (tool + LLM synthesis)
intent → harness tool, confidence < 0.7   → CLARIFY  (ask for missing info)
intent → unknown / ask_question / non-tool → PASS    (ChatPipeline unchanged)
```

Only intents that map to a callable registered in `HarnessAdapter` are considered "harness tool" intents. `ask_question`, `echo`, `time_query` (non-file tools) map to the `respond` / `time` harness callables — these are also dispatchable once registered. The classifier sees all catalog intents; the dispatcher filters by what's in the harness.

---

## Component Design

### `HarnessDispatcher`

**File:** `runtime/chat/telegram/harness_dispatcher.py`

**Dependencies (all injected at construction):**

| Dependency | Type | Purpose |
|---|---|---|
| `classifier` | `ModelBackedClassifier` | Async Ollama-backed intent classification |
| `registry` | `SkillRegistry` | Look up `SkillDescriptor` for classified intent |
| `runner` | `SkillRunner` | Extract args via `Tier1Reasoner` (OpenRouter) |
| `harness` | `HarnessAdapter` | Execute the tool in-process |
| `synthesizer` | `ModelClient` | OpenRouter smart model for natural reply |
| `tier3` | `Tier3Store` | Append user + bot turns after FIRED/CLARIFY |
| `tier1_loader` | `Tier1Loader` | Eva's identity/persona for the synthesis prompt |

**Public interface:**

```python
class DispatchOutcome(enum.Enum):
    FIRED = "fired"
    CLARIFY = "clarify"
    PASS = "pass"

class HarnessDispatcher:
    async def dispatch(
        self,
        *,
        chat_id: int,
        user_text: str,
        message: _Replyable,
    ) -> DispatchOutcome: ...
```

**Dispatch logic (pseudocode):**

```python
async def dispatch(self, *, chat_id, user_text, message):
    try:
        classification = await self._classifier.classify(user_text)
    except Exception:
        logger.exception("harness_dispatcher.classify_failed")
        return DispatchOutcome.PASS   # never break chat

    intent = classification.intent
    confidence = classification.confidence

    # Resolve to a SkillDescriptor
    descriptor = self._registry.for_intent(intent)
    if descriptor is None:
        return DispatchOutcome.PASS   # non-tool intent → ChatPipeline

    # Check tool is actually registered in the harness
    if not self._harness.has_tool(descriptor.tool):
        return DispatchOutcome.PASS   # tool exists in catalog but not harness

    if confidence < HARNESS_CONFIDENCE_THRESHOLD:
        question = _clarify_question(descriptor)
        await message.reply_text(question)
        self._tier3.append(str(chat_id), "user", user_text)
        self._tier3.append(str(chat_id), "bot", question)
        return DispatchOutcome.CLARIFY

    # High confidence → extract args, execute, synthesize
    tool_intent = await self._runner.build(descriptor, user_text)

    # SkillRunner degrades to tool=respond when arg extraction fails
    if tool_intent.tool == "respond":
        return DispatchOutcome.PASS   # let ChatPipeline handle it naturally

    result = self._harness.execute(tool_intent)
    reply = await self._synthesize(user_text, tool_intent, result)
    await message.reply_text(_clip(reply))
    self._tier3.append(str(chat_id), "user", user_text)
    self._tier3.append(str(chat_id), "bot", reply)
    return DispatchOutcome.FIRED
```

**Constants:**

```python
HARNESS_CONFIDENCE_THRESHOLD = 0.7   # tunable, no config entry needed yet
_MAX_REPLY_CHARS = 3500              # matches Telegram's safe message limit
```

### Synthesis Prompt

**File:** `runtime/reasoning/prompts/tool_synthesis.txt`

```
You are {identity}. The operator asked: "{user_text}"

You called the {tool} tool and got this result:

{tool_result}

Write a concise, natural reply using only the data above.
Do not fabricate file names, paths, or any data not present in the tool result.
If the result is empty or indicates an error, say so plainly.
```

`{identity}` is filled from the tier1 snapshot so Eva keeps her configured persona. If tier1 has no identity, the slot is filled with `"AEGIS, an operator-facing assistant"`.

The synthesis call uses `model=cfg.models.smart`, `temperature=0.2`, `max_tokens=512`.

### Clarifying Question Generation

`_clarify_question(descriptor)` produces a short question from the skill's description and `args_schema.required` fields. Examples:

- `list_files` (required: `path`) → `"Which folder should I list? (e.g. ~/Downloads)"`
- `read_file` (required: `path`) → `"Which file should I read? Please give the full path."`
- `search_files` (required: `directory`, `pattern`) → `"Which folder and pattern should I search? (e.g. ~/Downloads *.pdf)"`

These are generated deterministically from the schema — no LLM call.

---

## Wiring

### `build_harness_dispatcher(cfg, ...) -> HarnessDispatcher | None`

New factory function in `bot.py`. Returns `None` if:
- Ollama not reachable (classifier can't be constructed)
- `HarnessAdapter` has no tools beyond the builtins (`echo`, `respond`, `time`)
- Skill registry is empty

Otherwise returns a fully wired `HarnessDispatcher`. Construction never raises — failures produce `None` and a `logger.warning`.

### `route_chat` changes

Add `harness_dispatcher: HarnessDispatcher | None = None` parameter. Insert after the `IntentRouter` block (line ~441) and before the `ChatPipeline` fallthrough:

```python
if harness_dispatcher is not None and text:
    outcome = await harness_dispatcher.dispatch(
        chat_id=chat_id,
        user_text=text,
        message=message,
    )
    if outcome != DispatchOutcome.PASS:
        return
```

### `build_dispatcher` changes

Add `harness_dispatcher: HarnessDispatcher | None = None` parameter, thread through to `route_chat` partial.

### `build_application` changes

After the existing `FilesClient` instantiation block, add:

```python
harness_dispatcher = build_harness_dispatcher(cfg, ...)
```

Pass to `build_dispatcher(...)`.

---

## Error Handling

| Failure | Behaviour |
|---|---|
| Classifier raises | Log, return `PASS` — chat never breaks |
| Tier1 arg extraction degrades (no path found) | `SkillRunner` returns `tool=respond` → dispatcher returns `PASS` |
| `HarnessAdapter.execute()` returns `status=error` | Synthesizer receives error text; Eva replies "I tried to list files but got an error: ..." |
| Synthesis LLM call fails | Log, reply with raw tool result clipped to 3500 chars |
| `Tier3Store.append()` raises | Log, swallow — turn history gap is preferable to a crash |

---

## Testing

**File:** `tests/test_harness_dispatcher.py`

All tests use stub implementations — no real Ollama, no real OpenRouter, no filesystem.

| Test | Scenario |
|---|---|
| `test_fired_path` | Classifier returns `list_files` at 0.8 → tool executes → synthesis called → `FIRED` |
| `test_clarify_path` | Classifier returns `list_files` at 0.5 → clarifying question sent → `CLARIFY` |
| `test_pass_on_non_tool_intent` | Classifier returns `ask_question` → `PASS`, nothing sent |
| `test_pass_on_unknown_intent` | Classifier returns `"unknown"` → `PASS`, nothing sent |
| `test_pass_when_tool_not_in_harness` | Intent matches catalog but tool absent from harness → `PASS` |
| `test_pass_on_tier1_degrade` | SkillRunner returns `tool=respond` → `PASS` |
| `test_error_result_synthesized` | Harness returns `status=error` → Eva replies with soft error, still `FIRED` |
| `test_synthesis_failure_falls_back_to_raw` | Synthesizer raises → raw result clipped to 3500 chars, still `FIRED` |
| `test_classifier_exception_returns_pass` | Classifier raises → `PASS`, chat not broken |
| `test_tier3_written_on_fired` | After `FIRED`, both user and bot turns appended to `Tier3Store` |
| `test_tier3_written_on_clarify` | After `CLARIFY`, clarifying question appended to `Tier3Store` |
| `test_route_chat_dispatcher_fires_before_pipeline` | Integration: `route_chat` with dispatcher, pipeline never called on `FIRED` |

---

## Files Created / Modified

| File | Action |
|---|---|
| `runtime/chat/telegram/harness_dispatcher.py` | Create |
| `runtime/reasoning/prompts/tool_synthesis.txt` | Create |
| `runtime/chat/telegram/bot.py` | Modify — `route_chat`, `build_dispatcher`, `build_application`, `build_harness_dispatcher` |
| `tests/test_harness_dispatcher.py` | Create |

Minor addition to `HarnessAdapter`: add `has_tool(name: str) -> bool` public method (one-liner: `return name in self._tools`) so `HarnessDispatcher` can check tool registration without executing a dummy intent.

No changes to `ChatPipeline`, `SkillRunner`, `Tier1Reasoner`, or any file tools.
