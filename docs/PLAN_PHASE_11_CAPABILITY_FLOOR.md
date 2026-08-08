# Phase 11 — Capability floor + evidence ledger

> Status: Proposed
> Predecessors: `PLAN_MULTI_STEP_AGENT_LOOP.md` (multi-step loop + destructive
> guard ✅), `PLAN_PHASE_8_LOCAL_AND_SKILLS.md` (structured output helper ✅),
> `docs/superpowers/specs/2026-04-22-macos-files-design.md` (FilesClient,
> read-only tools ✅)

## Goal

Lift AEGIS from "chatbot with bounded skills" to "small agent with verified
actions" — without loosening any reliability guarantee. Four tracks:

1. **Track A — Schema-constrained decoding + JSON repair.** Push JSON
   validity enforcement from post-hoc validation into the decoder itself
   (Ollama structured outputs), and salvage malformed output deterministically
   before burning a retry. This is what makes 2B-class models viable.
2. **Track B — Evidence ledger.** Wire the existing-but-orphaned audit trail
   (`runtime/tools/record.py`) into `HarnessDispatcher`, so every tool call in
   a chain leaves a classified, queryable proof record.
3. **Track C — Explicit completion (`task_complete`) + completion gate.**
   Make "I'm done" a structural act the harness can intercept, and gate the
   claimed summary against the evidence ledger — the direct counter to models
   lying about completion.
4. **Track D — Guarded write + argv command runner.** `files_write` behind a
   working destructive-confirmation flow (today's flow forgets the pending
   intent), plus a `run_command` skill: argv-only, allowlisted binaries, no
   shell — the AEGIS-native answer to "give the agent a terminal."

## Non-goals

- MCP client, delegation/subagents, trajectory memory, verdict-gated model
  escalation — Tier 3/4 of the roadmap; separate phases.
- Free-form shell execution. `run_command` is argv-only against an operator
  allowlist. Shell metacharacters are data, never syntax (design pin).
- Snapshot/rollback before mutations. Follow-up once `files_write` has soaked.
- OpenRouter structured outputs. Track A targets `OllamaClient` only;
  `OpenRouterClient` keeps the `"json"` string fallback (open question #1).
- Declarative postconditions in `skill.yaml`. Next reliability phase; the
  ledger this phase builds is its substrate.

## Design

### Track A — Schema-constrained decoding + JSON repair

#### A1 — `ChatRequest.response_schema`

`runtime/llm/clients/base.py` — one new optional field on the frozen model
(additive, no callers break):

```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    response_format: Literal["text", "json"] = "text"
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON schema for decoder-enforced structured output. When set, "
            "takes precedence over response_format on clients that support "
            "it (Ollama). Clients that do not support it fall back to "
            "response_format."
        ),
    )
```

#### A2 — `OllamaClient` schema passthrough

`runtime/llm/clients/ollama_client.py:55-65` — the payload branch becomes:

```python
if request.response_schema is not None:
    payload["format"] = request.response_schema
elif request.response_format == "json":
    payload["format"] = "json"
```

Ollama ≥0.5 accepts a JSON schema object in `format` and constrains sampling
to it (GBNF under the hood). Invariant: **client-side validation stays** —
`structured_output._validate` remains the source of truth. Decoder constraint
is an optimization, not a trust boundary; a proxy or an old Ollama silently
ignoring `format` must not weaken anything.

Schema shapes must stay GBNF-friendly (lesson from hermes-agent's
`schema_sanitizer.py`): no bare `{"type": "object"}` without `properties`, no
`$ref`, no `anyOf` at the top level. Both existing builders
(`tier1_reasoner.py:254,283`) already comply — pin this with a test (A-t4).

#### A3 — `request_structured` threads the schema

`runtime/llm/structured_output.py:118` — `_one_call` builds:

```python
request = ChatRequest(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    response_format="json",
    response_schema=schema,
)
```

No signature change to `request_structured` — callers (`Tier1Reasoner`,
`ModelBackedClassifier`) get decoder enforcement for free.

#### A4 — Deterministic JSON repair

New function in `runtime/llm/structured_output.py`, called from `_validate`
before declaring `invalid_json`:

```python
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def repair_json(content: str) -> str | None:
    """Deterministic salvage of near-miss JSON. Returns a parseable string
    or None. Never guesses at semantics — only removes wrapper noise:

    1. markdown fences (```json ... ```)
    2. prose before the first '{' / after the last '}'
    3. trailing commas before '}' or ']'
    """
    candidate = content.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = candidate[start : end + 1]
    candidate = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate
```

`_validate` change: on `json.JSONDecodeError`, try `repair_json`; on success
re-validate the repaired string and record the salvage. `StructuredOutcome`
grows one field (frozen dataclass, default keeps old call sites valid):

```python
@dataclass(frozen=True)
class StructuredOutcome:
    attempts: int
    escalated: bool
    error_kind: ErrorKind
    final_model: str
    repaired: bool = False
```

New event type in `runtime/events/stream.py` `EventType`:

```python
LLM_JSON_REPAIRED = "llm.json_repaired"
```

Emitted with payload `{"call_site": call_site, "model": model}` — structural
only, no content bodies (design pin).

### Track B — Evidence ledger

The storage layer already exists: `record_tool_call` / `load_tool_calls` /
`ToolCallRecord` in `runtime/tools/record.py`, persisting as
`EventType.TOOL_INVOKED` events on the session shard. What's missing is the
producer: `HarnessDispatcher` executes tools without recording anything and
has no `EventStream` at all.

#### B1 — Verdict for in-process results

`record.py`'s verdicts come from the subprocess `ToolVerdict`
(`runtime/tools/harness.py:38`). In-process `HarnessAdapter` results are
`status: Literal["ok", "error"]`. Extend the vocabulary by one:

```python
# runtime/tools/harness.py
ToolVerdict = Literal[
    "verified",
    "argv_rejected",
    "exit_nonzero",
    "timeout",
    "schema_violation",
    "host_denied",
    "tool_error",        # NEW: in-process tool raised or returned error status
]
```

Mirror in `record.py:_VALID_VERDICTS`. Mapping helper in `record.py`:

```python
def verdict_for_result(result: ToolResult) -> ToolVerdict:
    """Map an in-process HarnessAdapter result onto the verdict vocabulary."""
    return "verified" if result.status == "ok" else "tool_error"
```

(`ToolResult` here is `runtime.harness.contract.ToolResult`.)

#### B2 — Dispatcher records every call

`HarnessDispatcher.__init__` gains two keyword-only params:

```python
events: EventStream | None = None,
clock: Callable[[], datetime] | None = None,
```

`None` keeps every existing test and call site working. Two small helpers on
the dispatcher (used by Tracks C and D as well):

```python
def _now(self) -> datetime:
    return self._clock() if self._clock is not None else datetime.now(tz=UTC)

def _append_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
    if self._events is not None:
        self._events.append(event_type, payload)
```

In both the single-shot path and `_run_multi_step`, immediately after
`self._harness.execute(tool_intent)`:

```python
record_tool_call(
    self._events,
    imp_id=turn_id,
    skill=descriptor.id,
    tool=tool_intent.tool,
    argv_hash=compute_argv_hash(
        [tool_intent.tool, json.dumps(tool_intent.args, sort_keys=True, default=str)]
    ),
    verdict=verdict_for_result(result),
    outcome_bytes=len(
        json.dumps(result.payload, ensure_ascii=False, default=str).encode("utf-8")
    ),
)
```

(As built, the None-guard and the call above live inside a single
`_record_tool_call` helper whose whole body is wrapped in
`try/except Exception: logger.warning(...)`.)

`turn_id` is minted once per `dispatch` call:
`turn_id = f"turn-{chat_id}-{uuid.uuid4().hex[:8]}"`. It scopes the ledger
query in Track C's gate to *this turn's* evidence — the freshness property.
Serialization must be total: `args`/`payload` are `dict[str, Any]`, so both
`json.dumps` calls take `default=str` (a set/Path/datetime leaf must degrade
to a string, not raise), `outcome_bytes` counts real UTF-8 bytes, and the
helper never lets a recording failure escape into the turn — recording is
telemetry; the turn is the product. (`record_tool_call` itself is idempotent
on the composite key and skips malformed prior lines.)

Invariant carried from `record.py`: **structural only** — `argv_hash` and
`outcome_bytes`, never argv contents or stdout bodies, in the event stream.

#### B3 — Wiring

`build_harness_dispatcher` (`runtime/chat/telegram/bot.py:813-895`) accepts
and forwards `events`; the construction site passes the bot's existing
shared `EventStream` (built at `bot.py:~1165`, before the dispatcher).
The CLI path needs no wiring: `runtime/chat/cli.py` builds a `Pipeline`
around a raw `HarnessAdapter` and never constructs a `HarnessDispatcher`
(a stale assumption in an earlier draft of this plan said otherwise).

### Track C — Explicit completion + gate

#### C1 — `task_complete` plan kind

`PlanStep` (`runtime/reasoning/tier1_reasoner.py:62-73`):

```python
class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern="^(tool_call|respond|task_complete)$")
    tool: str | None = Field(default=None, max_length=64)
    args: dict[str, Any] | None = Field(default=None)
    summary: str | None = Field(default=None, max_length=2048)
```

`_build_planner_schema` (`tier1_reasoner.py:254`) adds `task_complete` to the
`kind` enum and an optional `summary` string property, keeping
`"additionalProperties": False`. `runtime/reasoning/prompts/tier1_planner.txt`
gains the rule (verbatim addition):

```
When the user's request has been satisfied by the tool calls above, emit
{"kind": "task_complete", "summary": "<one sentence stating what was done>"}.
Only state things the tool results above actually show. If you have not
called any tool yet, use "respond" instead — never "task_complete".
```

Backward compatibility pin: `kind="respond"` keeps its exact current
semantics (loop break; empty-history → `PASS`). A model that never learns
`task_complete` loses nothing — essential for 2B-class planners.

#### C2 — Completion gate in `_run_multi_step`

New branch in the loop, before the `tool_call` handling:

```python
if plan.kind == "task_complete":
    summary = plan.summary or ""
    if not history:
        break  # nothing was done; same as respond-with-no-history → PASS
    gated = self._gate_completion(summary, history, turn_id)
    return _ChainResult(history=history, completion_summary=gated)
```

`_ChainResult` grows `completion_summary: str | None = None`. The gate:

```python
def _gate_completion(
    self,
    summary: str,
    history: list[tuple[ToolIntent, ToolResult]],
    turn_id: str,
) -> str:
    """Check the claimed summary against this turn's evidence.

    Evidence = ledger records for turn_id with verdict == "verified"
    (falls back to in-memory history when no EventStream is wired).
    A summary claiming tools that have no verified record gets the
    existing unverified-claim annotation; it is never silently trusted.
    """
    if self._events is not None:
        records = [
            r for r in load_tool_calls(self._events)
            if r.imp_id == turn_id and r.verdict == "verified"
        ]
        verified = {r.tool for r in records}
    else:
        verified = {
            call.tool for call, res in history if res.status == "ok"
        }
    failed = [call.tool for call, res in history if res.status != "ok"]
    if failed:
        summary = (
            f"{summary}\n\n⚠️ Note: {', '.join(sorted(set(failed)))} did not "
            f"complete successfully — the summary above may overstate what "
            f"was done."
        )
        self._append_event(
            EventType.HARNESS_COMPLETION_GATED,
            {"turn_id": turn_id, "failed_tools": sorted(set(failed))},
        )
    return annotate_unverified_claim(summary, verified_tools=verified)
```

When a gated completion exists, `_synthesize_chain` is skipped and the gated
summary is the reply — the model's own claim, checked, rather than a second
model call that could re-hallucinate. New event type:

```python
HARNESS_COMPLETION_GATED = "harness.completion_gated"
```

Gate posture for this phase: **annotate + event, don't hard-block.** The
event stream gives measurement (how often would a block have fired, and
would it have been right?) before enforcement — same soak pattern as the
destructive guard, which shipped as log-only defence-in-depth first. The
hard-block upgrade is open question #3.

### Track D — Guarded write + argv command runner

#### D1 — Pending-confirmation state (prerequisite for everything else)

Today the guard fires, sends "please confirm with a follow-up message", and
drops the intent on the floor — the follow-up "yes" enters `dispatch` as an
unrelated turn (`harness_dispatcher.py:241-250`). Fix with in-memory,
single-operator-scoped state on the dispatcher:

```python
@dataclass(frozen=True)
class _PendingConfirmation:
    intent: ToolIntent
    skill_id: str
    created_at: datetime


_CONFIRMATION_TTL_S = 120
_AFFIRMATIVES = frozenset({"yes", "y", "confirm", "do it", "go ahead"})
```

`self._pending: dict[int, _PendingConfirmation] = {}` keyed by `chat_id`.

At the top of `dispatch`, before classification:

```python
pending = self._pending.pop(chat_id, None)
if pending is not None:
    age = (self._now() - pending.created_at).total_seconds()
    if age <= _CONFIRMATION_TTL_S and user_text.strip().lower() in _AFFIRMATIVES:
        return await self._execute_confirmed(pending, chat_id=chat_id,
                                             message=message, reply=reply)
    # anything else — declined or expired; fall through to normal dispatch
    self._append_event(EventType.HARNESS_CONFIRMATION_DECLINED,
                       {"tool": pending.intent.tool, "expired": age > _CONFIRMATION_TTL_S})
```

`_execute_confirmed` runs the stored intent through `self._harness.execute`,
records it in the ledger (Track B), synthesizes via the single-tool path, and
returns `FIRED`. The guard branch (`:373-378`) stores the pending intent
before sending the confirmation text. Three new event types:

```python
HARNESS_CONFIRMATION_REQUESTED = "harness.confirmation_requested"
HARNESS_CONFIRMATION_ACCEPTED = "harness.confirmation_accepted"
HARNESS_CONFIRMATION_DECLINED = "harness.confirmation_declined"
```

Pins: exact-match affirmatives only (no model in the confirmation loop — a
2B model must not be the judge of its own destructive action); pop-before-
check so a pending intent is consumed exactly once; TTL hard-coded, not
config (a forgotten confirmation should die, not linger).

#### D2 — `files_write` tool

`FilesClient` (`runtime/files/client.py`) gains `write_file`, mirroring the
existing root-containment checks of `read_file`:

```python
def write_file(self, path: str, content: str) -> dict[str, Any]:
    """Write text to a file inside a configured root. Parent dirs are NOT
    created — writing into a missing directory is an error, not a mkdir.
    Returns {"path": str, "bytes_written": int}."""
```

Containment invariant (same as read path): resolve + expanduser, then
require the resolved path to be within `cfg.files.roots`; reject symlink
escapes by resolving before the containment check. Size cap:
`len(content.encode()) <= 256 * 1024`, else error — a chat-driven write has
no business being bigger.

`runtime/harness/tools/files_tool.py`: add `files_write` to
`make_files_tools`, wrapping `FilesClient.write_file` with the same
error-to-`ToolResult` mapping as the read tools. New bundled skill
`runtime/skills/_bundle/write_file/skill.yaml`:

```yaml
id: write_file
version: 0.1.0
description: Write text content to a file inside an allowed folder.
intents:
  - write_file
  - save_file
tool: files_write
args_schema:
  type: object
  properties:
    path:
      type: string
      minLength: 1
      maxLength: 512
      description: >-
        Absolute path for the file (e.g. /Users/michaelloh/Desktop/notes.txt).
        '~' is allowed. Must be inside an allowed folder.
    content:
      type: string
      minLength: 0
      maxLength: 65536
      description: The exact text content to write.
  required:
    - path
    - content
requires_tier1: true
```

`files_write` is **already** in `DESTRUCTIVE_TOOLS`
(`harness_dispatcher.py:32-36`) — the guard was built anticipating this tool.
With D1 in place, the step ≥2 interception now leads somewhere: confirm →
execute → ledger record.

Deliberate gap: step 1 destructive calls execute without confirmation (the
existing guard's documented design: a directly-requested single-step write is
the operator's stated intent). Open question #4 revisits this for writes
specifically.

#### D3 — `run_command` skill (argv-only, allowlisted)

New config section in `runtime/config.py`, following the `HarnessConfig`
pattern exactly:

```python
class CommandsConfig(BaseModel):
    """Argv-only command runner (run_command tool). No shell, ever."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_binaries: tuple[str, ...] = Field(
        default=("ls", "cat", "head", "tail", "wc", "grep", "find", "file"),
        description="Binaries the model may invoke as argv[0]. Read-only "
        "inspection tools by default; operators extend deliberately.",
    )
    timeout_ms: int = Field(default=15_000, ge=100, le=120_000)
    max_output_bytes: int = Field(default=32_768, ge=1024, le=262_144)
```

Registered on `AegisConfig` as
`commands: CommandsConfig = Field(default_factory=CommandsConfig)`.

New tool `runtime/harness/tools/command_tool.py` — synchronous (matching
`HarnessAdapter.execute`'s sync contract), argv-only:

```python
def make_command_tool(cfg: CommandsConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run_command(args: dict[str, Any]) -> dict[str, Any]:
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(t, str) for t in argv
        ):
            raise ValueError("argv must be a non-empty list of strings")
        binary = Path(argv[0]).name
        if binary not in cfg.allowed_binaries:
            raise PermissionError(f"binary not allowlisted: {binary!r}")
        proc = subprocess.run(          # noqa: S603 — argv list, shell=False
            argv,
            capture_output=True,
            timeout=cfg.timeout_ms / 1000,
            shell=False,
            text=False,
        )
        stdout_tail = proc.stdout[-cfg.max_output_bytes :].decode(
            "utf-8", errors="replace"
        )
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "verdict": "verified" if proc.returncode == 0 else "exit_nonzero",
        }
    return run_command
```

`subprocess.TimeoutExpired` propagates to `HarnessAdapter.execute`'s
catch-all → `ToolResult(status="error")` → ledger verdict `tool_error`.
Bundled skill `runtime/skills/_bundle/run_command/skill.yaml`:

```yaml
id: run_command
version: 0.1.0
description: Run a read-only inspection command (ls, cat, grep, ...) as an
  argv list. No shell features — pipes, redirects and globs do not work.
intents:
  - run_command
  - shell_command
tool: run_command
args_schema:
  type: object
  properties:
    argv:
      type: array
      minItems: 1
      maxItems: 16
      items:
        type: string
        maxLength: 512
      description: >-
        Command as an argv list, e.g. ["ls", "-la", "/Users/michaelloh/Desktop"].
        The first element must be an allowed binary name.
  required:
    - argv
requires_tier1: true
```

`run_command` stays **out of** `DESTRUCTIVE_TOOLS`: the allowlist is the
containment (read-only defaults), and an operator who adds `rm` to
`allowed_binaries` has made an explicit config-reviewed choice. The
`SubprocessRunner`-based `run_tool` harness (`runtime/tools/harness.py`) is
not reused here — it is async and template-driven where this path is sync
and argv-driven; unifying them is open question #2, not a blocker.

## Test plan

New test files (all `pytestmark = pytest.mark.unit` unless noted):

- `tests/test_json_repair.py` — `repair_json`: fenced JSON, prose-wrapped
  JSON, trailing commas, each recovered; truly broken input → `None`;
  already-valid input passes through unchanged.
- `tests/test_command_tool.py` — allowlisted binary runs and returns
  `verdict="verified"`; non-allowlisted binary raises `PermissionError`;
  non-zero exit → `exit_nonzero`; output truncated to `max_output_bytes`;
  argv type violations raise.
- `tests/test_commands_config.py` — defaults load; `extra="forbid"` rejects
  unknown keys (mirrors `tests/test_files_config.py`).
- `tests/test_files_write.py` — write inside root succeeds and returns
  `bytes_written`; path outside roots rejected; symlink escape rejected;
  missing parent dir errors (no mkdir); >256 KiB content rejected.

Extensions to existing tests:

- `tests/test_structured_output.py` — `FakeClient` asserts
  `request.response_schema` is populated; malformed-but-repairable content
  succeeds with `outcome.repaired is True` and zero retries; unrepairable
  content still walks the retry/escalation path unchanged.
- `tests/test_ollama_client.py` — `response_schema` set → payload `format`
  is the schema dict; unset + `response_format="json"` → `format` is the
  string `"json"` (existing behavior pinned).
- `tests/test_harness_dispatcher.py` — with an `EventStream` wired
  (tmp-path-backed, per `aegis_sandbox` fixture): each chain step produces a
  `TOOL_INVOKED` record with the turn's `imp_id`; `task_complete` with a
  failed tool in history appends the gate warning and emits
  `HARNESS_COMPLETION_GATED`; `task_complete` with empty history behaves as
  `respond` (PASS); guard fires → pending stored →
  `"yes"` within TTL executes and records → non-affirmative clears pending
  and dispatches normally → expired pending declines; `events=None`
  keeps every existing test green (constructor default).
- `tests/test_tier1_reasoner.py` — planner schema contains `task_complete`
  in the `kind` enum and the `summary` property; `_build_schema` /
  `_build_planner_schema` contain no `$ref`/`anyOf`/bare-object (the
  GBNF-compat pin, A-t4).
- `tests/test_skill_registry.py` — both new skill YAMLs load via
  `SkillRegistry.from_directory`.

Gate for every rollout step (per repo convention and standing feedback:
implementers run the **full suite**, not just the new file):

```
.venv/bin/ruff check .
.venv/bin/mypy runtime memory scripts
.venv/bin/pytest -m "unit or not e2e" tests/
```

## Rollout sequence

Each step ships independently and leaves the system working:

1. **A1 + A2 + A3 + tests** — schema passthrough. Behavior-neutral when
   Ollama ignores `format` objects; pure upside otherwise.
2. **A4 + tests** — JSON repair. Reduces retries; changes no interfaces.
3. **B1 + B2 + B3 + tests** — ledger wiring. `events=None` default means
   nothing breaks if wiring is partial; records simply don't appear.
4. **C1 + tests** — `task_complete` in schema + prompt. Planner may start
   emitting it; until C2 lands the loop treats unknown kinds as `respond`
   via the existing `plan.kind != "tool_call"` break — safe.
5. **C2 + tests** — completion gate (annotate + event, no hard block).
6. **D1 + tests** — confirmation state. Fixes an existing dead end; useful
   even before any destructive tool exists (guards against planner
   hallucinating `files_delete`).
7. **D2 + tests** — `files_write` + skill YAML. First real destructive tool;
   lands on a working confirmation flow.
8. **D3 + tests** — `run_command` + config + skill YAML.
9. **Live smoke** — Telegram bot (tmux session `aegis`, not launchd — TCC
   pin): write a file on Desktop via chat with confirmation; run
   `ls` via run_command; verify ledger records with
   `load_tool_calls`; verify a `task_complete` turn annotates honestly when
   a tool fails.

## Design pins (carried + new)

- Never raise out of `dispatch` — every new branch returns an outcome.
- Argv-only, `shell=False`, everywhere. Shell strings are data.
- Structural events only: hashes and byte counts in the ledger, never argv
  contents, stdout bodies, or file contents.
- Client-side schema validation is the trust boundary; decoder constraint is
  an optimization.
- `respond` semantics unchanged — small models that never learn
  `task_complete` keep working.
- No model in the confirmation loop: exact-match affirmatives only.
- `events=None` / new fields defaulted — every addition is opt-in at the
  constructor/model layer; existing tests must pass unmodified through step 2
  of each track.

## Open questions

1. **OpenRouter structured outputs.** `OpenRouterClient` could map
   `response_schema` to OpenRouter's `response_format: {type: "json_schema"}`.
   Deferred: local models are the point of this phase.
2. **Two subprocess paths.** `command_tool.py` (sync, argv-driven) vs
   `runtime/tools/harness.run_tool` (async, template-driven, currently
   uncalled). Unify after D3 soaks — likely by making `run_tool` the engine
   and `command_tool` a thin sync shim, or by retiring `run_tool` if the
   template path stays dead.
3. **Gate hard-block.** When `HARNESS_COMPLETION_GATED` events show the
   annotate-only gate would have fired correctly (say, ≥95% precision over a
   few weeks of events), upgrade to: failed-tool completions replace the
   summary with a deterministic honest report instead of annotating it.
4. **Step-1 destructive writes.** The guard deliberately allows a step-1
   `files_write` unconfirmed. For file writes specifically, always-confirm
   may be the better posture — decide after watching
   `HARNESS_CONFIRMATION_*` events.
5. **Ledger retention.** Session shards are append-only JSONL with no
   retention policy. Fine at single-operator volume; revisit if
   `load_tool_calls` scans get slow.
