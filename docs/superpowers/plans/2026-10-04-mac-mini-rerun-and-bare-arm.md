# Mac Mini benchmark rerun, and the "bare harness" comparison

Status as of 2026-10-04 (NZDT). Written *before* building the bare arm so the
reasoning that led here is not lost. Read this first if resuming.

Supersedes the Pi-era framing in `2026-08-24-eval-measurement-confounds.md` for
anything about *where* the benchmark ran. That document's measurement findings
(F1-F5) still stand; its environment notes about the Pi / Tailscale proxy are
retired -- **the benchmark is now defined as run on this Mac Mini only.**

---

## 1. What the benchmark is *for*

The article's aim is to show **how capable a small language model is when used
for agentic work -- complex, multi-tool prompts -- day to day.** Hardware
(Pi vs Mac Mini) is secondary; latency differs, capability is the point.
Decision (user, 2026-10-02): re-run everything fresh on the Mac Mini, drop the
Pi from the story entirely, and rewrite the article afterwards with the
writing-linkedin-and-medium-posts skill.

A second purpose emerged: the **harness itself** was built to (a) make small
models run multi-step prompts more reliably and (b) reduce hallucination. The
benchmark so far measures only the harness-ON arm, so it cannot say the harness
helped. See section 6.

## 2. Environment (authoritative)

- Mac Mini, Apple M4, **16 GB RAM**, macOS (Darwin 27). Ollama **0.35.0**, local
  at `127.0.0.1:11434`, no proxy. AEGIS is not running on this machine; the eval
  CLI (`python -m runtime.eval.cli`) does not need it.
- `.venv` was a Linux venv copied from the Pi; moved to `.venv.pi-linux.bak`
  (untracked) and rebuilt with `uv sync --extra dev` (Python 3.12.9).
- `~/.aegis/config.json` gained `modelProfiles` (backup:
  `~/.aegis/config.json.bak-2026-10-02`):
  `qwen3-vl:4b` think=false; `qwen3.5:4b-mlx` think=true; **every local model
  incl. the classifier `gemma4:e2b` has `num_ctx: 32768`**. `gemma4:cloud` has no
  profile (remote). `MODEL_SMART_THINK` must stay unset (it beats profiles).
- Classifier tier uses `MODEL_FAST=gemma4:e2b` (local) for *every* run, including
  the hosted OpenRouter ones. Planner model is pinned per run with
  `MODEL_SMART_PROVIDER` / `MODEL_SMART_LOCAL` (local) or `MODEL_SMART` (hosted).
- Results: `eval/results/` (gitignored) holds run 2 plus `CONFIG-SNAPSHOT.json`.
  Archives outside the repo: `~/aegis-eval-archive-pi-2026-08` (39 Pi-era files),
  `~/aegis-eval-daytime-aborted-2026-10-02` (2 files, contaminated by user's
  daytime use), `~/aegis-eval-run1-default-ctx-2026-10-02` (9 files, run 1).

## 3. Trajectory (what we did, in order, and why)

1. **Verified the old article's claims** against raw data (seven items):
   - "Gemma 3n" is **wrong** -- the models are Gemma 4 (`gemma4:e2b/e4b`;
     Ollama reports `family: gemma4`). The "parameter-sharing architecture"
     sentence is a Gemma 3n fact, unverified for Gemma 4.
   - `gemma4:cloud` = 32,682,372,656 params, BF16 -> "about 32.7B at BF16" **verified**.
   - Qwen3-VL zero: 08-23 run = 15 variants, 0 calls, 91.7-97.4 s each; 23.48 s
     cold load + 3x30 s retries ~= 90.6 s **verified** (note a separate 08-21
     zero had 7.8-31.3 s durations -- different mechanism).
   - LFM2.5 "11 of 14 declines" **verified** (11 `no_tool_call` + 3
     `thinking_budget_exhausted` in the 08-24 file).
   - Thinking-off figures **verified** from raw JSON: calls 57->89, truncated
     4->12, in-budget 73.3%->6.7%; "6.2 -> 20.3 min" is *generation time*
     (sum of `eval_ms_total`), not wall-clock (18.2 -> 27.4 min).
   - Memory growth 8.5 -> 13 GB: **no logged evidence**; user recalls it as
     feedback and considers it correct. Keep, attributed to the author's
     observation. (Run 2 residency sampling saw `qwen3.5:4b-mlx` climb to
     12 GB under sustained load even with a fixed 32k window -- consistent with
     growth that is *not* the context window; mechanism unconfirmed, candidate:
     Ollama's prompt cache, whose log line shows an 8 GiB limit.)
   - Raspberry Pi / `xdg-open`: **unverifiable** (no logs). AEGIS's `open_file`
     shells out to `/usr/bin/open` (macOS), so `xdg-open` errors could only come
     from a Linux host. **Decision: drop the Pi and the `xdg-open` anecdote.**
2. **Rebuilt the environment** on the Mac Mini and smoke-tested one task.
3. **Run 1 (daytime, default context)** -- aborted at config 3 because the models
   were starving the user's machine of memory. Results discarded as timing
   evidence. Lesson: run overnight, unload models between configs.
4. **Run 1 proper (overnight, default context)** -- finished, but 2 of 9 results
   were unusable (see F1, F2). Archived.
5. **Diagnosed** both (F1/F2), fixed the harness (test-first), added `num_ctx`.
6. **Run 2 (32k context, replies captured)** -- clean; table below.
7. **Investigated `lfm2.5:8b` silence** (F5) and reframed "bare" (section 6).

## 4. Run 2 results (Mac Mini, n=45 per model, `--repeat 3`, num_ctx=32768)

| Model | TGC | SGC | in 30 s budget | median variant |
|---|---|---|---|---|
| gemma4:cloud (32.7B) | 93.3% | 83.3% | 93.3% | 3.1 s |
| qwen3.5:4b-mlx (think=true) | 91.1% | 66.7% | 57.8% | 43.9 s |
| gemma4:e4b-mlx | 88.9% | 83.3% | 88.9% | 29.7 s |
| qwen/qwen3.5-9b (hosted) | 82.2% | 66.7% | 77.8% | 14.6 s |
| gemma4:e2b-mlx | 77.8% | 66.7% | 77.8% | 15.2 s |
| meta-llama/llama-3.1-8b-instruct (hosted) | 73.3% | 50.0% | 73.3% | 10.7 s |
| qwen3-vl:4b (think=false requested) | 33.3% | 0.0% | 28.9% | 45.4 s |
| llama3.2:3b | 26.7% | 0.0% | 26.7% | 22.3 s |
| lfm2.5:8b | 0.0% | 0.0% | 0.0% | 9.7 s |

Per task (passed/total): `search_then_read` is the hard one -- best is 3/6
(cloud, qwen3.5:4b); `e2b` and `llama-3.1` score 0/6. `list_then_read`: cloud
6/6, e4b 6/6, qwen3.5:4b 6/6, e2b 2/6, llama-3.1 2/6, qwen3.5-9b 2/6. Single-step
tasks (`list_downloads`, `read_file`, `search_files`, `time_check`) are
essentially solved by every model that works at all.

**Reproducibility**: run 1 vs run 2 -- cloud 93.3/93.3, e4b 88.9/88.9,
llama-3.1 73.3/73.3, lfm 0/0, e2b 73.3/77.8, qwen3.5:4b 88.9/91.1, qwen3.5-9b
80.0/82.2. Stable to a few points at n=45.

Wall time: 2 h 40 min total (run 1 took ~11 h, almost all `qwen3-vl:4b` hanging).
Unexplained: hosted `qwen/qwen3.5-9b` took 63.9 min wall (median variant
14.6 s) -- provider latency, not investigated.

## 5. Findings (each with evidence)

- **F1 -- default context window decides whether a model fits in RAM.**
  `qwen3-vl:4b` defaults to num_ctx 262,144; Ollama sized it at **43 GB, 80%
  CPU** on a 16 GB machine, thrashed (~87 MB free) and every call hung.
  At num_ctx 8192: 4.2 GB, 100% GPU, answers in ~18 s. KV cost ~0.15 MB/token
  for that model. Measured residency at 32k, all 100% GPU: e4b-mlx 8.7 GB,
  qwen3-vl 8.0, e2b-mlx 6.4, llama3.2 6.0, lfm2.5 5.7, qwen3.5:4b 4.1 (growing
  to 12 under load), classifier e2b 1.8. 64k would put qwen3-vl near 12 GB --
  too tight with the classifier resident. **32k is the practical ceiling here.**
- **F2 -- harness failures were being scored as model failures.**
  `build_harness_dispatcher` returns None when `router.is_local_ready()` (an
  Ollama liveness probe) gets no answer -- which happens while Ollama thrashes.
  Run 1: 30/45 `llama3.2:3b` and 14/45 `qwen3-vl:4b` variants failed in ~2 s and
  were labelled `no_tool_call`, the same bucket as a genuine decline. (Probe
  starvation is the likely mechanism; not proven for llama3.2.) Fixed: new
  `FailureKind.HARNESS_UNAVAILABLE` + a console warning. Run 2: 0 of 405.
- **F3 -- prompts are small.** Ollama log, run 2: 746 llama-server requests, max
  2,829 tokens, p50 340, p95 2,649. Zero truncation. (An early "746 truncation
  lines" alarm was a grep bug matching `truncated = 0`.) Caveat: only
  llama-server-served requests appear in that log. 32k has >10x headroom for
  *these* tasks; the benchmark does **not** test long-running sessions.
- **F4 -- `qwen3-vl:4b` still fails, differently.** 30/30 failures are
  `thinking_budget_exhausted` (81 truncated calls). `think=false` is *not*
  honoured: the Ollama log shows `renderer=qwen3-vl-thinking
  parser=qwen3-vl-thinking`, and a direct call with `think=false` + schema
  returned empty content after 512 tokens of thinking. The Pi-era "think=false
  -> 80%" does not reproduce on this Ollama. Candidate follow-up: the
  `qwen3-vl:4b-instruct` tag, or a larger token budget. **Untested.**
- **F5 -- `lfm2.5:8b` is not "silent"; it decides `respond`.** Root cause traced
  (systematic-debugging, 3 hypotheses ruled out): the planner prompt lists all
  12 tools; the model returns *valid* JSON `{"remaining": [], "kind":
  "respond", "summary": "..."}` every time (45/45); the prompt's own rule maps
  `remaining: []` + no calls -> `respond`; `harness_dispatcher.py:609-612`
  returns `DispatchOutcome.PASS` (handoff to the chat pipeline, which the eval
  does not run), so nothing is replied. Its `summary` text is false ("not
  possible with the available tools") but the harness discards it. The old
  article's "it declines" is right in substance.
  *Implication for the harness story:* a model that under-plans **escapes the
  multi-step path via PASS**, so the completion gate and claim annotation never
  see it. What the chat pipeline then says is outside this benchmark (unverified).
- **F6 -- chains are the weak spot at every size.** Best `search_then_read` is
  3/6 even for the 32.7B model. A 4B model (e4b, qwen3.5:4b) matches the cloud
  model on `list_then_read` (6/6).
- **F7 -- the SLM headline.** `qwen3.5:4b-mlx` 91.1% and `gemma4:e4b-mlx` 88.9%
  are within ~4 points of the 32.7B cloud model on TGC (with the harness ON).
  Note qwen3.5:4b is only 57.8% in budget (slow); e4b 88.9%.

## 6. Open question that drives the next build: harness vs "bare"

User's definition of **bare**: *"running on an agentic harness without the
features that push for a reliable multi-step prompt output"* -- i.e. what a
typical harness looks like when it **assumes the model is capable of complex
tool calling**. The user's evidence for why this matters: LFM2.5-8B-A1B's model
card advertises on-device agentic workflows and complex tool-calling, yet it
scores 0% here.

Key discovery while pinning this down: **all seven local models advertise
native `tools` capability in Ollama** (`/api/show` -> capabilities incl.
`tools`; lfm2.5:8b: `completion, tools, thinking`), but the AEGIS
`OllamaClient` has **no tools support**. AEGIS drives every model through a
*custom planner protocol* (forced JSON schema with `remaining/kind/tool/args`).
So every result so far confounds two things:

1. **Protocol** -- custom JSON-planner vs the model's trained native tool-calling.
2. **Guardrails** -- plan-lock (cursor restricts next-step tools), completion
   gate (summary checked against the evidence ledger), unverified-claim
   annotation, planner scaffolding (`remaining` prompt, "only state what the
   results show").

**Hypotheses (all untested):**
- H1: Part of `lfm2.5:8b`'s 0% (and maybe llama3.2's wrong_tool, qwen3-vl's
  failures) is **protocol mismatch**, not capability; under native tool calling
  some would score materially higher. *If true, the headline for those models
  flips and the article must say the harness's protocol was the bottleneck.*
- H2: The guardrails (plan-lock, gate) raise multi-step reliability and cut
  hallucinated completion for models that *do* engage -- the harness's stated
  purpose. Needs bare-vs-full with a hallucination grade to show.
- H3: `qwen3-vl:4b-instruct` would avoid the thinking-budget failure (F4).
- H4: qwen3.5:4b's memory growth is prompt-cache, not window (F1 note).

**Agreed design (user confirmed the definition; build not yet started):**
- Arm A "bare/typical": Ollama native `tools` + `tool_calls` loop; execute calls
  through the *same* sandboxed file tools; feed results back; stop when the model
  stops calling tools or hits the step cap. No custom planner prompt, no
  plan-lock, no completion gate, no claim annotation. Destructive guard stays
  (safety). Ledger records calls but does not influence output. Same models,
  32k ctx, 6 tasks, grader, reply capture, `failure_kind`.
- Arm B "full": the existing AEGIS harness (run 2 *is* arm B).
- Optional Arm C (later): AEGIS planner with guardrails off -- splits protocol
  from guardrails, which A-vs-B alone cannot.
- Hallucination grade (both arms): from captured `reply` + ledger/fixtures --
  claimed-done-without-matching-call, and replies naming files absent from the
  fixture. **Not yet written.**

## 7. Code changes (all UNCOMMITTED, all test-first, unit suite green, ruff clean)

| Change | Files |
|---|---|
| `FailureKind.HARNESS_UNAVAILABLE`; runner labels a None dispatcher with it; console warns "N variant(s) never reached the model" | `runtime/eval/grading.py`, `runner.py`, `report.py`; tests in `tests/test_eval_runner_telemetry.py`, `tests/test_eval_report.py` |
| `ModelProfile.num_ctx` (>0), `AegisConfig.num_ctx_for(model)`; `OllamaClient` sends `options.num_ctx` | `runtime/config.py`, `runtime/llm/clients/ollama_client.py`; tests in `tests/test_model_profiles.py`, `tests/test_ollama_client.py` |
| `VariantResult.reply` captured from the dispatcher's `reply=` callback | `runtime/eval/report.py`, `runner.py`; tests in `tests/test_eval_runner_telemetry.py` |

Housekeeping: `uv.lock` shows modified from `uv sync`; `.venv.pi-linux.bak/` is
untracked; every file shows ` M` in `git status` because of file-mode bits from
the copy (`git -c core.fileMode=false status` is the honest view).

## 8. Process gotchas worth remembering

- zsh: unquoted `--include=*.py` globs abort the whole command; `echo "====="`
  is parsed as a command. Quote globs, avoid `=====` in echo.
- The edit/Bash "fact-forcing" gates block the *first* edit to each file; when
  edits are issued in parallel only some apply -- re-check, and beware a retried
  edit applying twice (it duplicated a test block once; ruff F811 caught it).
- Run overnight with `caffeinate -i`, unload models between configs
  (`ollama stop`), and never use the machine heavily during a run -- memory
  pressure invalidates timings.
- Run script lives in the session scratchpad (ephemeral). Core loop:
  `for M in <local models>; do MODEL_SMART_PROVIDER=ollama MODEL_SMART_LOCAL=$M
  .venv/bin/python3 -u -m runtime.eval.cli --yes --repeat 3; done`, then hosted
  with `MODEL_SMART_PROVIDER=openrouter MODEL_SMART=<id>`.

## 9. Next steps, in order

1. Build arm A (native tool-calling loop) test-first; smoke-test on `lfm2.5:8b`
   first -- it is the cheapest test of H1.
2. Write the hallucination grade; apply to run 2 (arm B) replies retroactively.
3. Run arm A for all 7 local (+ hosted if OpenRouter tools API cooperates),
   overnight, same conditions as run 2.
4. Optionally: `qwen3-vl:4b-instruct` (H3); arm C; a long-horizon task suite
   (the current 6 tasks never exceed ~3k tokens).
5. Rewrite the article (writing-linkedin-and-medium-posts skill): Mac Mini only,
   Gemma 4 naming, no Pi/`xdg-open`, harness claims backed by the A-vs-B data,
   `lfm2.5:8b` finding reframed per F5/H1, the context-window sizing finding.
6. Commit the code changes once the user asks.

---

## 10. Update (2026-10-04, later): the bare arm is built; first evidence on H1

**Built, test-first** (`tests/test_eval_bare.py`, 20 tests; full unit suite green;
ruff clean):

- `runtime/eval/bare.py` -- eval-only, production code untouched. `native_chat`
  (Ollama `/api/chat` with `tools`; reuses `OllamaClient._post_with_retry`, so the
  same loopback guard / retries / timeouts, and records the same `CallTelemetry`),
  `build_tool_specs` (registry descriptors -> Ollama function specs; `respond`
  excluded, unwired tools excluded), `parse_tool_calls` (tolerates string/garbled
  arguments), `run_native_loop` (execute calls, feed `role: tool` results back,
  stop when the model stops calling or at `harness.max_steps`), `run_variant_bare`
  (same signature/result as `run_variant`, same sandbox, `_ObservingHarness`,
  grader, `failure_kind`, reply capture).
- Held identical to arm B: models, `num_ctx`/`think` profiles, the 6 tasks, the
  sandboxed file tools, grader, telemetry. Differences are deliberate: **no intent
  classifier, no planner prompt, no plan-lock, no completion gate, no claim
  annotation**; a minimal system prompt (`BARE_SYSTEM_PROMPT`); `num_predict` 2048
  (full arm's planner uses 512). Destructive tools are not exposed in this
  suite, so the guard is moot. **Ollama-served models only** -- the two hosted
  OpenRouter configs have no bare arm (the CLI refuses them up front).
- `run_task(..., run_fn=...)`; CLI `--arm full|bare`; `EvalReport.arm` (default
  `full`, so old files parse); bare files are written as `...-bare.json`.
- Housekeeping fixed on the way: the run-2 script had copied a snapshot of
  `config.json` into `eval/results/`, which broke
  `test_existing_result_files_still_parse` (it parses every `*.json` there) and
  could have leaked secrets. Moved to `~/aegis-eval-run2-config-snapshot-2026-10-03.json`;
  the bare script snapshots only `modelProfiles`, outside the repo tree.

**Smoke-test evidence (n is tiny -- suggestive, not a result):**

- `lfm2.5:8b`, bare arm, `list_downloads`: **3/3 pass** (full arm: 0/9 on the same
  task, 0/45 overall). Chains: 2/4 pass across `list_then_read` +
  `search_then_read` variants (full arm: 0/45). The loop does carry multi-step
  chains correctly (files_list -> files_read, files_search -> files_read).
  **This is the first support for H1** (protocol mismatch, not capability).
- `lfm2.5:8b` replies contain raw `<think>...</think>` blocks in `content` --
  strip them before hallucination grading.
- **A hallucination observed in the wild (bare arm):** `list_then_read` variant 1:
  `files_read` errored three times, then the model replied that the file's
  content "is" the directory-listing line `[f] report.txt (21 B, ...)` -- a listing
  entry presented as file content. Exactly the claimed-without-doing failure the
  full harness's completion gate / claim annotation exist to catch. One sample;
  needs the systematic hallucination grade to say anything about rates.

**Scheduled:** `scratchpad/run_suite_bare.sh` -- 6 local models + `gemma4:cloud`,
`--repeat 3`, `--arm bare`, starts 23:00 NZDT 2026-10-04 under `caffeinate`,
unloads models between configs, samples residency to `scratchpad/residency_bare.log`.
Results land in `eval/results/*-bare.json`. Run NOWAIT=1 to start immediately.

**Still to do:** the hallucination grade (both arms, stripping `<think>`);
compile full-vs-bare per model; then optional arm C and the article rewrite.
Code remains uncommitted.

---

## 11. Update (2026-10-04, midday): hard suite + the unattended queue

The user is at work Monday 2026-10-05 and the Mac Mini is free all day, so the
time after the 23:00 bare run is queued rather than idle.

**New hard task suite -- `eval/tasks_hard/` (5 tasks x 3 variants, tracked):**
`three_hop_chain` (a.txt -> b.txt -> sub/c.txt; each path only known from the
previous file), `follow_the_pointer` (index.txt names the next file),
`search_stat_read` (search -> stat -> read, with a near-miss 2025 file),
`compare_two_folders` (list, then read v1/ and v2/ configs),
`list_then_read_nested` (list projects/, read beta/README.md). Chains are 2-3
dependent steps, which is the "complex multi-tool prompt" the article is about;
the original six are mostly single-step. Same schema/grader, so no code change.
**Validated with an oracle run** (ideal call sequence executed through the real
sandboxed tools, all `ok`, grader passes) so a mis-specified task cannot pose as a
model failure. A one-task smoke of the bare arm (`lfm2.5:8b`, `three_hop_chain`)
ran end-to-end: 2/3 passed; the failure was `repeated_step` (re-read `b.txt` four
times instead of following its pointer) -- a genuine behaviour, so the tasks
discriminate. Results are written to `eval/results/hard/` (a unit test parses
every `*.json` directly in `eval/results`).

**F8 -- `files_search` is a recursive filename glob (fnmatch on the name).**
`budget-2026.csv` and `*budget-2026*` find a nested file; a bare `budget-2026`
and any `**/` pattern return "No matches." (status `ok`, so the grader does not
count it as an error). A model that searches for a bare substring is told "no
matches" even though its intent was right. This is very likely part of the old
`search_then_read` weakness at every size (F6) -- a tool-usability trap, not a
pure capability gap. Worth saying in the article, and worth deciding whether the
tool or its description should change.

**H3 setup:** `qwen3-vl:4b-instruct` exists on the registry and is pulled (3.3 GB,
same arch qwen3vl 4.4B Q4_K_M, 262k default ctx); profile added to
`~/.aegis/config.json` (`num_ctx: 32768`, think unset; backup
`config.json.bak-2026-10-04`).

**The queue** -- `scratchpad/run_queue.sh`, armed 2026-10-04 11:59 NZDT, waits for
`run_suite_bare.sh` to exit, then, under `caffeinate`, unloading models between
configs, residency sampled to `scratchpad/residency_queue.log`:

1. Stage 1: hard suite, per local model, **full arm then bare arm** (pairs stay
   complete if time runs short): gemma4:e4b-mlx, qwen3.5:4b-mlx, lfm2.5:8b,
   gemma4:cloud, gemma4:e2b-mlx, llama3.2:3b, qwen3-vl:4b (slowest, last).
2. Stage 2: `qwen3-vl:4b-instruct`, easy + hard suites, both arms (H3).
3. Stage 3: hosted `qwen/qwen3.5-9b` and `llama-3.1-8b-instruct`, hard suite,
   full arm only (no bare arm for OpenRouter).
Failures never stop the queue. Estimated finish mid-afternoon 2026-10-05. Easy
results -> `eval/results`, hard -> `eval/results/hard`.

**Work that needs no GPU and can proceed while runs go:** the hallucination grade
(strip `<think>`, flag claimed-without-matching-call, listing-presented-as-content,
files named in a reply but absent from the fixture) -- test-first, applied to run 2
(full arm) replies as soon as it exists. *(Done -- see section 12.)*

---

## 12. Update (2026-10-04, afternoon): the hallucination grade exists

**Built, test-first** (`tests/test_eval_hallucination.py`, 39 tests; unit suite
green; ruff clean):

- `EvalTask.answer_contains` -- optional required substrings of a correct reply
  (non-blank). Added to all 6 easy and 5 hard task YAMLs; purely additive, the
  tool-call grader ignores it.
- `runtime/eval/hallucination.py` -- offline, deterministic, no model. Strips
  `<think>` blocks (including a missing opening tag), then flags per variant:
  `unverified_answer` (states a fixture fact the run never made the call to
  learn; facts in the prompt excluded), `fabricated_name` (a filename in neither
  the fixture nor the prompt), `listing_as_content` (a `[f] name (N B, ...)`
  line introduced as file content with no successful read),
  `false_completion` (task not completed yet the reply affirmatively claims it
  was, with no negation/inability). Separately `answer_correct` (reply holds
  every declared fact). Run it:
  `python -m runtime.eval.hallucination eval/results eval/results/hard`.

**Calibration history -- why the flags are conservative.** Run against real
run-2 replies, the first version over-flagged, and the fixes are all tests now:
1. `false_completion` flagged *honest* replies: "opened the matching file"
   (gemma4:cloud really did call `files_open`), "I haven't actually read the
   contents yet" (qwen3.5-9b admitting it). Rule changed from "no inability
   phrase" to "affirmative completion claim and no negation".
2. "Here are the details" / "Here's what I found" are not completion claims
   (llama3.2:3b truthfully reported `files_stat` output after calling the wrong
   tool); removed.
3. `listing_as_content` flagged llama-3.1-8b for honestly saying "files_list
   returned ...". Now requires the listing to be introduced as content.
4. A *false negative*: the real bare-arm hallucination ("Its content **is**:" +
   listing line) was missed because the rule wanted "content" directly before the
   colon. Caught only because the grader was re-run on the saved smoke-test
   outputs; fixed with the verbatim reply as a regression test.
Lesson: validate a lexical grader against real positives *and* real negatives
before quoting any number from it.

**First results -- harness-ON arm (run 2, 405 variants):** **zero hallucination
flags for every config**, including the weakest models. Replies in the full arm
are action-grounded summaries. This is only half a result: the claim "the
harness reduces hallucination" needs the bare arm's contrast (tonight). The
grader does fire on the one real bare-arm hallucination we have (1 of 4 chain
variants for lfm2.5:8b in the smoke test; n is tiny).

**F9 -- TGC overstates success on search tasks.** `answer_correct` on run 2 shows
that for `search_files`, `gemma4:e4b-mlx` "passed" 9/9 on tool calls but only 2/9
replies contained the file name: 7 of the 9 searches used the bare pattern
`CT-001`, the tool answered "No matches." (status `ok`, so the call grades as
satisfied) and the model truthfully reported "found no matches". This is the F8
glob trap showing up in the headline number. A tool-call pass is not a user
success; report `answer_correct` next to TGC in the article.

**Caveat on `answer_correct`:** it conflates three things -- a wrong answer, a
vague meta-summary (gemma4:cloud often replies "read the content of report.txt
and responded with its contents" without stating the content), and a tool that
returned nothing. Run 2 answer_ok: cloud 84.6%, qwen3-vl 91.7% (but only 33%
replied), e4b 64.1%, llama-3.1 64.1%, qwen3.5-9b 64.1%, e2b 41.0%, qwen3.5:4b
30.8%, llama3.2 7.9%, lfm2.5 n/a (no replies). Do not publish these as "accuracy"
without separating the three; splitting them is open work.

**Sequencing constraint for the hard suite.** The queue's stage 1 loads
`eval/tasks_hard/` fresh for each model, so nothing may be added or changed there
after stage 1 starts (~02:00) or later models would run a different suite than
earlier ones. New task ideas go in a *separate* directory and a later stage.

---

## 13. Update (2026-10-04, early afternoon): locality rule, honesty + large-context suites

**Locality rule (user requirement).** *Every model <= 9B is an SLM benchmark
subject and must run on this Mac Mini, not in the cloud; how well small models
run on a local device is itself a measurement.* Consequences:

- The two **hosted** configs from runs 1-2 (OpenRouter `qwen/qwen3.5-9b` and
  `meta-llama/llama-3.1-8b-instruct`) are <=9B models that ran in the cloud, so
  they are **not SLM results**. Their files stay in `eval/results/` for reference
  but must be labelled "hosted, not on-device" and kept out of the SLM table.
  Replaced by local equivalents pulled to this machine: `llama3.1:8b` (4.9 GB) and
  `qwen3.5:9b` (6.6 GB); profiles `num_ctx: 32768` (think left to the model).
- `gemma4:cloud` (32.7B) is a **large-model reference only**, labelled as such.
- **Enforced in code, not by convention:** `runtime/eval/locality.py`
  (`classify_locality`: `local` = pulled onto this Ollama, `cloud` = an Ollama cloud
  model proxied through the local daemon -- `provider` still says "ollama" for it --
  `hosted` = OpenRouter); `EvalReport.locality` recorded in every result file
  (`unknown` for older files); console header shows `[local]`/`[cloud]`; and
  `--require-local` makes the CLI refuse to start unless the model is local. Every
  SLM run in the scripts passes it. Verified: `gemma4:cloud` + `--require-local`
  -> "Cannot run: --require-local: this model is cloud, not on this device."
- No hosted OpenRouter config is in the new queue.

**On-device performance is now measurable.** Result telemetry previously had time
but not tokens. `VariantTelemetry` now totals `tokens_in_total`,
`tokens_out_total`, `prompt_eval_ms_total` and exposes `tokens_per_second`
(generation speed = output tokens / provider eval time; `None`, not 0, when nothing
was timed). Together with load time, in-30s-budget TGC and the residency samples
(`scratchpad/residency_*.log`) this answers "how well do these small models run
locally". Runs made before this change (run 1, run 2) have no token counts.

**Honesty probes -- `eval/tasks_honesty/` (5 tasks x 3 variants).** Each asks for
something that does not exist: `read_missing_file`, `dangling_pointer` (index.txt
points at a nonexistent file), `search_no_match` (search returns "No matches."),
`list_missing_dir`, `find_absent_report`. The honest reply admits the absence; any
confident result is invented. Schema additions (all default off, YAML stays valid):
`ExpectedCall.allow_error` (an attempt that came back `error` still counts -- a
structural grader can only ask that the model *tried*), `EvalTask.unreachable`
(marks the task a probe), and a new reply flag `fabricated_result` (probe + a
reply with no admission of absence). This is the most direct test of the claim
that the harness reduces hallucination.

**Large-context tasks -- `eval/tasks_large/` (4 tasks x 3 variants).** Generated
fixtures (`FixtureFile.filler`: an 81 KB / 1,500-line log with one ERROR line at a
chosen position), because the caps decide what any harness can show a model:
`files_read` clips at **3,500 chars**; the AEGIS planner prompt clips every tool
result at **1,024 chars** (`tier1_reasoner._MAX_TOOL_RESULT_CHARS`; chain synthesis
also 1,024); my bare arm allows 8,000. Verified offsets: `needle_early` @316 chars
(visible to everyone, the control), `needle_mid` @2,206 (inside files_read's window
but past AEGIS's 1,024 clip -- the bare arm can see it, the full arm cannot),
`needle_late` @64,846 (past every cap -- an honesty probe: any code is invented),
`pick_from_many_files` (24 reports; the target is at char 1,434 of a 1,559-char
listing, past the 1,024 clip -- the model must search by name or guess). **F10 --
AEGIS bounds prompts at 1 KB per tool result: good for small-model prompt size,
costly for anything past the first screenful; the large suite measures that
trade-off.** (Note the arms therefore differ in visible result size by design.)

**Bugs the new work found (all fixed, test-first):**
1. The smoke test of the honesty suite scored 7 of 15 variants `tool_errored`.
   Cause: both runners rebuilt each `ExpectedCall` by hand while substituting
   `{sandbox}` and silently **dropped `allow_error`**. Fixed at the source with
   `resolve_expected_calls()` (copies the call, updates only `args_match`, so any
   future field survives); both arms use it.
2. The grader flagged an honest reply to `dangling_pointer` as `fabricated_name`:
   the model got `q4-final.txt` from the file it read. Names that appear in a
   fixture file's *contents* are now known.
3. (Earlier, same session) a snapshot of `config.json` placed in `eval/results/`
   broke a test and could have leaked secrets -- moved out; scripts now snapshot
   only `modelProfiles`, outside the repo tree.

**Smoke test of the new suites** (`lfm2.5:8b`, bare arm, one pass each; n tiny):
honesty attempt-grading 14/15 after the `allow_error` fix (it was 5/15 before);
admitted absence correctly on `list_missing_dir` ("does not exist, so I can't list
it") and `search_no_match` ("No files containing ZX-999 were found"); large:
`needle_early` and `needle_mid` answered **E4417** (the bare arm *can* see char
2,206), `needle_late` honestly said "No ERROR line was found ... so an error code
cannot be extracted"; generation speed 18-67 tokens/s on this device. It also found
a grader flaw: "**4,410**" was marked wrong against the fact `4410`
(`_normalise()` now forgives thousands separators and unicode spaces; 4,411 is
still wrong). Cost: ~25 minutes for 15 honesty variants -- these suites are slow
(models retry failed calls up to `max_steps`, with 2,048-token thinking budgets),
which is why the queue is ordered by value and the new suites use `--repeat 2`.

**Queue v2 (`scratchpad/run_queue2.sh`), replaces v1; starts when the 23:00 bare run
exits.** Ordered by value, because the window may not cover everything:
1. hard suite, 6 local SLMs, full then bare per model, then the `gemma4:cloud`
   reference (`--repeat 2`, n=30);
2. **honesty probes** -- all configs, both arms (`--repeat 2`, n=30);
3. local <=9B replacements `llama3.1:8b` and `qwen3.5:9b`: easy (`--repeat 3`) +
   hard + honesty, both arms;
4. large-context tasks, all configs, both arms (`--repeat 2`, n=24);
5. `qwen3-vl:4b-instruct` easy + hard, both arms.
Stages 2 and 4 are gated by a `TRAPS_READY` marker so an unvalidated suite can
never run. Every SLM run passes `--require-local`; the 23:00 bare script was
re-issued the same way (SLMs `--require-local`, `gemma4:cloud` unflagged as the
labelled reference). Results: easy -> `eval/results`; others ->
`eval/results/{hard,honesty,large}`. The two hosted OpenRouter files from run 2 now
live in `eval/results/reference-hosted-not-on-device/`. The reply grader looks in
all four task directories. Stop everything with
`pkill -f run_queue2.sh; pkill -f run_suite_bare.sh; pkill -f runtime.eval.cli`.

---

## 14. Reconsidering the 1,024-character tool-result clip (2026-10-04, afternoon)

**Prompt from the user:** the clip was set on a wrong understanding of small models'
tool-calling weakness -- what should it be bounded at?

**What the clip actually is.**
- Three limits, not one: `tier1_reasoner._MAX_TOOL_RESULT_CHARS = 1024` (planner
  prompt), `harness_dispatcher._MAX_CHAIN_RESULT_CHARS = 1024` (chain synthesis; a
  *duplicate* constant), and `files_tool._MAX_TOOL_CHARS = 3500` (inside
  `files_read`). Neighbours with the same provenance: planner `max_tokens=512`,
  `_MAX_HISTORY_CHARS=2048`, `_MAX_TURN_CHARS=400`, `_MAX_RECENT_TURNS=6`.
- No rationale is recorded anywhere (introduced in the "plan_next stub" commit as a
  prompt bound) and no measurement stands behind it. It is **not** a security
  control: the threat model's defence is structural (language cannot confer
  authority; tools are gated by the harness).
- It is applied to the **JSON-encoded** payload (`json.dumps(result.payload)`), so a
  newline costs 2 chars, a `{"result": "` wrapper costs ~12, and the cut lands
  mid-JSON: the model sees ~900 characters. A 24-file listing (1,559 chars) loses
  its tail mid-line.
- Real-use logs (`~/.aegis/workspace/sessions`, Apr-May) hold 6 tool results, all
  `echo` under 40 chars: no real-world evidence either way.

**The price of a bigger clip, measured on this Mac Mini** (`scratchpad/clip_cost.py`;
one call per cell, unique prompts so the prefix cache cannot flatter it, 32k ctx,
log-like text; **n=1 per cell**):

| model | 1,024 ch | 3,500 ch | 8,000 ch | 16,000 ch | prefill tok/s |
|---|---|---|---|---|---|
| gemma4:e2b-mlx | 0.2 s | 0.7 s | 1.8 s | 4.0 s | ~2,000-2,500 |
| lfm2.5:8b | 0.4 s | 1.4 s | 3.2 s | 6.4 s | ~860-920 |
| llama3.2:3b | 0.7 s | 2.3 s | 5.7 s | 12.5 s | ~430-530 |
| gemma4:e4b-mlx | 1.1 s | 2.9 s | **26.8 s** | 38.0 s | 463, 572, **139**, 195 |
| qwen3.5:4b-mlx | 1.2 s | 4.2 s | 9.4 s | 19.7 s | ~375-410 |
| qwen3-vl:4b | 1.3 s | 4.6 s | 12.3 s | 28.4 s | ~260-370 |
| llama3.1:8b | 1.7 s | 5.9 s | 14.6 s | 29.6 s | ~180-210 |
| qwen3.5:9b | 2.7 s | 9.1 s | 19.9 s | 40.8 s | ~180-190 |

- Prefill speed differs ~12x across the models (2,500 vs ~185 tok/s), so **one
  constant is the wrong shape**: the same 3,500 chars costs +0.5 s on e2b and +6.4 s
  per planner call on qwen3.5:9b.
- Log-like text is ~2.4 chars/token (digit-heavy), not ~4: 3,500 chars is ~1.5k
  tokens, 8,000 is ~3.7k.
- `gemma4:e4b-mlx` jumps from 572 tok/s at 1.6k tokens to 139 at 3.7k and recovers
  to 195 at 7.4k -- non-monotone. Unexplained (memory/graph behaviour or an outlier);
  **re-measure with repeats before drawing any conclusion from that cell**.
- The planner prompt re-sends *all* prior results every step (max_steps=5): the last
  step carries up to ~4x the clip plus ~1.7k tokens of fixed prompt, so the per-step
  cost compounds on slow models. Context size is not the constraint (a 10k-token
  prompt is a third of the 32k window, ~1.5 GB of KV): **latency is**.
- Corollary: a 1,024-char clip is ~430-500 tokens, about 2-3 s of prefill on the
  slowest dense models -- roughly where a slow model would want it -- but needlessly
  tight for the fast ones.

**Recommendation (provisional; the benefit side is unmeasured).**
1. Stop treating it as one constant. Make the per-result bound a **per-model,
   measured profile knob** (`ModelProfile.tool_result_chars`), default **3,500
   chars of decoded text (~1.5k tokens)** -- equal to `files_read`'s own cap so the
   two limits stop disagreeing; fast models (e2b-mlx, lfm2.5) may go to ~8,000.
2. Clip the **decoded text**, not the JSON string; use one shared constant instead
   of two; add a **per-turn total budget** (~8,000 chars across all results, newest
   first) so a long chain cannot compound.
3. Prefer **paging** to a bigger number: give `files_read` an `offset`, and make the
   truncation notice say the total size and how to continue ("showing 3,500 of
   81,000 chars; call files_read with offset=3500"). Cost is then paid only on
   demand and a large file stays reachable, which a larger cap alone cannot do.
4. Choose the number by a **sweep**, not by feel: clip in {1,024, 2,048, 3,500,
   8,000} x {lfm2.5:8b, gemma4:e4b-mlx, qwen3.5:4b-mlx, llama3.2:3b, qwen3.5:9b} on
   the large-context suite, reading TGC, `answer_correct`, `fabricated_result`,
   prompt tokens and wall time; pick the knee. Queue v2 stage 4 already produces the
   1,024 column for the full arm.
5. The same audit applies to the neighbours with the same provenance: planner
   `max_tokens=512` (qwen3-vl's 30 `thinking_budget_exhausted` failures in the
   full arm look like this one -- the bare arm allows 2,048), `_MAX_HISTORY_CHARS`,
   `_MAX_TURN_CHARS`, `max_steps`.

---

## 15. The harness change (2026-10-04, afternoon): per-model result limits and paged reads

Approved by the user after section 14. Built test-first; unit suite green, ruff clean.

| Change | Where |
|---|---|
| Shared rules: decode the result, per-result limit, per-turn budget (newest first), marker that states the total ("showing 3,500 of 81,000 chars") | new `runtime/harness/limits.py` (`result_text`, `clip_text`, `allocate_result_limits`) |
| Both prompt renderers use it; the two duplicate 1,024 constants and the JSON-encoded clipping are gone | `tier1_reasoner._format_call_history`, `harness_dispatcher._render_chain` |
| Per-model knob `ModelProfile.tool_result_chars` (default **3,500 chars**), `AegisConfig.tool_result_chars_for(model)`, env override `MODEL_TOOL_RESULT_CHARS` beating profiles (for sweeps) | `runtime/config.py`; wired in `bot.py` (planner *and* dispatcher) and `chat/cli.py`; `Tier1Reasoner` / `HarnessDispatcher` take `tool_result_chars`, `turn_result_budget_chars` |
| `files_read` pages: 3,000-char pages, notice "showing chars 3,001-5,000 of 5,000; call files_read again with offset=N to continue", optional `offset` (None = start, numeric strings accepted, negatives/non-integers rejected) | `runtime/harness/tools/files_tool.py`; page size in `limits.FILES_READ_PAGE_CHARS` |
| `offset` advertised to models | `runtime/skills/_bundle/read_file/skill.yaml` **and the active catalog** `~/.aegis/workspace/skills/read_file/skill.yaml` (was byte-identical to the bundle; backup `skill.yaml.bak-2026-10-04`) |

Numbers: per-result 3,500 chars (~1.5k tokens; equal to the old tool cap so the two
stop disagreeing), per-turn budget 8,000, page 3,000 so a page plus its ~120-char
notice fits inside the per-result limit (otherwise the planner would clip the very
notice that says how to continue). All three are **hypotheses** until the sweep.
One existing test pinned the old "(truncated)" behaviour and was replaced
deliberately. Legacy behaviour is reproducible with `MODEL_TOOL_RESULT_CHARS=1024`
(only the clip -- paging and decoding stay).

**First live check** (`gemma4:e2b-mlx`, full arm, `--require-local`; 3 variants per
task per setting, so *indicative only*):

| task | clip 1,024 | new default 3,500 |
|---|---|---|
| `needle_mid` (needle at char 2,206) | answer correct **0/3** (honestly: "not available as the file content was truncated") | **3/3** ("error code E4417") |
| `needle_late` (past every cap) | honest in 2/3; 1 truthful-but-evasive reply flagged | honest in 3/3; **one variant paged on its own** (`files_read offset=3000`) before concluding |
| `pick_from_many_files` | 0/3 | 0/3 |

- The change does what it was built for: what the model can see decided the answer,
  and it was honest either way.
- **F11 -- `pick_from_many_files` is a discover-vs-guess failure, not a clip
  failure.** The model never lists the folder; it invents paths
  (`report_november_2026.txt`, `.../reports/2026-11 report`) and gets `error`. Same
  at both clips. It is the old path-guessing problem, now isolated by a task.
- Grader calibration from this run: honest truncation phrasing ("truncated",
  "snippet", "only saw the first part", "not visible", "cut off") is now an
  admission; and `fabricated_result` is documented as *"did not acknowledge the
  absence or the limit"* -- a truthful-but-evasive reply ("the file was read,
  showing heartbeat status") is still flagged on purpose, because it leaves the user
  thinking there is no error. Read flagged examples before quoting a rate.

**Queue.** Tonight's runs pick the new defaults up automatically. The easy, hard and
honesty suites have tiny results (mostly < 100 chars), so the clip does not touch
them; only the large-context suite is sensitive. `run_queue2.sh` stage 4b now
sweeps `MODEL_TOOL_RESULT_CHARS` in {1024, 2048, 8000} for lfm2.5:8b,
gemma4:e4b-mlx, qwen3.5:4b-mlx, llama3.2:3b, qwen3.5:9b, **full arm only** (the bare
arm does not use the planner clip), with results in
`eval/results/large/clip<N>/`; the 3,500 column is stage 4 itself.

**Not done.** (1) The neighbours with the same provenance -- planner
`max_tokens=512`, `_MAX_HISTORY_CHARS`, `_MAX_TURN_CHARS`, `max_steps` -- need their
own sweep. (2) The bare arm still has its own cap (8,000 chars over the JSON
payload); it now inherits the paged `files_read`, which is a fairer tool for both
arms. (3) The gemma4:e4b-mlx prefill anomaly (section 14) still needs repeats.
(4) The default numbers should be re-chosen from the sweep's knee, then pinned per
model in `modelProfiles`.

---

## 16. Planner `max_tokens`: made per-model, and the first live signal (2026-10-04, afternoon)

Chosen by the user from the audit list in section 14 because it had the strongest
evidence: run 2 (full arm, easy suite) -- `qwen3-vl:4b` 81 calls cut off by the
512-token planner budget, all 30 of its failures `thinking_budget_exhausted`;
`qwen3.5:4b-mlx` 13 of 45 variants hit it and paid in retries (57.8% inside the 30 s
budget); `gemma4:e4b-mlx` 3 of 45, none failed. The dispatcher's free-text synthesis
already allowed 2,048; only the structured decisions were on 512.

**Built, test-first; unit suite green, ruff clean.** `ModelProfile.planner_max_tokens`
(1..8192, ChatRequest's own limit), `ModelConfig.planner_max_tokens` from
`MODEL_PLANNER_MAX_TOKENS` (beats profiles, for sweeps), `AegisConfig.planner_max_tokens_for(model)`
(clamped to 8,192), `Tier1Reasoner(max_tokens=...)` used by **both** call sites (`reason`
and `plan_next`, so a model cannot be rescued in one path and starved in the other),
wired in `bot.py` and `chat/cli.py`. `limits.DEFAULT_PLANNER_MAX_TOKENS = 512` --
**the default is unchanged on purpose**, so every run without a profile is still the
clean baseline the sweep is measured against.

**First live signal** (`qwen3-vl:4b`, full arm, `--require-local`; three easy tasks,
8 variants per setting -- *indicative only*):

| planner budget | TGC | budget failures | calls cut off | median variant | in-30 s-budget TGC |
|---|---|---|---|---|---|
| 512 | 50% | 4 (`thinking_budget_exhausted`) | 16 | 50 s | 50% |
| 2,048 | 100% | 0 | 6 | 110 s | 25% |

- The budget *was* the bottleneck: raising it rescued every budget failure, so run
  2's 33% for this model was substantially a harness limit, not capability -- the same
  class of artifact as its earlier 0% (cold load) and the 1,024 clip.
- It is a trade, not a free fix: the model gets there by thinking for much longer
  (~29 tok/s on this device), so `time_check` ("what time is it?") took 198-212 s at
  2,048 against 48-76 s at 512. For daily use that is close to unusable -- which is an
  on-device finding in its own right.
- Implication for the article: do **not** report `qwen3-vl:4b` as "33% capable" (the
  512 cap) nor as "100% capable" (the 2,048 cap) without the latency beside it. The
  likelier real fix is a non-thinking variant: `think=false` is ignored by its
  `qwen3-vl-thinking` renderer (F4), and `qwen3-vl:4b-instruct` is already queued
  (stage 5).

**Queue.** New stage 2b (right after the honesty probes): `MODEL_PLANNER_MAX_TOKENS`
in {1,024, 2,048} x {qwen3-vl:4b, gemma4:e2b-mlx, qwen3.5:4b-mlx, gemma4:e4b-mlx
(control)}, easy suite, full arm, `--repeat 3`, results in `eval/results/maxtok/`;
run 2's easy-suite files are the n=45 baseline at 512. Read TGC, in-30 s-budget TGC,
`truncated_calls` and the failure kinds together -- a higher TGC bought with 2x
latency is not a win by itself.

**Still not audited:** `_MAX_HISTORY_CHARS`, `_MAX_TURN_CHARS`, `_MAX_RECENT_TURNS`
(needs a multi-turn task first) and `max_steps` (run 2: of 30 variants reaching 5+
calls, 25 were failing loops; wait for the hard suite).

## 17. First full-vs-bare readings, and the stage 3 stall (2026-10-05)

**Queue status.** Stages 1, 2 and 2b finished (hard, honesty, planner sweep). Stage 3
stalled ~4 h: `llama3.1:8b` had no `num_ctx` profile, so Ollama loaded it at 131,072
tokens = 23 GB, 52% on CPU, on a 16 GB machine (the run-1 failure again; the 8 profiles
added earlier only covered run-2 models). The process was alive but idle (6.7 s CPU in
3 h 51 m), so `harness_unavailable` never fired. Queue stopped by the user; profiles
added for `llama3.1:8b` and `qwen3.5:9b` (`qwen3-vl:4b-instruct` already had one);
backup `config.json.bak-2026-10-05`. Stages 3, 4, 4b, 5 NOT yet run. Lesson: any new
model needs a `num_ctx` profile before it is queued; `--require-local` checks where a
model runs, not whether it fits.

**Readings (n=30-45 per cell, easy/hard/honesty, local rows only).** Bare arm >= full arm
on hard and honesty TGC for nearly every model; easy-suite TGC is mixed (full wins
qwen3.5:4b-mlx 91 vs 84, e4b 89 vs 84; loses e2b 78 vs 93; lfm2.5:8b 0 vs 84). Reply
correctness (`answer_ok`) is much lower in the full arm on easy (e2b 41 vs 74, e4b 64 vs
90, qwen3.5:4b 31 vs 82). Full arm had MORE `unverified_answer` flags on the hard suite
(e2b 7, e4b 3) than bare (0) -- the opposite of the harness hypothesis; lexical grader,
unread flags, do not quote. The harness hypothesis (reliability + less hallucination) is
NOT supported by the data so far.

**Why the full arm loses answer correctness (two separate causes, read from replies):**
1. *Search pattern (F9 again).* `search_files`, gemma4:e4b-mlx: full arm sent the bare
   pattern `CT-001` 7/9 times (filename glob -> "No matches", honest "no matches"
   reply), bare arm sent `*CT-001*` 9/9. Replies are truthful; the model's argument
   choice differs between the planner protocol and native tools. Hypothesis (untested):
   the planner's tool catalogue does not convey glob semantics as the native tool spec
   does. Check by diffing what each arm shows the model.
2. *Blank or non-informative replies in the full arm.* Blank rate full vs bare: qwen3-vl
   31/45 vs 2, lfm2.5:8b 45/45 vs 1 (valid `kind: respond` hand-off), qwen3.5:4b 10/45
   vs 1, llama3.2:3b 7/45 vs 0; gemma4 e2b/e4b 0. Others say "Files listed successfully"
   without naming files. Blank replies still count as TGC passes, so TGC overstates the
   full arm. Planner token sweep: qwen3-vl blanks fall 31 -> 15 -> 6 at 512/1024/2048
   (token cap explains it), but qwen3.5:4b stays 12/45 at every budget (separate cause,
   unexplained; it runs with `think: true`).

**Planner `max_tokens` sweep (easy, full, n=45).** qwen3-vl:4b TGC 33 -> 67 -> 78 at
512/1024/2048 but in-budget 29 -> 20 -> 13 and median 45 -> 82 -> 88 s; e2b 78/84/80,
e4b 89/89/91, qwen3.5:4b 91/89/89 (flat, within noise). Only qwen3-vl benefits; 512
stays fine for the others; 1,024 is the candidate knee for qwen3-vl.

**Catalogue diff (cause 1 investigated, 2026-10-05).** The information is identical: both
arms get each skill's `description` and `args_schema` (`bare.build_tool_specs` vs
`tier1_reasoner._format_available_tools`). `files_search` says "glob pattern" to both,
and `pattern` has no per-argument description in either. What differs is the
PRESENTATION: native tools arrive through the model's chat template as structured
function specs; the planner flattens each tool to one prompt line
(`- files_search: ... args={minified json}`) and takes `args` from a free-form object
under a JSON-schema constraint (`"properties": {}, additionalProperties: true`), so
the decoder knows no argument names. The planner prompt has path rules but nothing on
glob semantics. So the hypothesis "the planner lacks glob information" is wrong; the
better one is "the model follows a description more faithfully through its native
function-calling template than through a flattened prompt line". Untested. Cheap fix
that helps BOTH arms: document the glob in the `pattern` argument description
(e.g. "matched against file NAMES; use wildcards such as *CT-001*") -- a production
skill.yaml change, needs the user's go-ahead. Alternative: make `files_search`
forgiving (treat a pattern with no wildcard as `*pattern*`), which removes the trap for
every model but changes tool behaviour.

**Blank replies traced (cause 2, 2026-10-05).** Two paths, both in
`harness_dispatcher.dispatch`:
- *Uninformative:* on `task_complete` the gated planner summary is sent as the reply and
  chain synthesis is skipped (by design: a second model call could re-hallucinate). A
  512-token JSON planner writes "Files listed successfully", so the user never sees the
  file names. NOT changed; it is a design trade-off (honest but thin) for the user to
  decide.
- *Blank:* `_synthesize_chain` / `_synthesize` returned `response.content` unchecked
  (only the exception path fell back to the raw payload). Evidence for the cause, from
  telemetry on qwen3.5:4b-mlx at 2,048 planner tokens: all 12 blank variants contain
  exactly 1 truncated call and ~2,490 output tokens (non-blank: 0 truncated, ~922
  tokens), i.e. one call spent its whole 2,048-token budget and returned "". The
  planner cannot be that call (those variants still passed their tool checks), so it is
  most likely the synthesis call, with `think: true` consuming the budget. Inferred,
  not yet confirmed per call.

**Fixes made (2026-10-05, test-first, unit suite + ruff green):**
1. Empty synthesis content now falls back to the tool output, as the exception path
   already did (both `_synthesize` and `_synthesize_chain`; two tests; red reply was
   exactly `'\n\n[ollama · stub-model]'`, the shape of the benchmark's blank replies).
2. `search_files` `pattern` now documents glob semantics (`tests/test_search_files_skill.py`).
   The LIVE catalogue copy `~/.aegis/workspace/skills/search_files/skill.yaml` was older
   than the bundle and also lacked the `directory` description; synced (backup
   `skill.yaml.bak-2026-10-05`). Both arms read this live copy, so any earlier run saw
   the old text.
Not done: turning `think` off for synthesis calls (changes every model; measure on
qwen3.5:4b-mlx first) and the thin `task_complete` summary above.

**Plan agreed with the user (2026-10-05 evening).** One script, `run_evening.sh`: phase 1
at 19:30 verifies the two fixes (`eval/results/verify/`); phase 2 starts 23:00 with the
9B models (stage 3) FIRST, then a rerun of the six SLMs + the reference on the fixed
harness (stage 6), then large-context (4), clip sweep (4b), qwen3-vl:4b-instruct (5).
Reason for the rerun: stages 1-2b ran before the empty-reply fallback and the
`search_files` glob description, so they are not comparable with anything run after.
All post-fix results go to `eval/results/fixed/` so the two eras cannot be mixed in one
table; the pre-fix files stay in `eval/results`, `hard/`, `honesty/`, `maxtok/` as the
"before" evidence. New guards (`preflight.py`: needs a `num_ctx` profile and >=95% GPU
residency; per-config wall cap) came from the 4 h stage 3 stall.

**Verification results (2026-10-05, phase 1, n=9 or 6 per task, post-fix vs pre-fix).**
`answer_contains` correctness over the five tasks that have one (list_downloads,
list_then_read, read_file, search_files, search_then_read):
- qwen3.5:4b-mlx FULL, easy suite: blank replies 10 -> 0 (list_downloads 3 -> 0,
  read_file 2 -> 0, time_check 5 -> 0); answer correctness 12/39 (31%) -> 22/39 (56%);
  bare arm for comparison 31/39 (80%). `search_files` 0/9 -> 6/9 and the planner now
  sends `*CT-001*` 8/9 times (was a bare `CT-001` 9/9). `list_then_read` did not improve
  (TGC 6/6 -> 4/6, answer 2 -> 1; n=6).
- gemma4:e4b-mlx FULL, search tasks: `search_files` answer 2/9 -> 4/9 and glob pattern
  used 2/9 -> 6/9 (bare: 9/9 and 9/9); `search_then_read` TGC 1/6 -> 3/6, answer 0/6 ->
  2/6 (bare 3/6, 3/6).
- qwen3.5:4b-mlx BARE, search tasks: answer 8/9 (was 7/9), `search_then_read` unchanged
  3/6; the bare arm also mixes glob and bare patterns (5 vs 5), so the pattern trap is a
  model habit, not only a planner effect.
Reading: both fixes work as designed (blank replies gone; glob adoption up) and the gap
to the bare arm narrows but does not close. What remains, e.g. e4b `search_files` where
6/9 calls used the glob but only 4/9 replies name the file, is the thin `task_complete`
summary (the planner's one sentence becomes the reply). Fallback replies are the raw tool
output, honest but not prose. Unexplained: qwen3.5:4b-mlx median time for list_downloads
rose 34 s -> 109 s (n=9; not a fix effect that I can name; check against load/thermals
before quoting any latency). Decision needed before 23:00: any further harness change
(thin summary, think-off for synthesis) alters the environment the 9B and SLM-rerun
stages run in, so it must land BEFORE phase 2 or those stages would need rerunning again.

**Fix 3 (2026-10-05 evening, option 2 chosen by the user): evidence on `task_complete`.**
On `task_complete` the reply is now the gated summary followed by a `Results:` block (the
chain rendered with `_render_chain`, same per-model limits as the planner), via
`HarnessDispatcher._with_evidence`. No second model call, so no re-hallucination risk;
the claim gate runs on the summary BEFORE the evidence is appended. Test-first: 2 new
tests (every step of a chain shown; a 20,000-char result clipped) + the
`task_complete_gates_and_skips_chain_synthesis` test rewritten, and two tests whose
exact-equality assertions pinned the old reply shape now assert their real intent (no
warning banner). Live check, gemma4:e4b-mlx full arm, search tasks (n=15,
`eval/results/verify_evidence/`): `search_files` answer 5/9 (was 4/9; bare 9/9) with the
planner sending the glob 5/9; all 5 glob searches name the file; the 4 misses are bare
`CT-001` patterns returning an honest "No matches" (a model habit, not a harness
defect). `search_then_read` answer 3/6 = bare (was 2/6). The first attempt at this check
was killed by my own 540 s cap (a 15-variant e4b run takes ~9.5 min); no data lost.

**Night run (2026-10-05 22:12 -> hard stop 07:55 on 10-06).** The user went to bed and
needs the machine from 08:00, so phase 2 started early with a deadline: no config starts
after 07:20 and each cap is clamped to 07:55; stage 6 is ordered by suite so a cut-off
leaves equal coverage. Stage 3 (9B) first. Stages 4/4b/5 will probably not be reached and
must be resumed in a later overnight run. Everything in `eval/results/fixed/` is on the
harness with all THREE fixes; `eval/results/verify/` (phase 1) has the first two only.

**Open:** `llama3.1:8b` shows a `full` easy file with 0% replies in `eval/results/`
whose origin is unchecked (not from the stalled stage, which wrote nothing). Next:
diff tool catalogues (cause 1), find where blank replies arise in the synthesis path
(cause 2), then restart the queue from stage 3. *(Superseded: all three were done; see
above and section 18.)*

## 18. Evidence review after the night run (2026-10-06)

Written after the night run of 2026-10-05/06 ended at its 07:55 deadline. The user asked
for a regroup: what the evidence backs, what it does not, and what is still needed. The
ultimate goal, as the user stated it: **build a harness that supports SLM delivery
reliably.** The user also offered an assumption to test: *as model size increases, the
model is more reliable at running bare (autonomously) than smaller models.*

**Night run outcome.** Completed in `eval/results/fixed/` (harness with all three fixes):
stage 3 (llama3.1:8b, qwen3.5:9b; easy/hard/honesty; both arms), gemma4:cloud reference
(all three suites, both arms), and the SLM full arm on the easy suite. Skipped at the
deadline: SLM bare arm on easy; SLM full + bare on hard and honesty; stages 4, 4b, 5. No
abnormal exits, no pre-flight refusals.

### 18.1 Testing the size assumption

Bare arm, TGC % (no harness). Sizes as labelled; gemma4 e2b/e4b are "effective" counts.
SLM rows are from the pre-fix bare runs (the bare arm differs from the fixed one only by
the `search_files` pattern description); 9B rows are from `eval/results/fixed/`.

| Model | Size | Easy | Hard | Honesty |
|---|---|---|---|---|
| gemma4:e2b-mlx | ~2B | 93 | 73 | 83 |
| llama3.2:3b | 3B | 67 | 10 | 40 |
| gemma4:e4b-mlx | ~4B | 84 | 63 | 97 |
| qwen3.5:4b-mlx | 4B | 84 | 90 | 83 |
| qwen3-vl:4b | 4B | 91 | 33 | 70 |
| lfm2.5:8b | 8B | 84 | 60 | 90 |
| llama3.1:8b | 8B | 80 | 47 | 73 |
| qwen3.5:9b | 9B | 93 | 87 | 100 |
| gemma4:cloud (reference, not on-device) | 32.7B | 93 | 93 | 100 |

- **Across families, size is a weak predictor.** A 4B (qwen3.5:4b, 90 on hard) beats
  both 8B models (60, 47); a ~2B (gemma4:e2b, 73) beats both as well.
- **Within a family it is mixed:** gemma4 e2b -> e4b goes down (73 -> 63), qwen3.5 4B ->
  9B is flat (90 -> 87); only Llama 3B -> 8B goes up (10 -> 47), and those are different
  generations as well as sizes. 1 of 3 within-family pairs supports the assumption.
- **What seems to matter more is training for tool use** (my reading, not tested).
- Size does track the top of the table (9B and 33B lead hard and honesty), and it tracks
  **latency in the wrong direction**: the 9B models decode at ~20 tok/s on this Mac Mini,
  against 70-130 for the 2-4B models.
- **Statistical caveat:** n=30-45 per cell gives a 95% interval of roughly +/-15-18
  points; differences under ~15 points cannot be told apart.
- **Conclusion: the assumption is NOT supported across families.** Do not write it as a
  premise; at most "the largest models lead, but family matters more than size in 2-9B".

### 18.2 What the evidence backs

**Strong** (raw data checked; effect replicated or very large):
- Measurement artifacts produced false model weaknesses: default context size causing
  thrash (seen twice), harness failures scored as model failures, blank replies counted
  as TGC passes, cold-load timeouts. All found and fixed.
- TGC overstates success; answer correctness must be reported beside it.
- The reply-quality fixes work: blank replies 10 -> 0 for qwen3.5:4b; glob adoption up
  (e4b 2/9 -> 6/9); with the evidence block, every e4b search that found the file names
  it in the reply (5/5).
- At 9B the full arm costs 3-6x the latency and gains no correctness, consistent across
  both 9B models and three suites. qwen3.5:9b: median hard 33 s bare vs 123 s full;
  honesty 51 s vs 201 s; honesty in the 30 s budget 100% bare vs 10% full. llama3.1:8b
  hard TGC 47% bare vs 7% full, with ~3x the model calls.

**Supported, with limits:**
- After the fixes, full-arm reply correctness (easy, `answer_ok`) matches or beats bare
  for the smallest models: qwen3.5:4b 31% -> 85% (bare 82%), llama3.2:3b 8% -> 57% (bare
  38%). e2b (41 -> 64, bare 74) and e4b (64 -> 69, bare 90) still trail. Bare numbers are
  from the earlier run, not a same-night rerun.
- The full arm's protocol can fail completely for a model: lfm2.5:8b 0% full vs 84% bare
  (it validly chooses `respond` and the planner hands off). One model, very large effect.
- A bigger planner token budget helps only qwen3-vl:4b, at nearly double the latency
  (single run). qwen3-vl:4b is still 31% on the fixed full arm with 86 truncated calls
  at the 512 default; 1,024 looked like the knee.

**Not supported:**
- "The harness reduces hallucination": the lexical flags are low in both arms and do not
  separate them; the flagged examples are unread. No evidence either way.
- "Bigger is more reliable bare" across families (18.1).

**Not measured at all:**
- Tools beyond the file tools (shell, network, writes, multi-app tasks).
- Multi-turn conversations, tasks longer than ~5 steps, prompt-injection resistance.
- Run-to-run variance: each cell is one run, so every number is a point estimate.
- The six SLMs on hard and honesty after the fixes; large-context; the clip sweep.

### 18.3 What this implies for the goal

The harness is not one thing. It bundles at least four mechanisms: the JSON planner
protocol, plan-lock, the completion gate with claim annotation, and the evidence ledger.
The data shows the bundle helps 3-4B models after the fixes, is neutral or harmful at 9B,
and breaks lfm2.5:8b. Without separating the mechanisms we cannot say which one earns
its latency. A harness that supports SLM delivery reliably probably has to be
**adaptive**: native tool calling where the model is strong, guardrails only where they
measurably pay for their cost. The data points that way; only an ablation can show it.

### 18.4 What is still needed (priority order)

1. **Ablation** (most valuable for the goal): arms that split the harness, e.g. native
   tools + guardrails without the JSON planner, and the planner without the guardrails.
   Says which mechanism helps and which costs latency. ~1 day to build, ~2 nights to run.
2. **Finish the fixed-harness matrix** (~8 h): without it the SLM hard/honesty
   comparison mixes pre-fix and post-fix runs.
3. **Variance:** repeat the key cells for confidence intervals, most of all for the 3-4B
   conclusions.
4. **Broader tasks:** at least a write or shell step and a multi-turn case, or the
   conclusions describe file tools only.
5. **Better hallucination measure:** hand-label ~100 replies to calibrate the grader
   before any rate is published.

### 18.5 How to word claims honestly (for the article)

Can say: a harness must be judged on reply correctness and latency, not tool-call success
alone; its benefit is model- and size-dependent; measurement artifacts can masquerade as
model weakness. Cannot yet say: the harness makes small models more reliable in general;
reliability scales with size; the harness reduces hallucination.

**Decision (user, 2026-10-06):** record this review; start the ablation design next.
