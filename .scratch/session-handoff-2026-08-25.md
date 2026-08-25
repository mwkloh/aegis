# Session Handoff — 2026-08-25

For a fresh Claude Code session. **The next task is a suite re-run with the two measured model profiles active.** Everything needed to start is in "The task" below; the rest is context you should read before interpreting any result.

---

## The task: re-run the eval suite with the two profiles active

**Why:** `qwen3-vl:4b` needs `think=false` and `qwen3.5:4b-mlx` needs `think=true`. Both were measured (see "What was measured", below) but the suite has never been run with profiles *in place* — every number to date came from either an env override or the default. This run confirms the profile mechanism delivers the measured effect through the real config path, and produces the first results table that reflects how the system would actually be deployed.

**Step 1 — add the profiles to `~/.aegis/config.json`.** They are NOT there yet (verified at end of session: `model_profiles` is empty). This was left for the user deliberately; do not add them without confirming. Add a top-level section:

```json
"modelProfiles": {
  "qwen3-vl:4b":    { "think": false },
  "qwen3.5:4b-mlx": { "think": true }
}
```

**Step 2 — confirm they load before spending two hours:**

```bash
.venv/bin/python3 -c "
from runtime.config import get_config
c = get_config()
print({k: v.think for k, v in c.model_profiles.items()})
for m in ('qwen3-vl:4b','qwen3.5:4b-mlx','gemma4:e4b-mlx'):
    print(m, c.think_for(m))
"
# expect: {'qwen3-vl:4b': False, 'qwen3.5:4b-mlx': True}
#         qwen3-vl:4b False / qwen3.5:4b-mlx True / gemma4:e4b-mlx None
```

**Ensure `MODEL_SMART_THINK` is unset in `~/.aegis/.env`** — it is a global override that beats every profile, so a leftover value would silently invalidate the whole run. It was unset at end of session.

**Step 3 — run.** Local Ollama configs only; `think` is Ollama-only and the OpenRouter client never sends it, so the two hosted configs cannot be affected and re-running them would spend real tokens for nothing.

```bash
for M in gemma4:cloud gemma4:e2b-mlx gemma4:e4b-mlx qwen3.5:4b-mlx lfm2.5:8b llama3.2:3b qwen3-vl:4b; do
  MODEL_SMART_PROVIDER=ollama MODEL_SMART_LOCAL="$M" \
    .venv/bin/python3 -u -m runtime.eval.cli --yes
done
```

Budget **~2.5–3 h**. `qwen3-vl:4b` alone runs ~50 min even with thinking off. Run it in the background and let the completion notification wake you rather than polling.

**Step 4 — what success looks like.** Compare against `eval/results/2026-08-25T*.json` (thinking-off run) and `2026-08-24T*.json` (thinking-on run):

| Model | expect | because |
|---|---|---|
| `qwen3-vl:4b` | ~80% TGC | profile gives it `think=false` |
| `qwen3.5:4b-mlx` | ~100% TGC, ~73% in budget | profile gives it `think=true` |
| everything else | unchanged | no profile → model's own default |

If `qwen3-vl:4b` comes back at ~40%, the profile is not reaching the reasoner — check `AegisConfig.think_for` and the wiring at `runtime/chat/telegram/bot.py:914`, not the model.

**Step 5 — record the active profiles alongside the numbers.** Eval results now depend on `config.json` state (see "Known consequence" below). A results table without its profile set is not reproducible.

---

## Current state (verified at end of session, not assumed)

- Branch **`feat/eval-measurement-instrumentation`**, pushed to origin, 4 commits ahead of `origin/chore/ruff-zero-and-ci` (which is at `950ca48`, this branch's base). No PR opened — that was not discussed.
  - `13f37c9` eval instrumentation · `5e5076b` findings doc · `fe60842` thinking knob · `d8b5520` per-model profiles
- Working tree clean except `CLAUDE.md`, which was **already modified before this session started** — not this session's doing, leave it alone.
- 1498 unit tests pass. Ruff clean except 6 pre-existing `PLR0917` in files this session never touched.
- Live config: `smart_local=gemma4:e2b-mlx`, `smart_provider=ollama`, `smart_think=None`, **`model_profiles` empty**.
- `ollama-tailscale-proxy.py` running (PID 483188 at end of session — re-check, PIDs go stale). No models resident in Ollama.
- `eval/results/` is **gitignored** — the 16 result JSONs from this session are local-only and will not survive a fresh clone.

---

## What was measured (do not re-derive this)

Full evidence: `docs/superpowers/plans/2026-08-24-eval-measurement-confounds.md`. Published summary: the artifact at https://claude.ai/code/artifact/e5d2faca-5c81-4639-9e9b-7e59f5da19bd

Three published benchmark scores turned out to be measurement artifacts, not model properties:

- **`qwen3-vl:4b` 0% → 40%** — its 0% was a 23.5 s cold load against a 30 s read timeout, times three retries ≈ 90.6 s, matching the observed 91.7–97.4 s band exactly.
- **`qwen3.5:4b-mlx` → 100% TGC**, beating both hosted 8–9B models.
- **`lfm2.5:8b` 0% → 6.7%**, but its diagnosis held: 11 of 14 failures are genuine declines to call a tool.

Two findings that outlive this run:

1. **The budget gap is not local-hardware-specific.** Hosted `qwen/qwen3.5-9b` breached the shipped 30 s timeout on three calls (38.7 / 32.1 / 39.5 s). Always read `tgc` and `tgc_within_budget` together.
2. **`gemma4:e2b-mlx` and `llama3.2:3b` have no per-task pass/fail state, only a pass rate** — at n=45, 64.4% and 24.4%, with individual tasks splitting 5/9, 3/6, 1/9. Any n=1 claim about those two is one sample from a coin flip. Use `--repeat 3` minimum for them.

**Thinking mode is a per-model property, not a tier property.** `think=false` took `qwen3-vl:4b` from 40%→80% TGC and dropped `qwen3.5:4b-mlx`'s in-budget score from 73.3%→6.7%. The mechanism for the second is the interesting half: disabling thinking did not slow that model down directly — it made it emit worse-formed JSON, which sent `request_structured` into its retry loop (calls 57→89, truncations 4→12, eval time 6.2→20.3 min). For that model the reasoning channel was *how the answer got well formed on the first attempt*.

---

## Environment: read this before trusting any timing

`runtime/llm/clients/ollama_client.py` `_validate_local()` enforces a loopback host — but `127.0.0.1:11434` is served by `~/.aegis/ollama-tailscale-proxy.py`, a raw TCP forward to a **remote Mac** at `100.64.170.84`. Every "local" call crosses a machine boundary. The loopback guard does not mean what it appears to mean. This dev box is a Raspberry Pi 5 with 7 GB RAM and could not physically hold the 8.8 GB models being benchmarked.

Network is *not* a confound (RTT 1.24 ms) and the remote host was healthy (warm throughput 117.8 tok/s e2b / 78.3 e4b / 80.5 lfm2.5 / **15.0** qwen3-vl). But wall-clock on this setup is noisy: `llama3.2:3b` once showed a 3.5→13.3 min swing with an *identical* call count. Do not read timing differences as behaviour without checking `model_calls`.

---

## How to read the new eval output

The harness now distinguishes "the model could not do it" from "the harness cut it off" — the ambiguity that produced most of the original outliers.

- `failure_kind` per variant: `timeout_exhausted`, `thinking_budget_exhausted`, `no_tool_call`, `wrong_tool`, `tool_errored`, `incomplete_chain`, `repeated_step`. Classified from telemetry, never by parsing `reason`.
- `tgc_within_budget` alongside `tgc` — capability vs. the shipped 30 s timeout.
- `telemetry` per variant: `model_calls`, `load_ms_total`, `max_call_wall_ms`, `timed_out_calls`, `max_thinking_token_share`, `truncated_calls`.
- CLI flags: `--repeat N`, `--read-timeout` (eval default 300 s, production stays 30 s), `--no-prewarm`.

**Diagnostic shortcuts, learned the hard way:**
- Durations clustered near ~90–96 s ⇒ retry-exhausted timeouts, not model failure.
- Uniform ~5–6 s across *all* tasks ⇒ Tier-0 classifier gate rejection.
- `actual_calls` did not exist before `78f84e0`, so pre-2026-08-21 files show 0 calls because the field is **absent**, not because nothing was called.

---

## Known consequence, deliberately accepted

The eval harness builds its dispatcher through `bot.py`, so **benchmark runs inherit `modelProfiles`**. That is intentional — it measures each model at its intended config rather than through a second code path that does not match production — but it means eval results now depend on `config.json` state. Record the active profiles next to any published number.

---

## Open threads

1. **Article not updated.** `2026-08-20-aegis-medium-article-benchmark-results.md` (repo root) still contains the three superseded scores, and still asserts `llama3.2:3b` "pulled off a clean two-step chain" on `list_then_read` — at n=9 that task is 1/6, so the claim describes one sample from a coin flip. Whether to correct or retract is the user's call.
2. **No PR opened** for `feat/eval-measurement-instrumentation`.
3. **Profile fields are deliberately minimal.** `think` is the only one. `max_tokens` and retry policy are plausible next knobs — `qwen3.5:4b-mlx`'s regression ran through the retry loop — but both are unmeasured. Adding knobs on reasoning alone is the failure mode this whole line of work exists to stop; measure first.
4. **No family-prefix matching, on purpose.** `qwen3-vl:4b` and `qwen3.5:4b-mlx` share a vendor string and need opposite values. A `qwen*` fallback would hand one of them the other's measured-wrong setting while looking like sensible inheritance.
5. **Carried over, still open from 2026-08-21:** the classifier fails open on transport outage (`ModelBackedClassifier.classify()` returns `intent="unknown"` for a real outage, indistinguishable from a genuine miss — now reaches the full-catalog planner); and `search_then_read`'s path-guessing problem.
