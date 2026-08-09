# Maintaining & releasing AEGIS

The repeatable loop for changing, verifying, and releasing AEGIS. It matches
how the codebase is actually built — solo operator, local-model-first, tests
as the contract — so follow it rather than a generic GitHub flow.

## TL;DR

```
branch  →  change test-first  →  tests green + no new lint/type debt  →  PR  →  merge  →  tag a release
```

The gate is: **`make test-unit` and `make test-e2e` pass, and you introduce no
new `ruff` or `mypy` findings** on the files you touched. See §1.3 — `make
test` also runs `make lint`, which currently reports ~109 pre-existing findings
(accumulated debt), so `make test` is not green today; treat the pytest suite
plus baseline parity as the real bar until that debt is driven to zero.

---

## 1. The change lifecycle

### 1.1 Branch — never commit straight to `main`

```bash
git checkout main && git pull --ff-only
git checkout -b <kind>/<short-slug>     # feat/…  fix/…  docs/…  refactor/…
```

For anything larger than a one-file tweak, work in an **isolated worktree** so
the main checkout stays usable and parallel agents don't collide:

```bash
git worktree add .claude/worktrees/<slug> -b <kind>/<slug>
cd .claude/worktrees/<slug>
uv sync --extra dev        # each worktree needs its own venv
```

### 1.2 Change it test-first

AEGIS's reliability comes from tests, not from trusting the model. Every
behavioural change lands with a test:

1. Write the failing test first; run it and confirm it fails *for the right
   reason* (not an import error).
2. Write the minimal code to pass it.
3. Keep files focused — one responsibility each; split before they sprawl.

Reliability-sensitive code (the harness dispatcher, the gates, the tool
sandbox, config coercion) deserves an adversarial test, not just a happy-path
one: what happens on malformed input, a failed send, a non-zero exit, a path
that escapes the sandbox?

### 1.3 Run the full gate — always the whole suite

```bash
make test-unit && make test-e2e && make type    # the effective gate today
```

Run the **whole** suite, never just your new test file — sibling guard tests
(e.g. `test_files_harness.py`) break silently otherwise.

`make test` bundles `lint type test-unit test-e2e`, but `make lint`
(`ruff check .`) currently exits non-zero on ~109 pre-existing findings, so
`make test` as a whole is red regardless of your change. Until that debt is
cleaned up (a worthwhile near-term task — then `make test` becomes a true
one-command gate and a CI job can enforce it), the honest bar is: **pytest
green, and no *new* ruff/mypy findings on your files** (baseline parity). The
individual targets:

| Command | What it checks |
|---|---|
| `make lint` | `ruff check .` + `bandit -q -r runtime memory scripts -x tests` |
| `make type` | `mypy --strict runtime memory scripts` |
| `make test-unit` | `pytest -m "unit or not e2e" tests/` |
| `make test-e2e` | `pytest -m e2e tests/` |
| `make security` | bandit + `semgrep --config auto` |

The repo carries some pre-existing lint/type findings. The bar for a change is
**baseline parity** — introduce no *new* ruff or mypy findings on the files you
touch. To compare against the baseline without disturbing your tree, read the
old version out of git rather than stashing:

```bash
git show HEAD:path/to/file.py > /tmp/base.py && .venv/bin/ruff check /tmp/base.py
```

Do **not** use `git stash` / `rebase` / `merge` to set work aside — the stash
stack is shared across every worktree and another session can pop it. Make a
WIP commit instead.

### 1.4 Commit in small, honest steps

```bash
git add <exact files>
git commit -m "feat(harness): <what changed and why>"
```

Conventional prefixes (`feat` / `fix` / `docs` / `refactor` / `test` /
`chore`). Commit the code and its tests together. If you discover the *plan*
was wrong, fix the plan doc in the same PR — the `docs/PLAN_PHASE_*.md` files
are meant to stay a faithful record, not a historical fiction.

### 1.5 PR and merge

```bash
git push -u origin <branch>
gh pr create --base main --title "…" --body "…"   # Summary + Test Plan
```

CI runs `make lint` + unit + e2e on every PR (see §4); mypy is not yet
blocking, so **the PR body's test plan still carries evidence CI can't** —
note the gate result and check off the manual smoke steps automation can't
cover (below). Merge with `gh pr merge <n> --merge --delete-branch` (or the
GitHub UI). Then:

```bash
git checkout main && git pull --ff-only
git worktree remove .claude/worktrees/<slug>    # if you used one
git branch -d <branch>
```

### 1.6 The one thing tests can't cover: the live smoke

The unit + e2e suites never exercise the real Telegram round-trip. After a
change that touches the chat pipeline, harness, tools, or gates, run the bot
and drive it by hand:

```bash
tmux new -s aegis '.venv/bin/python -m runtime.serve'   # NOT launchd — macOS TCC
```

Send the operator flows the change affects (a guarded write with a "yes"
confirmation, a `run_command`, a `task_complete` turn where a tool fails) and
confirm the ledger recorded them (`load_tool_calls`). Launchd-spawned runs hang
on `~/Desktop` reads under macOS TCC with no UI — always use tmux.

---

## 2. Versioning

AEGIS follows [semver](https://semver.org/) once it hits `0.1.0`; while on
`0.0.x` it is explicitly pre-contract (schemas, tool ids, and gate patterns
change freely — the README says so). The version lives in one place:

```toml
# pyproject.toml
version = "0.0.1"
```

Bump it as the **last commit before a release tag**, matched to the change:

| Change | Bump | Example |
|---|---|---|
| Breaking: config/schema/tool-id change that an existing setup would notice | minor while 0.0.x, major at ≥1.0 | 0.0.1 → 0.1.0 |
| New capability, backward-compatible | patch while 0.0.x, minor at ≥1.0 | 0.0.1 → 0.0.2 |
| Fix / docs / internal only | patch | 0.0.2 → 0.0.3 |

Phase 11 (guarded write, run_command, evidence ledger, completion gate) is a
new-capability release: **0.0.1 → 0.0.2**.

---

## 3. Cutting a release

A release is a git tag plus a GitHub release with notes. Do it from a clean,
green `main`:

```bash
# 1. main is merged, pulled, and green (see §1.3 on what "green" means today)
git checkout main && git pull --ff-only && make test-unit && make test-e2e && make type

# 2. bump the version + update the changelog (see §3.1), commit
git commit -am "chore(release): 0.0.2"

# 3. tag and push
git tag -a v0.0.2 -m "AEGIS 0.0.2 — capability floor + evidence ledger"
git push origin main --tags

# 4. GitHub release with notes generated from the merged PRs
gh release create v0.0.2 --title "v0.0.2 — capability floor + evidence ledger" --generate-notes
```

There is nothing to publish to a package index — AEGIS is run from source, not
`pip install`ed — so the tag + GitHub release *is* the release. `make setup` on
the tag reproduces a working install.

### 3.1 CHANGELOG

Keep a `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/)
style — an `## [Unreleased]` section you append to as you merge PRs, renamed to
`## [x.y.z] - YYYY-MM-DD` at release time. It's the human-readable companion to
`git log`; the release notes draw from it.

---

## 4. What's deliberately not here yet

These are known gaps, not oversights — add them when the cost of doing without
them exceeds the cost of maintaining them:

- **CI runs a partial gate.** `.github/workflows/ci.yml` runs `make lint`
  (ruff + bandit), `make test-unit`, and `make test-e2e` as **blocking** checks
  on every PR and push to `main` — these are all green. `make type`
  (mypy --strict) runs **non-blocking** because ~7 pre-existing errors remain
  (optional-dependency typing in `serve.py`, defensive guards in the
  dispatcher). Clearing those and removing `continue-on-error` from the mypy
  step turns the whole `make test` into an enforced gate — the highest-value
  next cleanup.
- **No required reviewers.** Solo repo — you approve your own work. The
  discipline substitute is the branch → PR → self-review flow (and, for larger
  work, the multi-agent review pattern below), not a rubber-stamp merge.
- **Issues live in-repo**, as markdown under `.scratch/<feature>/`, not GitHub
  Issues (see `docs/agents/`). Design decisions live in `docs/adr/` and phase
  plans in `docs/PLAN_PHASE_*.md`.

---

## 5. For larger work: plan → multi-agent build

Anything phase-sized (like Phase 11) is worth the heavier loop that produced
this codebase:

1. **Write the plan first** — `docs/PLAN_PHASE_N_<name>.md`: goal, non-goals,
   lettered design items with real signatures, a test plan, a rollout sequence,
   design pins, and open questions. Correct the plan when a build step proves it
   wrong.
2. **Build task-by-task in a worktree**, each task: implement test-first →
   independent spec review (did it build exactly the spec?) → independent
   quality review (is it well-built, and does it survive adversarial input?) →
   fix loop → next task.
3. **Finish with a whole-branch review** for the cross-cutting bugs that live in
   the *seams* between tasks — the ones per-task review can't see.
4. Track deferred items as numbered open questions in the plan doc so nothing
   silently drops (e.g. the ledger-idempotency item that gates turning the
   completion gate from annotate to hard-block).

This is slower per change but it's what keeps a reliability harness actually
reliable: every claim gets verified by something other than the thing that made
the claim.
