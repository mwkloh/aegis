"""Render an `ApplyOutcome` to a `.result.md` file under the workspace.

Output naming: `CT-NNN__IMP-xxxxxxxx__YYYY-MM-DDTHHMMZ.result.md`.
Sits alongside the `.patch.md` it corresponds to. Append-only — a
re-apply at a later timestamp produces a new file (or a `-NN` suffix
if same minute).
"""
from __future__ import annotations

from datetime import UTC
from pathlib import Path

from .apply_outcome import ApplyOutcome
from .patch_writer import diffs_dir

_VERDICT_BLURB: dict[str, str] = {
    "applied_clean": "patch applied cleanly and the test suite passed",
    "applied_test_failed": "patch applied, but the test suite failed",
    "apply_conflict": "`git apply` rejected the patch",
    "precondition_failed": "apply was refused before touching anything",
}


def result_filename(outcome: ApplyOutcome) -> str:
    ts = outcome.applied_at.astimezone(UTC).strftime("%Y-%m-%dT%H%MZ")
    return f"{outcome.ct_id}__{outcome.imp_id}__{ts}.result.md"


def existing_results_for(workspace: Path, ct_id: str) -> list[Path]:
    """All prior `.result.md` records for a CT, oldest first."""
    return sorted(diffs_dir(workspace).glob(f"{ct_id}__*.result.md"))


def latest_result_for(workspace: Path, ct_id: str) -> Path | None:
    results = existing_results_for(workspace, ct_id)
    return results[-1] if results else None


def write_result(workspace: Path, outcome: ApplyOutcome) -> Path:
    """Write a `.result.md`. Adds a `-NN` suffix on minute-collision."""
    base = result_filename(outcome)
    target = diffs_dir(workspace) / base
    if target.exists():
        stem = base.removesuffix(".result.md")
        n = 2
        while True:
            candidate = diffs_dir(workspace) / f"{stem}-{n:02d}.result.md"
            if not candidate.exists():
                target = candidate
                break
            n += 1
    target.write_text(_render(outcome), encoding="utf-8")
    return target


def _render(outcome: ApplyOutcome) -> str:
    ts = outcome.applied_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")
    blurb = _VERDICT_BLURB.get(outcome.verdict, outcome.verdict)
    head = (
        f"# {outcome.ct_id} — {outcome.imp_id} (verdict: {outcome.verdict})\n\n"
        f"_{blurb}._\n\n"
        f"- **Applied:** {ts}\n"
        f"- **Branch:** {outcome.branch or '—'}\n"
        f"- **Patch:** {outcome.patch_path or '—'}\n"
    )
    head += _tests_line(outcome) + "\n"
    if outcome.reason:
        head += f"\n## Reason\n{outcome.reason}\n"
    head += "\n## Test output (tail)\n"
    if outcome.tests_stdout_tail:
        head += "```text\n" + outcome.tests_stdout_tail.rstrip() + "\n```\n"
    else:
        head += "_no output captured_\n"
    head += "\n## Next steps\n" + _next_steps(outcome)
    return head


def _tests_line(outcome: ApplyOutcome) -> str:
    if outcome.tests_exit_code is None and outcome.tests_duration_s is None:
        return "- **Tests:** not run\n"
    exit_str = "—" if outcome.tests_exit_code is None else str(outcome.tests_exit_code)
    dur = outcome.tests_duration_s
    dur_str = "—" if dur is None else f"{dur:.1f}s"
    return f"- **Tests:** exit {exit_str} in {dur_str}\n"


def _next_steps(outcome: ApplyOutcome) -> str:
    branch = outcome.branch or "<branch>"
    if outcome.verdict == "applied_clean":
        return (
            f"- Review: `git diff main...{branch}`\n"
            f"- Ship: merge or PR the branch\n"
            f"- Discard: `git branch -D {branch}`\n"
        )
    if outcome.verdict == "applied_test_failed":
        return (
            f"- Inspect failures in the test output above\n"
            f"- Iterate on `{branch}` manually, or `git branch -D {branch}` to discard\n"
            f"- Re-draft the CT with `make harness ARGS=\"--task {outcome.ct_id} --force\"`\n"
        )
    if outcome.verdict == "apply_conflict":
        return (
            f"- Inspect: `git status` (branch was created but apply failed mid-way "
            f"or was rejected up-front)\n"
            f"- Re-draft with current HEAD: "
            f"`make harness ARGS=\"--task {outcome.ct_id} --force\"`\n"
            f"- Discard: `git branch -D {branch}` (after switching off it)\n"
        )
    return "- Resolve the precondition flagged above, then re-run `make apply`.\n"
