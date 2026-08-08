"""Apply a drafted patch onto a fresh branch and run the test suite.

Plane-3 only. The applier MAY shell out to ``git`` and ``make`` but it
MUST refuse to operate on protected branches or a dirty working tree,
and it NEVER pushes. ``apply_patch`` returns an ``ApplyOutcome`` for
every code path — subprocess failures are verdicts, not exceptions.

The orchestrator never spawns subprocesses directly; it talks to a
``Runner`` (a small protocol). The default real-subprocess runner
lives next to the CLI so unit tests can swap in a scripted fake.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, Protocol

from .apply_outcome import ApplyOutcome

_DEFAULT_PROTECTED: Final[tuple[str, ...]] = ("main", "master", "staging")
_PROTECTED_ENV: Final[str] = "AEGIS_PROTECTED_BRANCHES"
_MAX_TAIL: Final[int] = 8192
_REASON_CAP: Final[int] = 400


@dataclass(frozen=True, slots=True)
class GitResult:
    """One subprocess invocation result. Used by every Runner."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False


class Runner(Protocol):
    """Subprocess executor abstraction. Real impl lives in apply_cli."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> GitResult: ...


@dataclass(frozen=True, slots=True)
class FakeGitRunner:
    """Test double for ``_check_preconditions`` — keyed on full argv tuple."""

    responses: dict[tuple[str, ...], GitResult] = field(default_factory=dict)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> GitResult:
        argv_t = tuple(argv)
        for length in range(len(argv_t), 0, -1):
            key = argv_t[:length]
            if key in self.responses:
                return self.responses[key]
        return GitResult(exit_code=0)


@dataclass
class ScriptedRunner:
    """Test double for ``apply_patch`` — first matching script entry wins.

    Records every call (argv + stdin) so tests can assert on order and
    inputs. Unmatched calls fall through to ``GitResult(exit_code=0)``.
    """

    script: list[tuple[Callable[[Sequence[str]], bool], GitResult]] = field(
        default_factory=list,
    )
    calls: list[tuple[tuple[str, ...], str | None]] = field(default_factory=list)

    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float | None = None,
    ) -> GitResult:
        argv_t = tuple(argv)
        self.calls.append((argv_t, stdin))
        for matcher, resp in self.script:
            if matcher(argv_t):
                return resp
        return GitResult(exit_code=0)


def has(*tokens: str) -> Callable[[Sequence[str]], bool]:
    """Argv matcher: all ``tokens`` appear (in any order)."""
    needles = tuple(tokens)

    def _match(argv: Sequence[str]) -> bool:
        argv_set = set(argv)
        return all(t in argv_set for t in needles)

    return _match


def _protected_branches() -> tuple[str, ...]:
    raw = os.environ.get(_PROTECTED_ENV, "")
    if not raw.strip():
        return _DEFAULT_PROTECTED
    return tuple(b.strip() for b in raw.split(",") if b.strip())


def _check_preconditions(repo_root: Path, *, git: Runner) -> tuple[str | None, str]:
    """Returns ``(reason, current_branch)``.

    ``reason`` is ``None`` when it's safe to apply. ``current_branch`` is the
    detected branch (used by Phase 6 auto-revert to restore HEAD on failure)
    or ``""`` if it could not be determined.
    """
    res = git(["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"])
    if res.exit_code != 0:
        return f"not a git repository: {repo_root}", ""

    res = git(["git", "-C", str(repo_root), "status", "--porcelain"])
    if res.exit_code != 0:
        return f"git status failed: {(res.stderr or res.stdout).strip()}", ""
    if res.stdout.strip():
        return "working tree dirty (uncommitted changes)", ""

    res = git(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    if res.exit_code != 0:
        return (
            f"could not determine current branch: {(res.stderr or res.stdout).strip()}",
            "",
        )
    branch = res.stdout.strip()
    if branch in _protected_branches():
        return f"refusing to apply on protected branch {branch!r}", branch

    return None, branch


def _revert_apply(
    repo_root: Path,
    *,
    git: Runner,
    original_branch: str,
    apply_branch: str,
) -> str | None:
    """Restore HEAD to ``original_branch`` and delete ``apply_branch``.

    Sequence:
      1. ``git reset --hard HEAD`` on the apply branch — discards the
         unstaged ``git apply`` changes so the next checkout can succeed.
      2. ``git checkout <original_branch>`` — leaves the user where they
         started.
      3. ``git branch -D <apply_branch>`` — drops the now-empty branch.

    Returns ``None`` on success, or a short error string on partial
    failure (the verdict is unchanged; the failure surfaces in
    ``ApplyOutcome.reason``).
    """
    if not original_branch or not apply_branch:
        return "missing branch context for revert"

    res = git(["git", "-C", str(repo_root), "reset", "--hard", "HEAD"])
    if res.exit_code != 0:
        return f"reset failed: {(res.stderr or res.stdout).strip()}"

    res = git(["git", "-C", str(repo_root), "checkout", original_branch])
    if res.exit_code != 0:
        return f"checkout failed: {(res.stderr or res.stdout).strip()}"

    res = git(["git", "-C", str(repo_root), "branch", "-D", apply_branch])
    if res.exit_code != 0:
        return f"branch -D failed: {(res.stderr or res.stdout).strip()}"

    return None


_DIFF_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"## Unified diff\n```diff\n(.*?)\n```",
    re.DOTALL,
)
_REFUSED_RE: Final[re.Pattern[str]] = re.compile(r"## Unified diff\n_Refused\b")
_STUB_RE: Final[re.Pattern[str]] = re.compile(r"## Unified diff\n_Stub\b")


def _extract_unified_diff(patch_md_text: str) -> str | None:
    """Pull the unified diff out of a ``.patch.md``.

    Always returns a trailing newline — ``git apply`` rejects a patch
    whose final line lacks one (``error: corrupt patch at line N``).
    """
    if _REFUSED_RE.search(patch_md_text) or _STUB_RE.search(patch_md_text):
        return None
    m = _DIFF_FENCE_RE.search(patch_md_text)
    if not m:
        return None
    diff = m.group(1).strip()
    if not diff:
        return None
    return diff + "\n"


_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(CT-\d{3,5})__(IMP-[a-f0-9]{8})__\d{4}-\d{2}-\d{2}T\d{4}Z(?:-\d{2})?\.patch\.md$",
)


def _parse_patch_filename(name: str) -> tuple[str, str]:
    m = _FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"unrecognised patch filename: {name!r}")
    return m.group(1), m.group(2)


def _branch_for(ct_id: str, imp_id: str) -> str:
    return f"aegis/{ct_id}-{imp_id.removeprefix('IMP-')}"


def _tail(text: str, limit: int = _MAX_TAIL) -> str:
    """Last ``limit`` chars of ``text``, snapped to a line boundary."""
    if len(text) <= limit:
        return text
    snippet = text[-limit:]
    nl = snippet.find("\n")
    return snippet[nl + 1 :] if nl >= 0 else snippet


def _fail(
    *,
    ct_id: str,
    imp_id: str,
    verdict: str,
    reason: str,
    branch: str,
    patch_path: Path,
    applied_at: datetime,
    tests_exit_code: int | None = None,
    tests_duration_s: float | None = None,
    tests_stdout_tail: str = "",
) -> ApplyOutcome:
    return ApplyOutcome(
        ct_id=ct_id,
        imp_id=imp_id,
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason[:_REASON_CAP],
        branch=branch,
        patch_path=str(patch_path),
        tests_exit_code=tests_exit_code,
        tests_duration_s=tests_duration_s,
        tests_stdout_tail=_tail(tests_stdout_tail),
        applied_at=applied_at,
    )


def apply_patch(
    repo_root: Path,
    patch_path: Path,
    *,
    runner: Runner,
    clock: Callable[[], datetime],
    run_tests: bool = True,
    test_timeout_secs: float = 300.0,
    auto_revert: bool = True,
) -> ApplyOutcome:
    """Apply ``patch_path`` on a fresh branch and return the outcome.

    Order: parse filename → read patch → extract diff → preconditions →
    pre-flight branch availability → ``git apply --check`` → create
    branch → ``git apply`` → run ``make test`` (unless skipped).
    Subprocess failures NEVER raise.

    When ``auto_revert=True`` (default) and the verdict is
    ``applied_test_failed`` or ``apply_conflict`` *after* the apply
    branch was created, the original branch is restored and the apply
    branch is deleted. The verdict is preserved either way; revert
    failure is appended to ``reason`` as a soft warning.
    """
    ct_id, imp_id = _parse_patch_filename(patch_path.name)
    branch = _branch_for(ct_id, imp_id)
    now = clock()

    try:
        patch_text = patch_path.read_text(encoding="utf-8")
    except OSError as e:
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="precondition_failed",
            reason=f"could not read patch: {e}",
            branch="", patch_path=patch_path, applied_at=now,
        )

    diff = _extract_unified_diff(patch_text)
    if diff is None:
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="precondition_failed",
            reason="patch contains no applicable diff (refused/stub or empty)",
            branch="", patch_path=patch_path, applied_at=now,
        )

    fail_reason, original_branch = _check_preconditions(repo_root, git=runner)
    if fail_reason is not None:
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="precondition_failed",
            reason=fail_reason,
            branch="", patch_path=patch_path, applied_at=now,
        )

    res = runner(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"refs/heads/{branch}"],
    )
    if res.exit_code == 0:
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="precondition_failed",
            reason=f"branch {branch} already exists; discard with: git branch -D {branch}",
            branch="", patch_path=patch_path, applied_at=now,
        )

    res = runner(
        ["git", "-C", str(repo_root), "apply", "--check"],
        stdin=diff,
    )
    if res.exit_code != 0:
        # No branch was created — nothing to revert.
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="apply_conflict",
            reason=f"git apply --check failed: {(res.stderr or res.stdout).strip()}",
            branch="", patch_path=patch_path, applied_at=now,
        )

    res = runner(["git", "-C", str(repo_root), "checkout", "-b", branch])
    if res.exit_code != 0:
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="precondition_failed",
            reason=f"could not create branch {branch}: {(res.stderr or res.stdout).strip()}",
            branch="", patch_path=patch_path, applied_at=now,
        )

    res = runner(["git", "-C", str(repo_root), "apply"], stdin=diff)
    if res.exit_code != 0:
        reason = f"git apply failed after --check passed: {(res.stderr or res.stdout).strip()}"
        if auto_revert:
            reason = _maybe_revert(
                repo_root, runner=runner,
                original_branch=original_branch, apply_branch=branch,
                base_reason=reason,
            )
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="apply_conflict",
            reason=reason,
            branch=branch, patch_path=patch_path, applied_at=now,
        )

    if not run_tests:
        return ApplyOutcome(
            ct_id=ct_id, imp_id=imp_id,
            verdict="applied_clean",
            reason="tests skipped (--no-tests)",
            branch=branch, patch_path=str(patch_path),
            tests_exit_code=None, tests_duration_s=None, tests_stdout_tail="",
            applied_at=now,
        )

    test_res = runner(
        ["make", "-C", str(repo_root), "test"],
        timeout=test_timeout_secs,
    )
    combined = test_res.stdout
    if test_res.stderr:
        combined = (combined + "\n" + test_res.stderr) if combined else test_res.stderr

    if test_res.timed_out:
        reason = f"tests timed out after {test_timeout_secs:.1f}s"
        if auto_revert:
            reason = _maybe_revert(
                repo_root, runner=runner,
                original_branch=original_branch, apply_branch=branch,
                base_reason=reason,
            )
        return _fail(
            ct_id=ct_id, imp_id=imp_id,
            verdict="applied_test_failed",
            reason=reason,
            branch=branch, patch_path=patch_path, applied_at=now,
            tests_exit_code=None,
            tests_duration_s=test_res.duration_s or test_timeout_secs,
            tests_stdout_tail=combined,
        )

    if test_res.exit_code == 0:
        return ApplyOutcome(
            ct_id=ct_id, imp_id=imp_id,
            verdict="applied_clean",
            branch=branch, patch_path=str(patch_path),
            tests_exit_code=0,
            tests_duration_s=test_res.duration_s,
            tests_stdout_tail=_tail(combined),
            applied_at=now,
        )

    reason = f"make test exited {test_res.exit_code}"
    if auto_revert:
        reason = _maybe_revert(
            repo_root, runner=runner,
            original_branch=original_branch, apply_branch=branch,
            base_reason=reason,
        )
    return _fail(
        ct_id=ct_id, imp_id=imp_id,
        verdict="applied_test_failed",
        reason=reason,
        branch=branch, patch_path=patch_path, applied_at=now,
        tests_exit_code=test_res.exit_code,
        tests_duration_s=test_res.duration_s,
        tests_stdout_tail=combined,
    )


def _maybe_revert(
    repo_root: Path,
    *,
    runner: Runner,
    original_branch: str,
    apply_branch: str,
    base_reason: str,
) -> str:
    """Run revert; append a suffix to ``base_reason`` describing the result."""
    err = _revert_apply(
        repo_root, git=runner,
        original_branch=original_branch, apply_branch=apply_branch,
    )
    if err is None:
        return f"{base_reason}; auto-reverted to {original_branch}"
    return f"{base_reason}; auto-revert failed: {err}"
