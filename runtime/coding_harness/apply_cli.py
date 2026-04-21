"""Apply a drafted patch onto a fresh branch and run the test suite.

```text
python -m runtime.coding_harness.apply_cli CT-001
python -m runtime.coding_harness.apply_cli CT-001 --dry-run
python -m runtime.coding_harness.apply_cli CT-001 --no-tests
python -m runtime.coding_harness.apply_cli --status CT-001
```

For each invocation we locate the latest ``.patch.md`` for ``CT-NNN``
in the AEGIS workspace, pass it to the applier (which talks to a
``Runner`` — defaults to a real ``subprocess.run`` wrapper), write a
``.result.md`` next to the patch, and append one row to
``DECISIONS.md`` mapping the verdict onto the Phase-3 enum.

Exit codes: ``0`` on ``applied_clean`` (or successful ``--status`` /
``--dry-run``), ``1`` otherwise.
"""
from __future__ import annotations

import argparse
import os

# Bandit: B404 is intentional — required to invoke git/make for the apply flow.
import subprocess  # nosec
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from runtime.config import AegisConfig, get_config
from runtime.events import EventStream
from runtime.improvement.decisions import ApplierVerdict, record_decision

from .applier import GitResult, Runner, _check_preconditions, _extract_unified_diff, apply_patch
from .apply_outcome import ApplyOutcome
from .patch_writer import existing_drafts_for
from .result_writer import latest_result_for, write_result

_DEFAULT_TEST_TIMEOUT_S = 300.0
_TIMEOUT_ENV = "MAKE_TEST_TIMEOUT_SECS"


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _to_str(v: str | bytes | None) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def real_runner(
    argv: Sequence[str],
    *,
    stdin: str | None = None,
    timeout: float | None = None,
) -> GitResult:
    """Default ``Runner`` implementation. Wraps ``subprocess.run``.

    Never raises on non-zero exit. ``TimeoutExpired`` is caught and
    surfaced via ``GitResult(timed_out=True)``.
    """
    start = time.monotonic()
    try:
        result = subprocess.run(  # noqa: S603  # nosec
            list(argv),
            input=stdin,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return GitResult(
            exit_code=-1,
            stdout=_to_str(e.stdout),
            stderr=_to_str(e.stderr),
            duration_s=time.monotonic() - start,
            timed_out=True,
        )
    return GitResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=time.monotonic() - start,
    )


def _resolve_test_timeout() -> float:
    raw = os.environ.get(_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_TEST_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TEST_TIMEOUT_S


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg = get_config()
    return run(
        cfg,
        ct_id=args.ct_id,
        repo_root=Path(args.repo_root) if args.repo_root else Path.cwd(),
        status=args.status,
        dry_run=args.dry_run,
        run_tests=not args.no_tests,
        auto_revert=not args.no_revert,
        runner=real_runner,
        clock=_now,
    )


def run(
    cfg: AegisConfig,
    *,
    ct_id: str,
    repo_root: Path,
    status: bool,
    dry_run: bool,
    run_tests: bool,
    runner: Runner,
    clock: Callable[[], datetime],
    auto_revert: bool = True,
) -> int:
    if status:
        return _do_status(cfg.aegis_home, ct_id)

    drafts = existing_drafts_for(cfg.aegis_home, ct_id)
    if not drafts:
        print(
            f"[apply] error: no .patch.md found for {ct_id} "
            f"(run `make harness ARGS=\"--task {ct_id}\"` first)",
            file=sys.stderr,
        )
        return 1
    patch_path = drafts[-1]
    print(f"[apply] {ct_id} — patch {patch_path.name}")

    if dry_run:
        return _do_dry_run(repo_root, patch_path, runner=runner)

    outcome = apply_patch(
        repo_root,
        patch_path,
        runner=runner,
        clock=clock,
        run_tests=run_tests,
        test_timeout_secs=_resolve_test_timeout(),
        auto_revert=auto_revert,
    )
    result_path = write_result(cfg.aegis_home, outcome)
    events = EventStream(cfg.storage.sessions_dir)
    record_decision(
        cfg.aegis_home,
        imp_id=outcome.imp_id,
        verdict=_as_applier_verdict(outcome.verdict),
        rationale=_rationale_for(outcome),
        events=events,
        when=outcome.applied_at,
    )
    _print_outcome(outcome, result_path)
    return 0 if outcome.verdict == "applied_clean" else 1


def _as_applier_verdict(v: str) -> ApplierVerdict:
    """Map an ``ApplyOutcome.verdict`` onto the decisions ``ApplierVerdict``.

    They overlap on three values; ``precondition_failed`` records as
    ``apply_conflict`` so the audit log only carries the four enum values
    declared in ``decisions.py``.
    """
    if v in ("applied_clean", "applied_test_failed", "apply_conflict"):
        return v  # type: ignore[return-value]
    return "apply_conflict"


def _rationale_for(outcome: ApplyOutcome) -> str:
    bits: list[str] = []
    if outcome.branch:
        bits.append(f"branch {outcome.branch}")
    if outcome.tests_exit_code is not None:
        dur = (
            f"{outcome.tests_duration_s:.1f}s"
            if outcome.tests_duration_s is not None
            else "—"
        )
        bits.append(f"tests exit {outcome.tests_exit_code} in {dur}")
    if outcome.reason:
        bits.append(outcome.reason)
    return "; ".join(bits) or outcome.verdict


def _print_outcome(outcome: ApplyOutcome, result_path: Path) -> None:
    print(f"[apply] verdict: {outcome.verdict}")
    if outcome.branch:
        print(f"[apply] branch: {outcome.branch}")
    if outcome.tests_exit_code is not None:
        dur = (
            f"{outcome.tests_duration_s:.1f}s"
            if outcome.tests_duration_s is not None
            else "—"
        )
        print(f"[apply] tests: exit {outcome.tests_exit_code} in {dur}")
    if outcome.reason:
        print(f"[apply] note: {outcome.reason}")
    print(f"[apply] wrote {result_path}")


def _do_status(workspace: Path, ct_id: str) -> int:
    drafts = existing_drafts_for(workspace, ct_id)
    if not drafts:
        print(f"[apply] no patches found for {ct_id}")
        return 1
    latest = latest_result_for(workspace, ct_id)
    print(f"[apply] {ct_id}: {len(drafts)} patch(es), latest {drafts[-1].name}")
    if latest is None:
        print("[apply] no apply attempts recorded yet.")
        return 0
    print(f"[apply] latest result: {latest.name}")
    print()
    print(latest.read_text(encoding="utf-8"))
    return 0


def _do_dry_run(
    repo_root: Path,
    patch_path: Path,
    *,
    runner: Runner,
) -> int:
    """Check preconditions + ``git apply --check``. No branch, no tests."""
    text = patch_path.read_text(encoding="utf-8")
    diff = _extract_unified_diff(text)
    if diff is None:
        print(
            "[apply] dry-run: refused — patch has no applicable diff",
            file=sys.stderr,
        )
        return 1
    fail, _branch = _check_preconditions(repo_root, git=runner)
    if fail is not None:
        print(f"[apply] dry-run: refused — {fail}", file=sys.stderr)
        return 1
    res = runner(
        ["git", "-C", str(repo_root), "apply", "--check"],
        stdin=diff,
    )
    if res.exit_code != 0:
        msg = (res.stderr or res.stdout).strip()
        print(f"[apply] dry-run: would CONFLICT — {msg}", file=sys.stderr)
        return 1
    print("[apply] dry-run: would apply cleanly (no branch created, tests not run)")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="apply-cli")
    p.add_argument(
        "ct_id",
        metavar="CT-NNN",
        help="Coding-task id whose latest patch should be applied.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preconditions + `git apply --check` only. No branch, no tests.",
    )
    p.add_argument(
        "--no-tests",
        action="store_true",
        help="Apply onto the branch but skip `make test`. Verdict = applied_clean.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print the latest .result.md for CT-NNN and exit (no apply).",
    )
    p.add_argument(
        "--no-revert",
        action="store_true",
        help=(
            "Do NOT auto-revert on test failure or apply conflict. Leaves the "
            "broken branch checked out for forensic inspection (Phase 5 behaviour)."
        ),
    )
    p.add_argument(
        "--repo-root",
        default=None,
        metavar="PATH",
        help="Git repo to apply onto. Defaults to the current working directory.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
