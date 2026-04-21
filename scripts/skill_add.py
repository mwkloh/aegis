"""`aegis-skill-add` — operator-facing skill staging CLI.

Phase 8 §C5. Given a path to a YAML descriptor, stage it into
``~/.aegis/skills_staging/`` (or an override via ``--staging-dir``),
printing the scanner findings in a form operators can review before
running ``/skills confirm <id>`` from Telegram — or ``--confirm`` from
the same CLI to install locally without round-tripping through the bot.

The staging path is deliberately separate from the active catalog so
a freshly-pasted LLM-generated descriptor can never reach the loader
without one human pass across the scanner output. Blocking findings
(shell binary as argv[0], URL in argv without ``allow_net``) cause a
non-zero exit; warn/info findings are shown but permit staging.

Exit codes:
  * 0 — staged or confirmed successfully (warnings allowed).
  * 1 — rejected by the scanner or validator.
  * 2 — bad invocation (missing path, unreadable source, etc.).

Keep this script thin. The heavy lifting lives in
``runtime.skills.installer`` and ``runtime.skills.scanner`` so the
Telegram slash handler shares one authoritative code path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runtime.skills.installer import (
    InstallOutcome,
    confirm_skill,
    list_staged,
    stage_skill,
)
from runtime.skills.scanner import ScanFinding

DEFAULT_STAGING = Path.home() / ".aegis" / "skills_staging"
DEFAULT_CATALOG = Path.home() / ".aegis" / "skills"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-skill-add",
        description="Stage (and optionally confirm) an AEGIS skill descriptor.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="Path to the YAML descriptor to stage.",
    )
    parser.add_argument(
        "--confirm",
        metavar="SKILL_ID",
        help="Move an already-staged skill into the active catalog.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the ids currently staged and exit.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING,
        help=f"Staging directory (default: {DEFAULT_STAGING}).",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Active catalog directory (default: {DEFAULT_CATALOG}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        return _cmd_list(args.staging_dir, as_json=args.json)
    if args.confirm:
        return _cmd_confirm(
            args.confirm,
            staging_dir=args.staging_dir,
            catalog_dir=args.catalog_dir,
            as_json=args.json,
        )
    if args.source is None:
        print(
            "error: provide a source path, or use --confirm <id> / --list",
            file=sys.stderr,
        )
        return 2
    return _cmd_stage(args.source, staging_dir=args.staging_dir, as_json=args.json)


# --- subcommands -------------------------------------------------------------


def _cmd_list(staging_dir: Path, *, as_json: bool) -> int:
    staged = list_staged(staging_dir)
    if as_json:
        print(json.dumps({"staged": staged}))
        return 0
    if not staged:
        print(f"No skills staged in {staging_dir}.")
        return 0
    print(f"Staged in {staging_dir}:")
    for skill_id in staged:
        print(f"  • {skill_id}")
    return 0


def _cmd_stage(source: Path, *, staging_dir: Path, as_json: bool) -> int:
    outcome = stage_skill(source, staging_dir=staging_dir)
    if as_json:
        print(json.dumps(_outcome_to_json(outcome)))
    else:
        _print_stage_outcome(outcome)
    return 0 if outcome.is_success() else 1


def _cmd_confirm(
    skill_id: str, *, staging_dir: Path, catalog_dir: Path, as_json: bool
) -> int:
    outcome = confirm_skill(
        skill_id, staging_dir=staging_dir, catalog_dir=catalog_dir
    )
    if as_json:
        print(json.dumps(_outcome_to_json(outcome)))
    else:
        _print_confirm_outcome(outcome)
    return 0 if outcome.is_success() else 1


# --- human-readable renderers ------------------------------------------------


def _print_stage_outcome(outcome: InstallOutcome) -> None:
    if outcome.verdict == "staged" and outcome.stage_path is not None:
        print(f"✓ Staged {outcome.skill_id!r} → {outcome.stage_path}")
        _print_findings(outcome.findings)
        print("")
        print(f"Next: /skills confirm {outcome.skill_id}")
        print(f"   or aegis-skill-add --confirm {outcome.skill_id}")
        return
    print(f"✗ {outcome.verdict}: {outcome.error or '(no detail)'}")
    _print_findings(outcome.findings)


def _print_confirm_outcome(outcome: InstallOutcome) -> None:
    if outcome.verdict == "confirmed" and outcome.final_path is not None:
        print(f"✓ Installed {outcome.skill_id!r} → {outcome.final_path}")
        return
    print(f"✗ {outcome.verdict}: {outcome.error or '(no detail)'}")
    _print_findings(outcome.findings)


def _print_findings(findings: tuple[ScanFinding, ...]) -> None:
    if not findings:
        return
    print("")
    print("Scan findings:")
    for f in findings:
        marker = {"block": "✗", "warn": "!", "info": "·"}.get(f.severity, "·")
        tool_tag = f" [{f.tool}]" if f.tool else ""
        print(f"  {marker} {f.severity}:{f.code}{tool_tag} — {f.message}")


def _outcome_to_json(outcome: InstallOutcome) -> dict[str, object]:
    return {
        "verdict": outcome.verdict,
        "skill_id": outcome.skill_id,
        "stage_path": str(outcome.stage_path) if outcome.stage_path else None,
        "final_path": str(outcome.final_path) if outcome.final_path else None,
        "error": outcome.error,
        "findings": [
            {
                "severity": f.severity,
                "code": f.code,
                "message": f.message,
                "tool": f.tool,
            }
            for f in outcome.findings
        ],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
