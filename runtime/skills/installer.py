"""Skill install pipeline — stage + confirm.

Phase 8 §C5. Operator runs ``aegis-skill-add <path>`` which:

1. Parses the YAML, validates via ``SkillDescriptor``.
2. Runs ``scan_descriptor``; if any finding is ``block``-severity,
   the install is rejected outright.
3. Otherwise copies the file to a staging directory, keyed by
   skill id, returning an ``InstallOutcome`` with all findings.
4. The operator reviews the findings, then runs ``/skills confirm <id>``
   from Telegram (or ``aegis-skill-add --confirm <id>`` locally)
   which moves the staged file into the active catalog.

Two stages keep an LLM-generated or third-party descriptor from
reaching the active catalog without a human pass through the
findings. The staging dir is per-operator (``~/.aegis/skills_staging/``
by default) so concurrent installs don't race.

Never raises — every failure path returns an ``InstallOutcome`` with
a descriptive verdict so the CLI and slash surface share one set of
error shapes.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from runtime.skills.registry import SkillDescriptor
from runtime.skills.scanner import ScanFinding, scan_descriptor

InstallVerdict = Literal[
    "staged",             # passed validation + scan, ready to confirm
    "confirmed",          # moved into active catalog
    "rejected_scan",      # scan produced blocking findings
    "rejected_invalid",   # YAML failed pydantic validation
    "rejected_source",    # source path missing / unreadable
    "rejected_conflict",  # staging slot already occupied
    "rejected_not_staged",  # confirm called for an id that isn't staged
]


@dataclass(frozen=True)
class InstallOutcome:
    """Everything the CLI + slash handler need to report back."""

    verdict: InstallVerdict
    skill_id: str
    stage_path: Path | None
    final_path: Path | None
    findings: tuple[ScanFinding, ...] = field(default_factory=tuple)
    error: str = ""

    def is_success(self) -> bool:
        return self.verdict in ("staged", "confirmed")


def stage_skill(  # noqa: PLR0911 - one return per distinct rejection verdict
    source: Path,
    *,
    staging_dir: Path,
) -> InstallOutcome:
    """Validate + scan ``source``; copy into ``staging_dir`` on success.

    Returns an outcome; never raises. ``staging_dir`` is created if
    missing. If a file for the same ``skill_id`` is already staged
    (i.e. a previous install run wasn't confirmed), we reject with
    ``rejected_conflict`` so the operator explicitly resolves the
    collision.
    """
    if not source.is_file():
        return InstallOutcome(
            verdict="rejected_source",
            skill_id="",
            stage_path=None,
            final_path=None,
            error=f"source {source!s} is not a file",
        )

    try:
        raw_text = source.read_text(encoding="utf-8")
    except OSError as exc:
        return InstallOutcome(
            verdict="rejected_source",
            skill_id="",
            stage_path=None,
            final_path=None,
            error=f"could not read {source!s}: {exc}",
        )

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id="",
            stage_path=None,
            final_path=None,
            error=f"YAML parse error: {exc}",
        )
    if not isinstance(raw, dict):
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id="",
            stage_path=None,
            final_path=None,
            error="descriptor must be a YAML mapping",
        )

    try:
        descriptor = SkillDescriptor(**raw)
    except ValidationError as exc:
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id=str(raw.get("id", "")),
            stage_path=None,
            final_path=None,
            error=_short_validation_error(exc),
        )

    findings = tuple(scan_descriptor(descriptor))
    blocking = [f for f in findings if f.is_blocking()]

    staging_dir.mkdir(parents=True, exist_ok=True)
    stage_path = staging_dir / f"{descriptor.id}.yaml"

    if blocking:
        return InstallOutcome(
            verdict="rejected_scan",
            skill_id=descriptor.id,
            stage_path=None,
            final_path=None,
            findings=findings,
            error="scan produced blocking findings; fix the descriptor and retry",
        )

    if stage_path.exists():
        return InstallOutcome(
            verdict="rejected_conflict",
            skill_id=descriptor.id,
            stage_path=stage_path,
            final_path=None,
            findings=findings,
            error=(
                f"{stage_path.name} already staged; run "
                f"/skills confirm {descriptor.id} or delete the staged file"
            ),
        )

    # Write atomically via a tmp sibling so a crash mid-write doesn't
    # leave a half-file in staging that would block re-runs.
    tmp = stage_path.with_suffix(".yaml.tmp")
    try:
        tmp.write_text(raw_text, encoding="utf-8")
        tmp.replace(stage_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return InstallOutcome(
            verdict="rejected_source",
            skill_id=descriptor.id,
            stage_path=None,
            final_path=None,
            findings=findings,
            error=f"failed to write staging file: {exc}",
        )

    return InstallOutcome(
        verdict="staged",
        skill_id=descriptor.id,
        stage_path=stage_path,
        final_path=None,
        findings=findings,
    )


def confirm_skill(
    skill_id: str,
    *,
    staging_dir: Path,
    catalog_dir: Path,
) -> InstallOutcome:
    """Move a staged skill into the active catalog.

    The staged file is re-read and re-validated on confirm so a
    tampered staging file can't slip through. On success the staged
    file is removed.
    """
    stage_path = staging_dir / f"{skill_id}.yaml"
    if not stage_path.is_file():
        return InstallOutcome(
            verdict="rejected_not_staged",
            skill_id=skill_id,
            stage_path=None,
            final_path=None,
            error=f"no staged descriptor for {skill_id!r}",
        )

    # Re-validate the staged file — if someone edited it post-staging
    # with a block-severity issue, reject before it reaches the catalog.
    outcome = _revalidate_staged(stage_path, skill_id)
    if outcome is not None:
        return outcome

    final_dir = catalog_dir / skill_id
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "skill.yaml"
    try:
        shutil.copyfile(stage_path, final_path)
        stage_path.unlink()
    except OSError as exc:
        return InstallOutcome(
            verdict="rejected_source",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            error=f"failed to install: {exc}",
        )

    return InstallOutcome(
        verdict="confirmed",
        skill_id=skill_id,
        stage_path=stage_path,
        final_path=final_path,
    )


def list_staged(staging_dir: Path) -> list[str]:
    """Return staged skill ids in deterministic filename order."""
    if not staging_dir.is_dir():
        return []
    return sorted(p.stem for p in staging_dir.glob("*.yaml"))


def _revalidate_staged(stage_path: Path, skill_id: str) -> InstallOutcome | None:
    try:
        raw = yaml.safe_load(stage_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            error=f"staged file unreadable: {exc}",
        )
    if not isinstance(raw, dict):
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            error="staged descriptor must be a YAML mapping",
        )
    try:
        desc = SkillDescriptor(**raw)
    except ValidationError as exc:
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            error=_short_validation_error(exc),
        )
    if desc.id != skill_id:
        return InstallOutcome(
            verdict="rejected_invalid",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            error=f"staged file id {desc.id!r} does not match {skill_id!r}",
        )
    findings = tuple(scan_descriptor(desc))
    if any(f.is_blocking() for f in findings):
        return InstallOutcome(
            verdict="rejected_scan",
            skill_id=skill_id,
            stage_path=stage_path,
            final_path=None,
            findings=findings,
            error="re-scan produced blocking findings; staged file was modified",
        )
    return None


def _short_validation_error(exc: ValidationError) -> str:
    """Condense pydantic ValidationError to one line."""
    errors = exc.errors(include_url=False, include_input=False)
    if not errors:
        return "validation failed"
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "invalid")
    extra = "" if len(errors) == 1 else f" (+{len(errors) - 1} more)"
    return f"{loc}: {msg}{extra}"


__all__ = [
    "InstallOutcome",
    "InstallVerdict",
    "confirm_skill",
    "list_staged",
    "stage_skill",
]
