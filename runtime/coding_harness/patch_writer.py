"""Render a `Draft` to a `.patch.md` file under the workspace.

Output naming: `CT-NNN__IMP-xxxxxxxx__YYYY-MM-DDTHHMMZ.patch.md`.
The directory is append-only — `--force` writes a new timestamped
file alongside any prior draft so history is never destroyed.
"""
from __future__ import annotations

from datetime import UTC
from pathlib import Path

from .draft import Draft

_DIFFS_DIRNAME = "diffs"


def coding_harness_dir(workspace: Path) -> Path:
    out = Path(workspace) / "coding_harness"
    out.mkdir(parents=True, exist_ok=True)
    return out


def diffs_dir(workspace: Path) -> Path:
    out = coding_harness_dir(workspace) / _DIFFS_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def patch_filename(draft: Draft) -> str:
    ts = draft.drafted_at.astimezone(UTC).strftime("%Y-%m-%dT%H%MZ")
    return f"{draft.ct_id}__{draft.imp_id}__{ts}.patch.md"


def existing_drafts_for(workspace: Path, ct_id: str) -> list[Path]:
    """All prior `.patch.md` drafts for a given CT, oldest first."""
    return sorted(diffs_dir(workspace).glob(f"{ct_id}__*.patch.md"))


def write_patch(workspace: Path, draft: Draft) -> Path:
    """Write a `.patch.md`. Adds a `-N` suffix if a file at that minute exists."""
    base = patch_filename(draft)
    target = diffs_dir(workspace) / base
    if target.exists():
        stem = base.removesuffix(".patch.md")
        n = 2
        while True:
            candidate = diffs_dir(workspace) / f"{stem}-{n:02d}.patch.md"
            if not candidate.exists():
                target = candidate
                break
            n += 1
    target.write_text(_render(draft), encoding="utf-8")
    return target


def _render(draft: Draft) -> str:
    ts = draft.drafted_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")
    head = (
        f"# {draft.ct_id} — {draft.imp_id} (status: {draft.status})\n\n"
        f"- **Drafted:** {ts}\n"
        f"- **Model:** {draft.model}\n"
    )
    if draft.reason:
        head += f"- **Reason:** {draft.reason}\n"
    head += "\n"

    summary_block = f"## Summary\n{draft.summary or '—'}\n\n"

    if draft.status == "ok":
        diff_block = "## Unified diff\n```diff\n" + draft.unified_diff.rstrip() + "\n```\n\n"
    elif draft.status == "refused":
        diff_block = (
            "## Unified diff\n"
            "_Refused — task scope or model output touches canonical files._\n\n"
        )
    else:
        diff_block = (
            "## Unified diff\n"
            f"_Stub — no diff produced. Reason: {draft.reason or 'unknown'}._\n\n"
        )

    notes_block = f"## Test notes\n{draft.test_notes or '—'}\n\n"
    rollback_block = f"## Rollback\n{draft.rollback or '—'}\n"

    return head + summary_block + diff_block + notes_block + rollback_block
