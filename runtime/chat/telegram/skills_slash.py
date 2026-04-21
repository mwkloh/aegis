"""`/skills` slash — list, show, enable, disable, confirm.

Phase 8 §C4. One handler slot on the dispatcher. Sub-verbs keep the
surface small and operator-legible:

* ``/skills list`` — show enabled skills plus any disabled for this chat.
* ``/skills show <id>`` — render the progressive-disclosure YAML blob
  (same bytes the reasoner would see), including tool specs.
* ``/skills enable <id>`` / ``/skills disable <id>`` — flip the
  per-chat toggle. Unknown ids report "unknown skill" rather than
  silently creating a phantom entry.
* ``/skills staged`` — list staged but not-yet-confirmed skills.
* ``/skills confirm <id>`` — move a staged descriptor into the active
  catalog. After confirmation, the loader picks it up on the next
  directory mtime change (or an explicit ``refresh`` call).
"""
from __future__ import annotations

from pathlib import Path

from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.skills.chat_state import ChatSkillState
from runtime.skills.installer import confirm_skill, list_staged
from runtime.skills.loader import SkillLoader

_USAGE = (
    "Usage: /skills list | show <id> | enable <id> | disable <id> "
    "| staged | confirm <id>"
)

_MAX_SHOW_CHARS = 3500  # matches MAX_LONG_RUNNING_CHARS — under Telegram's 4096 cap


def skills_handler(
    *,
    loader: SkillLoader,
    state: ChatSkillState,
    staging_dir: Path,
    catalog_dir: Path,
) -> Handler:
    """Factory for the dispatcher. Closes over the loader + state deps."""

    def _handle(msg: IncomingMessage, cmd: ParsedCommand) -> str:  # noqa: PLR0911 - one return per sub-verb
        if not cmd.args:
            return _USAGE
        sub = cmd.args[0].strip().lower()
        tail = cmd.args[1:]

        if sub == "list":
            return _list(loader, state, chat_id=msg.chat_id)
        if sub == "show":
            return _show(loader, tail)
        if sub in ("enable", "disable"):
            return _toggle(loader, state, sub, tail, chat_id=msg.chat_id)
        if sub == "staged":
            return _staged(staging_dir)
        if sub == "confirm":
            return _confirm(
                tail, staging_dir=staging_dir, catalog_dir=catalog_dir, loader=loader
            )
        return _USAGE

    return _handle


# --- sub-verb implementations ------------------------------------------------


def _list(loader: SkillLoader, state: ChatSkillState, *, chat_id: int) -> str:
    ids = _known_skill_ids(loader)
    if not ids:
        return "No skills installed."
    disabled = set(state.list_disabled(chat_id))
    lines = ["Installed skills:"]
    for skill_id in ids:
        marker = "✗" if skill_id in disabled else "✓"
        lines.append(f"  {marker} {skill_id}")
    if disabled:
        lines.append("")
        lines.append("✗ = disabled for this chat")
    return "\n".join(lines)


def _show(loader: SkillLoader, tail: tuple[str, ...]) -> str:
    if not tail:
        return "Usage: /skills show <id>"
    skill_id = tail[0].strip()
    desc = loader.get(skill_id)
    if desc is None:
        return f"Unknown skill: {skill_id!r}."
    body = SkillLoader.render_for_prompt(desc)
    if len(body) > _MAX_SHOW_CHARS:
        body = body[: _MAX_SHOW_CHARS - 3] + "..."
    return f"Skill {skill_id!r}:\n\n{body}"


def _toggle(
    loader: SkillLoader,
    state: ChatSkillState,
    sub: str,
    tail: tuple[str, ...],
    *,
    chat_id: int,
) -> str:
    if not tail:
        return f"Usage: /skills {sub} <id>"
    skill_id = tail[0].strip()
    if loader.get(skill_id) is None:
        return f"Unknown skill: {skill_id!r}."
    enabled = sub == "enable"
    state.set_enabled(chat_id=chat_id, skill_id=skill_id, enabled=enabled)
    verb = "enabled" if enabled else "disabled"
    return f"Skill {skill_id!r} {verb} for this chat."


def _staged(staging_dir: Path) -> str:
    staged = list_staged(staging_dir)
    if not staged:
        return "No staged skills. Use `aegis-skill-add <path>` to stage one."
    lines = ["Staged skills (awaiting confirm):"]
    lines.extend(f"  • {skill_id}" for skill_id in staged)
    lines.append("")
    lines.append("Run /skills confirm <id> to install.")
    return "\n".join(lines)


def _confirm(
    tail: tuple[str, ...],
    *,
    staging_dir: Path,
    catalog_dir: Path,
    loader: SkillLoader,
) -> str:
    if not tail:
        return "Usage: /skills confirm <id>"
    skill_id = tail[0].strip()
    outcome = confirm_skill(
        skill_id, staging_dir=staging_dir, catalog_dir=catalog_dir
    )
    if outcome.verdict == "confirmed":
        loader.refresh()  # pick up the newly-installed descriptor
        return f"Installed {skill_id!r} into the active catalog."
    if outcome.verdict == "rejected_not_staged":
        return f"No staged descriptor for {skill_id!r}. Stage it first."
    if outcome.verdict == "rejected_scan":
        return (
            f"Cannot confirm {skill_id!r}: staged file has blocking findings. "
            "Re-stage after fixing."
        )
    if outcome.verdict == "rejected_invalid":
        return f"Cannot confirm {skill_id!r}: {outcome.error}"
    return f"Confirm failed for {skill_id!r}: {outcome.error}"


def _known_skill_ids(loader: SkillLoader) -> list[str]:
    # Loader doesn't expose an `all()` — but the internal index maps ids
    # to paths and the installer writes files named `<id>.yaml`. We
    # inspect the id index via a sorted walk; falls back to empty if
    # the catalog dir is missing.
    loader._ensure_index()
    return sorted(loader._id_index.keys())


__all__ = ["skills_handler"]
