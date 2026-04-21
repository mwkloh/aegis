"""Phase 7 §4.3 — read-only slash handlers.

Each handler is a pure function `(IncomingMessage, ParsedCommand) -> str`
closed over its workspace path + any state loaders it needs. The
dispatcher holds the registry; this module just assembles handlers
and exposes a single `build_read_only_handlers()` factory so tests
and `bot.py` construct the dispatch table identically.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from runtime.chat.memory.cold_storage import (
    ColdStorageMismatch,
    ColdStorageMissing,
    ColdStorageReader,
)
from runtime.chat.memory.recall import VaultBodyLoader
from runtime.chat.memory.tier2 import ColdRef, Tier2Store
from runtime.chat.memory.vault_indexer import ReindexResult, VaultIndexer
from runtime.chat.telegram.cron_handler import cron_handler
from runtime.skills import SkillRegistry
from runtime.chat.telegram.dispatch import Handler, IncomingMessage, ParsedCommand
from runtime.chat.telegram.formatters import (
    DEFAULT_DECISIONS_TAIL,
    render_decision_idempotent,
    render_decision_recorded,
    render_decisions,
    render_help,
    render_logs,
    render_pending,
    render_proposal,
    render_status,
    render_system_info,
    render_vault_note,
    render_vault_reindex,
    render_vault_sources,
    render_vault_status,
    render_verbatim,
)
from runtime.chat.telegram.logs import DEFAULT_LOG_LINES, MAX_LOG_LINES, tail_events
from runtime.chat.telegram.status import collect_system_info, compute_status
from runtime.config import AegisConfig
from runtime.events import EventStream
from runtime.improvement.decisions import (
    HumanVerdict,
    latest_by_imp,
    load_decisions,
    record_decision,
)
from runtime.improvement.proposal_loader import load_proposals
from runtime.model_router.router import ModelRouter
from runtime.scheduler.store import ScheduledJobStore

Clock = Callable[[], datetime]


class ColdRefResolver(Protocol):
    """Maps a raw `session_id` to its `ColdRef` (or None).

    Implementations MUST be side-effect-free. Raising means "treat
    as lookup failure" — the recall handler catches every exception
    and renders a friendly reply rather than propagating. Real
    implementations typically wrap a `Tier2Store` query.
    """

    def resolve(self, session_id: str) -> ColdRef | None: ...


_RECALL_USAGE = (
    "Usage: /recall verbatim <session_id>  OR  /recall vault:<slug>"
)

_IMP_HEX_LEN = 8
_IMP_HEX_CHARS = frozenset("0123456789abcdef")


def decisions_handler(workspace: Path) -> Handler:
    """`/decisions [N]` — tail the last N rows of DECISIONS.md."""

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        tail = DEFAULT_DECISIONS_TAIL
        if cmd.args:
            try:
                tail = int(cmd.args[0])
            except ValueError:
                return (
                    f"/decisions expects an integer tail, got {cmd.args[0]!r}. "
                    f"Usage: /decisions [N]."
                )
        decisions = load_decisions(workspace)
        return render_decisions(decisions, tail=tail)

    return _handle


def pending_handler(workspace: Path) -> Handler:
    """`/pending` — list proposals still awaiting a verdict."""

    def _handle(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        proposals = load_proposals(workspace)
        latest = latest_by_imp(load_decisions(workspace))
        return render_pending(proposals, latest)

    return _handle


def proposal_handler(workspace: Path) -> Handler:
    """`/proposal IMP-xxxxxxxx` — pretty-print one proposal."""

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return "Usage: /proposal IMP-xxxxxxxx"
        wanted = _normalize_imp_id(cmd.args[0])
        if wanted is None:
            return (
                f"/proposal expects an IMP-<8 hex> id, got {cmd.args[0]!r}."
            )
        proposals = load_proposals(workspace)
        match = next((p for p in proposals if p.imp_id == wanted), None)
        latest = latest_by_imp(load_decisions(workspace)).get(wanted)
        return render_proposal(match, latest, imp_id=wanted)

    return _handle


def _normalize_imp_id(raw: str) -> str | None:
    """Return a canonical `IMP-xxxxxxxx` (lowercase hex) or None."""
    candidate = raw.strip()
    if not candidate.lower().startswith("imp-"):
        return None
    hex_part = candidate[4:].lower()
    if len(hex_part) != _IMP_HEX_LEN:
        return None
    if any(c not in _IMP_HEX_CHARS for c in hex_part):
        return None
    return f"IMP-{hex_part}"


def status_handler(
    sessions_dir: Path,
    *,
    clock: Clock | None = None,
    cfg: AegisConfig | None = None,
    router: ModelRouter | None = None,
) -> Handler:
    """`/status` — 24h rollup of sessions, patterns, decisions, applies.

    When both ``cfg`` and ``router`` are supplied, the reply is prefixed
    with a model/runtime posture block (current SMART routing, tier
    models, vault state, allowlist size) so the operator gets the
    full picture in one shot. Either missing → legacy activity-only
    reply, preserving the prior formatter contract.
    """
    _clock: Clock = clock if clock is not None else _default_clock

    def _handle(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        snapshot = compute_status(sessions_dir, now=_clock())
        activity = render_status(snapshot)
        if cfg is not None and router is not None:
            info = collect_system_info(cfg, router=router)
            return render_system_info(info) + "\n\n" + activity
        return activity

    return _handle


def logs_handler(
    sessions_dir: Path, *, clock: Clock | None = None
) -> Handler:
    """`/logs [N]` — tail today's last N structural events.

    `N` defaults to `DEFAULT_LOG_LINES` and is clamped to `MAX_LOG_LINES`
    inside `tail_events`. Non-integer arguments return a usage reply
    rather than crashing the dispatcher.
    """
    _clock: Clock = clock if clock is not None else _default_clock

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        requested = DEFAULT_LOG_LINES
        if cmd.args:
            try:
                requested = int(cmd.args[0])
            except ValueError:
                return (
                    f"/logs expects an integer line count, got {cmd.args[0]!r}. "
                    f"Usage: /logs [N] (default {DEFAULT_LOG_LINES}, "
                    f"max {MAX_LOG_LINES})."
                )
        entries, total = tail_events(sessions_dir, n=requested, now=_clock())
        return render_logs(entries, total=total)

    return _handle


def help_handler(*, descriptions: Mapping[str, str]) -> Handler:
    """`/help` — list registered commands + one-line descriptions.

    ``descriptions`` is a filtered view of the default help map —
    `build_dispatcher` passes in only the slashes that were actually
    wired this run, so the operator never sees a command the bot
    can't answer (e.g. ``/recall`` without cold storage, or ``/vault``
    without an indexer).
    """
    items = tuple(descriptions.items())

    def _handle(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        return render_help(items)

    return _handle


DEFAULT_COMMAND_HELP: dict[str, str] = {
    "/help": "List available commands.",
    "/status": "24h activity + current model stack.",
    "/decisions": "Tail the last N entries of DECISIONS.md (default 10).",
    "/pending": "List proposals still awaiting a verdict.",
    "/proposal": "Pretty-print one proposal: /proposal IMP-xxxxxxxx.",
    "/recall": "Read cold storage: /recall verbatim <sid> | /recall vault:<slug>.",
    "/vault": "Vault ops: /vault status | /vault reindex [source] | /vault sources.",
    "/approve": "Record approve: /approve IMP-xxxxxxxx [rationale...].",
    "/reject": "Record reject: /reject IMP-xxxxxxxx [rationale...].",
    "/defer": "Record defer: /defer IMP-xxxxxxxx [rationale...].",
    "/apply": "Apply a coding task: /apply CT-NNN [--dry-run|--no-tests|--status].",
    "/harness": "Run the coding harness CLI: /harness --task CT-NNN ...",
    "/brief": "Generate + post the morning brief from the vault.",
    "/cron": (
        "Scheduler: /cron add \"<cron>\" <skill> [args...] | list | rm <id> "
        "| pause <id> | resume <id>."
    ),
    "/restart": "Restart the bot process (refused while a command is in flight).",
    "/logs": (
        f"Tail today's last N structural events "
        f"(default {DEFAULT_LOG_LINES}, max {MAX_LOG_LINES})."
    ),
    "/health": "Scheduler heartbeat + job roster.",
}


def help_descriptions_for(commands: Iterable[str]) -> dict[str, str]:
    """Filter ``DEFAULT_COMMAND_HELP`` to the registered subset.

    Unknown slashes get a generic fallback so operators never see
    ``None`` in the listing — better to show the command with a
    vague description than hide a working endpoint.
    """
    out: dict[str, str] = {}
    for name in commands:
        out[name] = DEFAULT_COMMAND_HELP.get(name, "(no description available)")
    return out


def _default_clock() -> datetime:
    return datetime.now(tz=UTC)


def recall_handler(
    *,
    resolver: ColdRefResolver,
    reader: ColdStorageReader,
    vault_loader: VaultBodyLoader | None = None,
) -> Handler:
    """`/recall verbatim <session_id>` or `/recall vault:<slug>`.

    Two forms, one registry slot. The handler never raises — every
    failure path (resolver error, cold storage missing/corrupt,
    vault loader failure) renders a human-readable reply so the
    Telegram surface stays operator-legible under outages.
    """

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return _RECALL_USAGE
        first = cmd.args[0].strip()
        if first == "verbatim":
            return _handle_verbatim(cmd.args[1:], resolver=resolver, reader=reader)
        if first.startswith("vault:"):
            slug = first[len("vault:") :].strip()
            return _handle_vault(slug, loader=vault_loader)
        return _RECALL_USAGE

    return _handle


def _handle_verbatim(
    tail: tuple[str, ...],
    *,
    resolver: ColdRefResolver,
    reader: ColdStorageReader,
) -> str:
    if not tail:
        return "Usage: /recall verbatim <session_id>"
    session_id = tail[0].strip()
    if not session_id:
        return "Usage: /recall verbatim <session_id>"
    ref = _resolve_cold_ref(resolver, session_id)
    if isinstance(ref, str):
        return ref  # already a reply (error or not-found)
    return _read_and_render(reader, ref, session_id)


def _resolve_cold_ref(
    resolver: ColdRefResolver, session_id: str
) -> ColdRef | str:
    try:
        ref = resolver.resolve(session_id)
    except Exception:
        return f"Failed to look up session {session_id!r}."
    if ref is None:
        return f"No cold storage for session {session_id!r}."
    return ref


def _read_and_render(
    reader: ColdStorageReader, ref: ColdRef, session_id: str
) -> str:
    try:
        result = reader.read(ref)
    except ColdStorageMissing:
        return f"Session {session_id!r} aged out of cold storage."
    except ColdStorageMismatch:
        return f"Cold storage for session {session_id!r} failed integrity check."
    except Exception:
        return f"Failed to read session {session_id!r}."
    return render_verbatim(result)


def _handle_vault(slug: str, *, loader: VaultBodyLoader | None) -> str:
    if not slug:
        return "Usage: /recall vault:<slug>"
    if loader is None:
        return f"Vault loader not configured (slug: {slug})."
    try:
        body = loader.load(slug)
    except Exception:
        return f"Failed to load vault note {slug!r}."
    if not body:
        return f"Vault note {slug!r} is empty or missing."
    return render_vault_note(slug, body)


_VAULT_USAGE = (
    "Usage: /vault status  OR  /vault reindex [source]  OR  /vault sources"
)


class VaultState:
    """Holds the most recent `ReindexResult` across `/vault` calls.

    One instance per bot process, owned by the `/vault` handler factory
    and seeded on startup by `build_application` after the initial
    reindex. Mutation is single-writer (the handler runs on the
    Telegram event loop) so no lock is needed.
    """

    def __init__(self, *, last_result: ReindexResult | None = None) -> None:
        self.last_result = last_result


def vault_handler(
    *,
    indexer: VaultIndexer,
    tier2: Tier2Store,
    state: VaultState,
) -> Handler:
    """`/vault status|reindex|sources` operator commands."""

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return _VAULT_USAGE
        sub = cmd.args[0].strip().lower()
        cfg = indexer.config
        if sub == "sources":
            return render_vault_sources(cfg)
        if sub == "status":
            total = len(tier2.list_vault_notes())
            return render_vault_status(
                cfg, last_result=state.last_result, total_notes=total
            )
        if sub == "reindex":
            scope = cmd.args[1].strip() if len(cmd.args) > 1 else None
            try:
                result = indexer.reindex(only_label=scope)
            except Exception:
                return "Reindex failed (internal error)."
            state.last_result = result
            return render_vault_reindex(result, scope=scope)
        return _VAULT_USAGE

    return _handle


_HEALTH_STALE_SECONDS = 120.0


def health_handler(
    heartbeat_path: Path,
    store: ScheduledJobStore,
    *,
    clock: Clock | None = None,
) -> Handler:
    """`/health` — scheduler heartbeat + job roster."""
    _clock: Clock = clock if clock is not None else _default_clock

    def _handle(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        now = _clock()
        if not heartbeat_path.exists():
            return "Scheduler has not ticked yet (no heartbeat file)."

        mtime = heartbeat_path.stat().st_mtime
        last_tick = datetime.fromtimestamp(mtime, tz=UTC)
        age_seconds = (now - last_tick).total_seconds()

        if age_seconds > _HEALTH_STALE_SECONDS:
            status = "STALE"
            status_note = f"  ⚠️ Last tick was {round(age_seconds)}s ago — scheduler may have stopped."
        else:
            status = "OK"
            status_note = ""

        jobs = store.list_all()
        job_lines: list[str] = []
        for job in jobs:
            last_status = job.last_status if job.last_status is not None else "never"
            last_run = (
                job.last_run_at.strftime("%Y-%m-%dT%H:%MZ")
                if job.last_run_at is not None
                else "—"
            )
            job_lines.append(f"• {job.id}  {job.skill}  {last_status}  {last_run}")

        header = f"Health: {status}  (last tick {round(age_seconds)}s ago)"
        if status_note:
            header += "\n" + status_note
        roster = f"\nJobs ({len(jobs)}):\n" + "\n".join(job_lines) if jobs else f"\nJobs (0): none"
        return header + roster

    return _handle


def build_read_only_handlers(
    workspace: Path,
    *,
    sessions_dir: Path | None = None,
    clock: Clock | None = None,
    cfg: AegisConfig | None = None,
    router: ModelRouter | None = None,
    recall_resolver: ColdRefResolver | None = None,
    recall_reader: ColdStorageReader | None = None,
    vault_loader: VaultBodyLoader | None = None,
    vault_indexer: VaultIndexer | None = None,
    vault_tier2: Tier2Store | None = None,
    vault_state: VaultState | None = None,
    heartbeat_path: Path | None = None,
    health_store: ScheduledJobStore | None = None,
) -> dict[str, Handler]:
    """Map slash → handler for every read-only Phase 7 command.

    `sessions_dir` defaults to `<workspace>/sessions` so callers that
    have already built an `AegisConfig` can pass `cfg.storage.sessions_dir`
    directly without reshaping.

    When ``cfg`` and ``router`` are both supplied, ``/status`` prepends
    the current model stack + runtime posture block; otherwise the
    reply is the legacy activity-only body.

    `/recall` is registered only when both `recall_resolver` and
    `recall_reader` are supplied — the slash surface-area is
    skipped entirely when cold storage isn't wired, so the
    dispatcher reports `unknown_command` rather than a stub reply.
    """
    sdir = sessions_dir if sessions_dir is not None else workspace / "sessions"
    handlers: dict[str, Handler] = {
        "/decisions": decisions_handler(workspace),
        "/pending": pending_handler(workspace),
        "/proposal": proposal_handler(workspace),
        "/status": status_handler(sdir, clock=clock, cfg=cfg, router=router),
        "/logs": logs_handler(sdir, clock=clock),
    }
    if recall_resolver is not None and recall_reader is not None:
        handlers["/recall"] = recall_handler(
            resolver=recall_resolver,
            reader=recall_reader,
            vault_loader=vault_loader,
        )
    if vault_indexer is not None and vault_tier2 is not None:
        handlers["/vault"] = vault_handler(
            indexer=vault_indexer,
            tier2=vault_tier2,
            state=vault_state if vault_state is not None else VaultState(),
        )
    if heartbeat_path is not None and health_store is not None:
        handlers["/health"] = health_handler(
            heartbeat_path,
            health_store,
            clock=clock,
        )
    return handlers


_VERDICT_USAGE: dict[HumanVerdict, str] = {
    "approve": "Usage: /approve IMP-xxxxxxxx [rationale...]",
    "reject": "Usage: /reject IMP-xxxxxxxx [rationale...]",
    "defer": "Usage: /defer IMP-xxxxxxxx [rationale...]",
}


def _verdict_handler(
    workspace: Path,
    *,
    verdict: HumanVerdict,
    clock: Clock | None = None,
    events: EventStream | None = None,
) -> Handler:
    """Core `/approve|/reject|/defer` handler factory.

    Idempotent: re-issuing the same verdict against an imp_id that
    already has that verdict on file returns a "no change" reply
    without touching `DECISIONS.md`. Issuing a *different* verdict
    writes a new section whose `supersedes` points at the prior
    decision (handled inside `record_decision`).
    """
    usage = _VERDICT_USAGE[verdict]
    _clock: Clock = clock if clock is not None else _default_clock

    def _handle(_msg: IncomingMessage, cmd: ParsedCommand) -> str:
        if not cmd.args:
            return usage
        wanted = _normalize_imp_id(cmd.args[0])
        if wanted is None:
            return (
                f"/{verdict} expects an IMP-<8 hex> id, "
                f"got {cmd.args[0]!r}. {usage}"
            )
        rationale = " ".join(a for a in cmd.args[1:] if a).strip()
        proposals = load_proposals(workspace)
        prior = latest_by_imp(load_decisions(workspace)).get(wanted)
        known = prior is not None or any(p.imp_id == wanted for p in proposals)
        if not known:
            return f"Proposal {wanted} not found."
        if prior is not None and prior.verdict == verdict:
            return render_decision_idempotent(prior)
        decision = record_decision(
            workspace,
            imp_id=wanted,
            verdict=verdict,
            rationale=rationale,
            when=_clock(),
            events=events,
        )
        return render_decision_recorded(decision)

    return _handle


def approve_handler(
    workspace: Path,
    *,
    clock: Clock | None = None,
    events: EventStream | None = None,
) -> Handler:
    """`/approve IMP-xxxxxxxx [rationale...]`."""
    return _verdict_handler(workspace, verdict="approve", clock=clock, events=events)


def reject_handler(
    workspace: Path,
    *,
    clock: Clock | None = None,
    events: EventStream | None = None,
) -> Handler:
    """`/reject IMP-xxxxxxxx [rationale...]`."""
    return _verdict_handler(workspace, verdict="reject", clock=clock, events=events)


def defer_handler(
    workspace: Path,
    *,
    clock: Clock | None = None,
    events: EventStream | None = None,
) -> Handler:
    """`/defer IMP-xxxxxxxx [rationale...]`."""
    return _verdict_handler(workspace, verdict="defer", clock=clock, events=events)


def build_write_handlers(
    workspace: Path,
    *,
    clock: Clock | None = None,
    events: EventStream | None = None,
    scheduler_store: ScheduledJobStore | None = None,
    skill_registry: SkillRegistry | None = None,
) -> dict[str, Handler]:
    """Map slash → handler for the human-verdict + scheduler write commands.

    `events` is a single `EventStream` shared across verdict slashes
    so every write lands on the same shard; bot.py is responsible for
    rotating the stream at UTC midnight.

    `scheduler_store` opts `/cron` into the dispatch table. When
    omitted, `/cron` is skipped and the dispatcher reports
    `unknown_command` — the correct behaviour when the scheduler
    isn't wired (e.g. unit tests, degraded deployments).

    `skill_registry` is optional — when supplied, `/cron add`
    rejects skills the scheduler can't run (unknown id or
    descriptor without an `argv_template`) at add time rather than
    letting them fail silently every tick.
    """
    _clock: Clock = clock if clock is not None else _default_clock
    handlers: dict[str, Handler] = {
        "/approve": approve_handler(workspace, clock=clock, events=events),
        "/reject": reject_handler(workspace, clock=clock, events=events),
        "/defer": defer_handler(workspace, clock=clock, events=events),
    }
    if scheduler_store is not None:
        handlers["/cron"] = cron_handler(
            store=scheduler_store, clock=_clock, registry=skill_registry
        )
    return handlers


__all__ = [
    "DEFAULT_COMMAND_HELP",
    "Clock",
    "ColdRefResolver",
    "VaultState",
    "approve_handler",
    "build_read_only_handlers",
    "build_write_handlers",
    "decisions_handler",
    "defer_handler",
    "help_descriptions_for",
    "help_handler",
    "logs_handler",
    "pending_handler",
    "proposal_handler",
    "recall_handler",
    "reject_handler",
    "status_handler",
    "vault_handler",
]
