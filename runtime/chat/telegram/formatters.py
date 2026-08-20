"""Phase 7 §4.3 — reply renderers for the operator surface.

Pure functions. One per read-only slash. No Telegram SDK imports
here — these return plain `str`; the bot entrypoint is responsible
for `parse_mode`, truncation to Telegram's 4096-byte limit, and any
`edited message` semantics.

Formatting conventions:

* Use monospace fences (```) for tabular data so Telegram's MarkdownV2
  renders it with a fixed-width font without us having to escape
  every operator-facing string.
* Never echo raw user-supplied text into a MarkdownV2 reply — all
  renderers here derive their output from plane-3 files
  (`DECISIONS.md`, `PROPOSALS.md`) and `AegisConfig`.
* Keep output under ~3500 chars by default so we leave headroom
  for the bot entrypoint to append status lines.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from runtime.chat.memory.cold_storage import ColdStorageRead
from runtime.chat.memory.vault_indexer import ReindexResult
from runtime.chat.telegram.status import StatusSnapshot, SystemInfo
from runtime.config import VaultIndexingConfig
from runtime.improvement.decisions import Decision
from runtime.improvement.proposal_loader import LoadedProposal

DEFAULT_DECISIONS_TAIL = 10
MAX_DECISIONS_TAIL = 50
MAX_PENDING_ROWS = 25
MAX_VERBATIM_CHARS = 3500
MAX_VAULT_BODY_CHARS = 3500
MAX_LOG_LINE_CHARS = 120
# Length of an ISO-8601 timestamp up to and including seconds
# ("2026-04-20T18:34:02" = 19 chars). Used to slice the HH:MM:SS piece
# out without failing on shorter truncations.
_ISO_SECONDS_LEN = 19
_PAYLOAD_FIELDS_PER_LINE = 3
_PAYLOAD_VALUE_MAX = 20
_PAYLOAD_HIDE: frozenset[str] = frozenset({"session_id", "ts"})


def render_decisions(
    decisions: Sequence[Decision], *, tail: int = DEFAULT_DECISIONS_TAIL
) -> str:
    """Render the last `tail` decisions in `DECISIONS.md` order.

    `tail <= 0` is coerced to the default; a tail larger than
    `MAX_DECISIONS_TAIL` is clamped. An empty log renders a
    human-readable "no decisions yet" string so the operator sees
    an explicit state rather than silence.
    """
    if tail <= 0:
        tail = DEFAULT_DECISIONS_TAIL
    tail = min(tail, MAX_DECISIONS_TAIL)
    if not decisions:
        return "No decisions recorded yet."
    recent = list(decisions)[-tail:]
    lines = [f"Last {len(recent)} decision(s):"]
    for d in recent:
        ts = d.decided_at.strftime("%Y-%m-%d %H:%MZ")
        rationale = d.rationale or "—"
        lines.append(f"• {ts} — {d.imp_id} — {d.verdict} — {rationale}")
    return "\n".join(lines)


def pending_proposals(
    proposals: Sequence[LoadedProposal],
    latest_decisions: Mapping[str, Decision],
) -> list[LoadedProposal]:
    """Filter to proposals that still need an operator verdict.

    "Pending" = no decision on file, OR the latest decision is
    `defer` (deferred proposals re-surface until resolved).
    Applier verdicts (`applied_clean` etc.) don't make a proposal
    pending again — once approved, a later applier outcome is
    reporting state, not review state.
    """
    out: list[LoadedProposal] = []
    for proposal in proposals:
        latest = latest_decisions.get(proposal.imp_id)
        if latest is None or latest.verdict == "defer":
            out.append(proposal)
    return out


def render_pending(
    proposals: Sequence[LoadedProposal],
    latest_decisions: Mapping[str, Decision],
) -> str:
    """One-line summary per pending proposal, capped at MAX_PENDING_ROWS."""
    pending = pending_proposals(proposals, latest_decisions)
    if not pending:
        return "No proposals drafted."
    visible = pending[:MAX_PENDING_ROWS]
    lines = [f"Pending proposals ({len(visible)} of {len(pending)}):"]
    for p in visible:
        deferred = (
            " (previously deferred)"
            if (latest_decisions.get(p.imp_id) is not None)
            else ""
        )
        change_preview = _truncate(p.change, 80)
        lines.append(
            f"• {p.imp_id} [{p.risk}] {p.pattern_detector} — "
            f"{change_preview}{deferred}"
        )
    return "\n".join(lines)


def render_proposal(
    proposal: LoadedProposal | None,
    latest_decision: Decision | None,
    *,
    imp_id: str | None = None,
) -> str:
    """Pretty-print one proposal, optionally annotated with its latest decision."""
    if proposal is None:
        label = imp_id or "(unknown)"
        return f"Proposal {label} not found."
    affected = ", ".join(proposal.affected) if proposal.affected else "—"
    rationale = proposal.rationale or "—"
    lines = [
        f"{proposal.imp_id} — {proposal.pattern_detector} (risk: {proposal.risk})",
        f"Run: {proposal.source_run}",
        f"Affected: {affected}",
        f"Change: {proposal.change}",
        f"Rationale: {rationale}",
    ]
    if latest_decision is not None:
        ts = latest_decision.decided_at.strftime("%Y-%m-%d %H:%MZ")
        lines.append(
            f"Latest decision: {latest_decision.verdict} at {ts} — "
            f"{latest_decision.rationale or '—'}"
        )
    else:
        lines.append("Latest decision: (none — drafted)")
    return "\n".join(lines)


def render_status(snapshot: StatusSnapshot) -> str:
    """Render a 24h activity snapshot as a short block."""
    start = snapshot.window_start.strftime("%Y-%m-%d %H:%MZ")
    end = snapshot.window_end.strftime("%Y-%m-%d %H:%MZ")
    return (
        f"Last 24h ({start} → {end}):\n"
        f"• sessions:  {snapshot.sessions}\n"
        f"• patterns:  {snapshot.patterns}\n"
        f"• decisions: {snapshot.decisions}\n"
        f"• applies:   {snapshot.applies}"
    )


def render_system_info(info: SystemInfo) -> str:
    """Render the currently-routed model stack + runtime posture."""
    smart_line = f"{info.smart_provider}:{info.smart_model}"
    local_state = "reachable" if info.local_ready else "unreachable"
    openrouter_state = "configured" if info.openrouter_configured else "missing"
    vault_line: str
    if info.vault_enabled and info.vault_root is not None:
        vault_line = f"{info.vault_root} ({info.vault_sources} source(s))"
    elif info.vault_root is not None:
        vault_line = f"{info.vault_root} (no sources)"
    else:
        vault_line = "disabled"
    return (
        "Models:\n"
        f"• smart:      {smart_line}\n"
        f"• smart_local:{info.smart_local_model}\n"
        f"• fast:       {info.fast_model}\n"
        f"• reflection: {info.reflection_model}\n"
        f"• coding:     {info.coding_model}\n"
        "Runtime:\n"
        f"• ollama:     {info.ollama_base_url} ({local_state})\n"
        f"• openrouter: {openrouter_state}\n"
        f"• vault:      {vault_line}\n"
        f"• allowlist:  {info.allowlist_size} chat(s)"
    )


def render_help(items: Sequence[tuple[str, str]]) -> str:
    """Render a sorted list of `(slash, description)` pairs.

    The command dispatch table is the source of truth for what is
    *registered*; the handler closes over a description map so `/help`
    only lists commands that actually answer in this deployment
    (e.g. `/recall` appears only when cold storage is wired).
    """
    if not items:
        return "No commands registered."
    lines = ["Available commands:"]
    for slash, desc in sorted(items, key=lambda pair: pair[0]):
        lines.append(f"• {slash} — {desc}")
    return "\n".join(lines)


def render_verbatim(result: ColdStorageRead) -> str:
    """Render a cold-storage slice as a chat replay.

    Header names the session + turn range so operators can quote it
    back; body is one line per turn. A soft cap at `MAX_VERBATIM_CHARS`
    protects against pathological logs — Telegram's 4 KB per-message
    limit still applies, and `bot.py` is the chunker of record.
    """
    ref = result.ref
    start, end = ref.turn_range
    header = (
        f"Session {ref.session_id} — turns [{start},{end}) "
        f"({len(result.turns)} turn(s))"
    )
    lines = [header]
    for turn in result.turns:
        ts = turn.ts.strftime("%Y-%m-%d %H:%MZ")
        lines.append(f"[{ts}] {turn.role}: {turn.text}")
    body = "\n".join(lines)
    return _clip(body, MAX_VERBATIM_CHARS)


def render_vault_note(slug: str, body: str) -> str:
    """Render one vault note body with a slug header."""
    clipped = _clip(body.rstrip(), MAX_VAULT_BODY_CHARS)
    return f"Vault: {slug}\n\n{clipped}"


def render_vault_sources(cfg: VaultIndexingConfig) -> str:
    """Render the configured vault sources as a short block."""
    if not cfg.is_enabled():
        return "Vault indexing not configured."
    lines = [f"Vault root: {cfg.vault_root}"]
    lines.append(f"Reindex interval: {cfg.reindex_interval_hours}h")
    lines.append(f"Sources ({len(cfg.sources)}):")
    for src in cfg.sources:
        label = src.label or "—"
        excludes = ", ".join(src.exclude) if src.exclude else "—"
        lines.append(
            f"• {src.path} (label: {label}, priority: {src.priority}, "
            f"glob: {src.glob}, exclude: {excludes})"
        )
    return "\n".join(lines)


def render_vault_status(
    cfg: VaultIndexingConfig,
    *,
    last_result: ReindexResult | None,
    total_notes: int,
) -> str:
    """Render `/vault status` — last run summary + current corpus size."""
    if not cfg.is_enabled():
        return "Vault indexing not configured."
    lines = [
        f"Vault root: {cfg.vault_root}",
        f"Sources configured: {len(cfg.sources)}",
        f"Notes currently indexed: {total_notes}",
    ]
    if last_result is None:
        lines.append("Last run: never")
        return "\n".join(lines)
    finished = last_result.finished_at.strftime("%Y-%m-%d %H:%MZ")
    lines.append(
        f"Last run: {finished} — added: {last_result.added}, "
        f"updated: {last_result.updated}, skipped: {last_result.skipped}, "
        f"pruned: {last_result.pruned}"
    )
    if last_result.errors:
        lines.append(f"Errors ({len(last_result.errors)}):")
        for err in last_result.errors[:5]:
            lines.append(f"  · {err}")
    return "\n".join(lines)


def render_vault_reindex(result: ReindexResult, *, scope: str | None = None) -> str:
    """Render `/vault reindex [source]` outcome."""
    target = f"source {scope!r}" if scope else "all sources"
    if result.errors and result.total_indexed == 0 and result.pruned == 0:
        first = result.errors[0]
        return f"Reindex ({target}) failed: {first}"
    lines = [
        f"Reindex {target} complete.",
        f"added: {result.added}, updated: {result.updated}, "
        f"skipped: {result.skipped}, pruned: {result.pruned}",
    ]
    if result.errors:
        lines.append(f"Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            lines.append(f"  · {err}")
    return "\n".join(lines)


def render_decision_recorded(decision: Decision) -> str:
    """Confirm a freshly-written verdict back to the operator."""
    ts = decision.decided_at.strftime("%Y-%m-%d %H:%MZ")
    rationale = decision.rationale or "—"
    lines = [
        f"Recorded {decision.verdict} for {decision.imp_id} at {ts}.",
        f"Rationale: {rationale}",
    ]
    if decision.supersedes:
        lines.append(f"Supersedes: {decision.supersedes}")
    return "\n".join(lines)


def render_decision_idempotent(prior: Decision) -> str:
    """Reply when an operator re-issues the same verdict already on file."""
    ts = prior.decided_at.strftime("%Y-%m-%d %H:%MZ")
    return (
        f"{prior.imp_id} is already {prior.verdict} "
        f"(decided at {ts}). No change recorded."
    )


def render_logs(
    entries: Sequence[Mapping[str, object]], *, total: int
) -> str:
    """Render today's tail of structural events for `/logs`.

    Each entry is rendered as one line: ``HH:MM:SS type k=v k=v k=v``.
    Complex payload values (lists, dicts, nested objects) are skipped —
    a log tail is only useful if each line fits on a phone screen.
    Long string values are clipped to keep lines readable.
    """
    if total == 0 or not entries:
        return "No events recorded yet today."
    shown = len(entries)
    header = f"Last {shown} of {total} event(s) today (UTC):"
    body_lines = [_format_log_entry(e) for e in entries]
    return header + "\n" + "\n".join(body_lines)


def _format_log_entry(entry: Mapping[str, object]) -> str:
    ts = entry.get("ts")
    time_part = (
        ts[11:_ISO_SECONDS_LEN]
        if isinstance(ts, str) and len(ts) >= _ISO_SECONDS_LEN
        else "?"
    )
    raw_type = entry.get("type")
    event_type = raw_type if isinstance(raw_type, str) and raw_type else "?"
    raw_payload = entry.get("payload")
    payload: Mapping[str, object] = (
        raw_payload if isinstance(raw_payload, Mapping) else {}
    )
    fields = _format_payload_fields(payload)
    base = f"{time_part} {event_type}"
    if fields:
        base = f"{base} {fields}"
    if len(base) > MAX_LOG_LINE_CHARS:
        base = base[: MAX_LOG_LINE_CHARS - 1] + "…"
    return base


def _format_payload_fields(payload: Mapping[str, object]) -> str:
    pairs: list[str] = []
    for key, value in payload.items():
        if key in _PAYLOAD_HIDE:
            continue
        rendered = _format_payload_value(value)
        if rendered is None:
            continue
        pairs.append(f"{key}={rendered}")
        if len(pairs) >= _PAYLOAD_FIELDS_PER_LINE:
            break
    return " ".join(pairs)


def _format_payload_value(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.replace("\n", " ").strip()
        if not stripped:
            return None
        if len(stripped) > _PAYLOAD_VALUE_MAX:
            return stripped[: _PAYLOAD_VALUE_MAX - 1] + "…"
        return stripped
    return None


def _truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


__all__ = [
    "DEFAULT_DECISIONS_TAIL",
    "MAX_DECISIONS_TAIL",
    "MAX_LOG_LINE_CHARS",
    "MAX_PENDING_ROWS",
    "MAX_VAULT_BODY_CHARS",
    "MAX_VERBATIM_CHARS",
    "pending_proposals",
    "render_decision_idempotent",
    "render_decision_recorded",
    "render_decisions",
    "render_help",
    "render_logs",
    "render_pending",
    "render_proposal",
    "render_status",
    "render_system_info",
    "render_vault_note",
    "render_vault_reindex",
    "render_vault_sources",
    "render_vault_status",
    "render_verbatim",
]
