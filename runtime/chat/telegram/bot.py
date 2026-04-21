"""Phase 7 §4.3 — long-poll Telegram entrypoint.

`bot.py` is the only module in `runtime/chat/telegram/` that touches the
real `python-telegram-bot` SDK. Every upstream module (auth, dispatch,
handlers, formatters, status, recall) is pure Python with no network
dependency — this file glues them to `telegram.ext.Application` so the
rest of the package stays unit-testable without the SDK installed.

Design pins:

* **Lazy SDK import.** `telegram.ext` is only imported inside
  `build_application`. Tests drive `route_command` / `route_chat`
  directly with duck-typed `FakeUpdate` objects and never need the
  real package.
* **One source of truth for the slash table.** `build_dispatcher`
  merges `build_read_only_handlers(...)` with `build_write_handlers(...)`
  so the operator surface is assembled in exactly one place. A single
  `EventStream` is shared across write slashes so every
  `governance.decision` event lands on the same JSONL shard.
* **4 KB Telegram limit.** `_chunk` splits long replies on newline
  boundaries when possible, hard-slicing only when a single line is
  itself too long. Chunks are sent in order via repeated `reply_text`
  calls.
* **Conversational path is a stub.** `route_chat` answers with a
  "not yet wired" message; the real Tier 3/2/1 pipeline wiring is a
  follow-up step (plan §6 step 13 E2E smoke).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from telegram.constants import ChatAction

from memory.embeddings import Bgem3Embedder, Embedder, FakeEmbedder, build_embedder
from runtime.chat.memory.cold_storage import ColdStorageReader
from runtime.chat.memory.context_builder import ContextBuilder
from runtime.chat.memory.recall import RecallPolicy, VaultBodyLoader
from runtime.chat.memory.tier1 import Tier1Loader
from runtime.chat.memory.tier2 import Tier2Store
from runtime.chat.memory.tier3 import Tier3Store
from runtime.chat.memory.vault_indexer import (
    FilesystemVaultBodyLoader,
    VaultIndexer,
)
from runtime.chat.pipeline import ChatPipeline
from runtime.chat.telegram.auth import Authorizer
from runtime.chat.telegram.dispatch import Dispatcher, IncomingMessage, parse_command
from runtime.chat.telegram.handlers import (
    Clock,
    ColdRefResolver,
    VaultState,
    build_read_only_handlers,
    build_write_handlers,
    help_descriptions_for,
    help_handler,
)
from runtime.chat.telegram.long_running import (
    InFlightRegistry,
    LongRunningRunner,
    SubprocessRunner,
)
from runtime.chat.telegram.restart import reexec_in_place
from runtime.config import AegisConfig
from runtime.events import EventStream
from runtime.model_router.clients import OpenRouterClient
from runtime.model_router.clients.base import ModelClient
from runtime.model_router.clients.ollama_client import OllamaClient, OllamaHostError
from runtime.model_router.clients.openrouter_client import OpenRouterConfigError
from runtime.model_router.router import LocalReadyProbe, ModelRouter, ModelTier
from runtime.scheduler import (
    JobRunner,
    ScheduledJobStore,
    SchedulerEngine,
    seed_system_jobs,
    system_clock,
    system_sleeper,
)
from runtime.skills.intent_router import IntentRouter
from runtime.skills.registry import SkillDescriptor, SkillRegistry

SkillArgResolver = Callable[[SkillDescriptor], list[str] | None]

_CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "catalog"

MAX_TELEGRAM_CHARS = 4000

_CHAT_STUB_REPLY = (
    "AEGIS conversational replies are not yet wired. "
    "Use /pending, /status, /decisions, /proposal, or /recall for now."
)
_CHAT_UNAUTHORIZED_REPLY = "unauthorized"
_CHAT_THINKING_PLACEHOLDER = "⏳ Thinking…"
_RESTART_ACK = "🔄 Restarting AEGIS…"
_RESTART_BUSY = (
    "Cannot restart — another command is still running. "
    "Wait for it to finish and try /restart again."
)
# Delay the exec just long enough that the ack send completes and
# python-telegram-bot has flushed its outbound queue. 0.5s is generous.
_RESTART_EXEC_DELAY = 0.5
# Telegram's ChatAction auto-expires at ~5s; refresh at 4s so the header
# typing animation stays continuous through multi-second LLM turns.
_TYPING_REFRESH_SECONDS = 4.0

logger = logging.getLogger(__name__)


class _Replyable(Protocol):
    """Duck-typed Telegram Message — only `reply_text` is required."""

    async def reply_text(self, text: str) -> Any: ...


def build_dispatcher(
    workspace: Path,
    *,
    authorizer: Authorizer,
    sessions_dir: Path | None = None,
    clock: Clock | None = None,
    cfg: AegisConfig | None = None,
    router: ModelRouter | None = None,
    events: EventStream | None = None,
    recall_resolver: ColdRefResolver | None = None,
    recall_reader: ColdStorageReader | None = None,
    vault_loader: VaultBodyLoader | None = None,
    vault_indexer: VaultIndexer | None = None,
    vault_tier2: Tier2Store | None = None,
    vault_state: VaultState | None = None,
    scheduler_store: ScheduledJobStore | None = None,
    skill_registry: SkillRegistry | None = None,
    extra_command_help: dict[str, str] | None = None,
) -> Dispatcher:
    """Assemble a `Dispatcher` wired with both read and write slashes.

    `events` is threaded only into the write handlers — read-only
    slashes are side-effect-free. `recall_resolver` + `recall_reader`
    must both be supplied for `/recall` to register; otherwise the
    dispatcher reports `unknown_command` for `/recall`, which is the
    correct behaviour when cold storage isn't wired yet. Same shape
    for `/vault`: both `vault_indexer` and `vault_tier2` must be
    supplied or the slash is skipped.

    ``cfg`` + ``router`` (both required) extend ``/status`` with a
    model-stack + runtime-posture preamble. ``extra_command_help``
    lets the caller (``build_application``) teach ``/help`` about
    slashes that live outside the dispatcher — the long-running set
    (``/apply`` etc.) and the out-of-band ``/restart``.
    """
    read_only = build_read_only_handlers(
        workspace,
        sessions_dir=sessions_dir,
        clock=clock,
        cfg=cfg,
        router=router,
        recall_resolver=recall_resolver,
        recall_reader=recall_reader,
        vault_loader=vault_loader,
        vault_indexer=vault_indexer,
        vault_tier2=vault_tier2,
        vault_state=vault_state,
    )
    write = build_write_handlers(
        workspace,
        clock=clock,
        events=events,
        scheduler_store=scheduler_store,
        skill_registry=skill_registry,
    )
    merged: dict[str, Any] = {**read_only, **write}
    # /help is a pure lookup so it goes in last, closed over the
    # complete set (read + write + any extras from the caller).
    registered = {*merged.keys(), "/help"}
    if extra_command_help:
        registered.update(extra_command_help.keys())
    descriptions = help_descriptions_for(registered)
    if extra_command_help:
        descriptions.update(extra_command_help)
    merged["/help"] = help_handler(descriptions=descriptions)
    return Dispatcher(authorizer=authorizer, handlers=merged)


async def route_command(
    update: Any,
    _context: Any,
    *,
    dispatcher: Dispatcher,
    long_runner: LongRunningRunner | None = None,
    restart_fn: Callable[[], None] | None = None,
) -> None:
    """Map an incoming slash Update → Dispatcher → reply_text(s).

    Missing or malformed updates are ignored silently — there is no
    one to reply to. The dispatcher itself never raises (see §2.8),
    so this coroutine is safe to hand to `python-telegram-bot` without
    an outer try/except.

    When `long_runner` is supplied, `/apply` and `/harness` bypass the
    sync `Dispatcher` and are handled by the long-running surface
    instead (per plan §4.2). Auth is still enforced — we reuse the
    dispatcher's `Authorizer` so the allowlist is the single source
    of truth.

    `/restart` is also handled out-of-band: it re-execs the bot process
    in place via `os.execv` so the operator can pick up new code or
    config without shell access. Refused when a long-running command
    is busy (a mid-flight `/apply` would be killed). `restart_fn` is
    an injectable hook the production path sets to a no-arg wrapper
    around `os.execv`; tests swap in a spy.
    """
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    message = getattr(update, "effective_message", None)
    if chat is None or user is None or message is None:
        return
    text = getattr(message, "text", None) or ""
    chat_id = int(chat.id)
    user_id = int(user.id)

    parsed = parse_command(text)
    if parsed is not None and parsed.name == "/restart":
        await _handle_restart(
            message,
            chat_id=chat_id,
            dispatcher=dispatcher,
            long_runner=long_runner,
            restart_fn=restart_fn,
        )
        return

    if (
        long_runner is not None
        and parsed is not None
        and parsed.name in long_runner.commands
    ):
        decision = dispatcher.authorizer.check(chat_id)
        if not decision:
            await _reply(message, "unauthorized")
            logger.info(
                "telegram.long_running.denied",
                extra={"command": parsed.name, "chat_id": chat_id},
            )
            return
        await long_runner.run(chat_id=chat_id, cmd=parsed, message=message)
        logger.info(
            "telegram.long_running.dispatched",
            extra={"command": parsed.name, "chat_id": chat_id},
        )
        return

    incoming = IncomingMessage(chat_id=chat_id, user_id=user_id, text=text)
    outcome = dispatcher.handle(incoming)
    await _reply(message, outcome.reply)
    logger.info(
        "telegram.dispatch",
        extra={
            "kind": outcome.kind,
            "command": outcome.command,
            "chat_id": incoming.chat_id,
        },
    )


async def _handle_restart(
    message: Any,
    *,
    chat_id: int,
    dispatcher: Dispatcher,
    long_runner: LongRunningRunner | None,
    restart_fn: Callable[[], None] | None,
) -> None:
    """Authorize + refuse-if-busy + ack + schedule the re-exec."""
    if not dispatcher.authorizer.check(chat_id):
        await _reply(message, _CHAT_UNAUTHORIZED_REPLY)
        logger.info("telegram.restart.denied", extra={"chat_id": chat_id})
        return
    if long_runner is not None and long_runner.registry.any_in_flight():
        await _reply(message, _RESTART_BUSY)
        logger.info("telegram.restart.busy", extra={"chat_id": chat_id})
        return
    await _reply(message, _RESTART_ACK)
    logger.info("telegram.restart.scheduled", extra={"chat_id": chat_id})
    fn = restart_fn if restart_fn is not None else reexec_in_place
    # `call_later` fires on the running loop so the ack above is already
    # in Telegram's outbound queue by the time the restart runs. The
    # callback replaces the process image; nothing after it executes.
    asyncio.get_running_loop().call_later(_RESTART_EXEC_DELAY, fn)


async def _keep_typing(
    send_action: Any, interval: float = _TYPING_REFRESH_SECONDS
) -> None:
    """Re-send `ChatAction.TYPING` every `interval` seconds until cancelled.

    Telegram's single-shot typing indicator auto-expires after ~5s, so we
    refresh it every 4s while the pipeline runs. Individual send failures
    are swallowed (a transient network blip shouldn't kill the keep-alive
    loop); `CancelledError` is allowed to propagate so the caller's
    `finally` block cleans up deterministically.
    """
    while True:
        try:
            await send_action(ChatAction.TYPING)
        except Exception:
            logger.debug("telegram.chat.typing_refresh_failed", exc_info=True)
        await asyncio.sleep(interval)


async def route_chat(
    update: Any,
    _context: Any,
    *,
    pipeline: ChatPipeline | None = None,
    authorizer: Authorizer | None = None,
    intent_router: IntentRouter | None = None,
    long_runner: LongRunningRunner | None = None,
    skill_arg_resolver: SkillArgResolver | None = None,
) -> None:
    """Handle non-slash conversational messages.

    Flow:
      1. Drop silently if the update is missing chat/user/message.
      2. Authorize if an `Authorizer` is provided (shares the
         allowlist with `route_command`). Unauthorized chats get the
         same "unauthorized" reply the slash path uses.
      3. Route to `ChatPipeline.turn` when one is wired; else reply
         with the legacy stub so the bot remains responsive during
         incremental rollout.

    Hybrid typing indicator: when the incoming `chat` exposes
    `send_chat_action` (production SDK — FakeUpdates in tests don't),
    we post a "⏳ Thinking…" placeholder bubble, spawn a keep-alive task
    that refreshes `ChatAction.TYPING` every 4s, then delete the
    placeholder + send the final reply as a new message once the
    pipeline returns. (Editing the placeholder looks slicker but
    `editMessageText` does not clear `sendChatAction`, so the header
    typing animation lingers ~5s after the reply arrives — a jarring
    UX glitch. Sending a new message auto-stops the header indicator.)
    Tests using bare `SimpleNamespace` chats fall through to the legacy
    `_reply` flow unchanged.

    Never raises — the pipeline is stub-on-failure and this wrapper
    is the last line of defense for the long-poll event loop.
    """
    chat = getattr(update, "effective_chat", None)
    user = getattr(update, "effective_user", None)
    message = getattr(update, "effective_message", None)
    if chat is None or user is None or message is None:
        return
    text = getattr(message, "text", None) or ""

    if pipeline is None:
        await _reply(message, _CHAT_STUB_REPLY)
        return

    chat_id = int(chat.id)
    if authorizer is not None:
        decision = authorizer.check(chat_id)
        if not decision:
            await _reply(message, _CHAT_UNAUTHORIZED_REPLY)
            logger.info(
                "telegram.chat.denied",
                extra={"chat_id": chat_id},
            )
            return

    # Intent short-circuit: when the operator's free-form text matches
    # a declared skill intent AND the resolver can fill the skill's
    # argv, run the skill subprocess and skip the LLM entirely. This
    # is the deterministic half of the intent-classifier plan (§3.3
    # step 2) — a miss falls through to the regular pipeline path so
    # the LLM still answers everything else.
    if (
        intent_router is not None
        and long_runner is not None
        and skill_arg_resolver is not None
        and text
    ):
        hit = intent_router.match(text)
        if hit is not None:
            argv = skill_arg_resolver(hit)
            if argv is None:
                await _reply(
                    message,
                    f"Recognised intent for {hit.id!r} but the skill "
                    "is not configured in this deployment.",
                )
                return
            try:
                await long_runner.run_skill(
                    chat_id=chat_id,
                    skill_id=hit.id,
                    argv=argv,
                    message=message,
                )
            except Exception:
                logger.exception(
                    "telegram.chat.skill_dispatch_failed",
                    extra={"chat_id": chat_id, "skill_id": hit.id},
                )
                await _reply(message, f"internal error running {hit.id}")
            logger.info(
                "telegram.chat.intent_dispatched",
                extra={"chat_id": chat_id, "skill_id": hit.id},
            )
            return

    # Hybrid indicator: header ChatAction + inline placeholder bubble.
    # Gated on `send_chat_action` so unit tests (bare SimpleNamespace
    # chats) skip straight to the legacy `_reply` flow.
    placeholder, typing_task = await _start_typing_indicator(chat, message)

    try:
        try:
            reply = await pipeline.turn(str(chat_id), text)
        except Exception:
            logger.exception(
                "telegram.chat.pipeline_crashed", extra={"chat_id": chat_id}
            )
            reply = _CHAT_STUB_REPLY
    finally:
        await _stop_typing_indicator(typing_task)

    if not reply:
        return

    await _deliver_reply(message, placeholder, reply, chat_id=chat_id)
    logger.info(
        "telegram.chat.replied",
        extra={"chat_id": chat_id, "reply_bytes": len(reply.encode("utf-8"))},
    )


async def _start_typing_indicator(
    chat: Any, message: Any
) -> tuple[Any, asyncio.Task[None] | None]:
    """Post a "Thinking…" placeholder and spawn the header keep-alive task.

    Returns `(placeholder, task)` — both may be `None` if the chat does
    not expose `send_chat_action` (i.e. unit-test fakes), in which case
    the caller falls back to the legacy reply flow.
    """
    send_action = getattr(chat, "send_chat_action", None)
    if send_action is None:
        return None, None
    placeholder: Any
    try:
        placeholder = await message.reply_text(_CHAT_THINKING_PLACEHOLDER)
    except Exception:
        logger.debug("telegram.chat.placeholder_failed", exc_info=True)
        placeholder = None
    typing_task = asyncio.create_task(_keep_typing(send_action))
    return placeholder, typing_task


async def _stop_typing_indicator(typing_task: asyncio.Task[None] | None) -> None:
    if typing_task is None:
        return
    typing_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await typing_task


async def _deliver_reply(
    message: Any, placeholder: Any, reply: str, *, chat_id: int
) -> None:
    """Delete the placeholder, then send the reply as a new message.

    Telegram UX quirk: `editMessageText` does NOT clear
    `sendChatAction` state, so the header "typing…" animation lingers
    up to ~5s after an edit-in-place reply arrives. Sending a new
    message DOES auto-stop the header indicator instantly. We trade
    the morph-in-place UX for a tight indicator-stop: delete the
    "⏳ Thinking…" bubble (best-effort) and send the real reply via
    the normal `_reply` flow so chunking + the 4KB cap are reused.

    Placeholder delete failures are swallowed — leaves an orphan
    "Thinking…" bubble in the chat, but still ships the reply.
    """
    delete = getattr(placeholder, "delete", None) if placeholder is not None else None
    if delete is not None:
        try:
            await delete()
        except Exception:
            logger.debug(
                "telegram.chat.placeholder_delete_failed",
                exc_info=True,
                extra={"chat_id": chat_id},
            )
    await _reply(message, reply)


def _chunk(text: str, limit: int = MAX_TELEGRAM_CHARS) -> list[str]:
    """Split `text` into chunks ≤ `limit` chars, preferring newline cuts.

    Telegram's message cap is 4096 characters; we leave headroom for
    the bot to prepend status prefixes without re-chunking. A single
    line longer than `limit` is hard-sliced at `limit` — the operator
    sees the whole reply across multiple messages in order.
    """
    if limit <= 0:
        return [text] if text else []
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def _reply(message: _Replyable, text: str) -> None:
    for chunk in _chunk(text):
        await message.reply_text(chunk)


class AsyncioSubprocessRunner:
    """Production `SubprocessRunner` — argv-only (no shell interpolation).

    Uses asyncio's argv-based subprocess spawner: each element of
    `argv` is passed as a separate exec argument, so operator input
    (e.g. a CT id) is never concatenated into a shell string. Stdout
    and stderr are merged onto one pipe so the final edit shows the
    two streams interleaved in the order the child produced them.
    """

    async def run(self, argv: list[str], *, cwd: Path) -> tuple[int, str]:
        spawn = asyncio.create_subprocess_exec  # argv-only; no shell=True.
        proc = await spawn(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out_bytes, _ = await proc.communicate()
        output = (
            out_bytes.decode("utf-8", errors="replace") if out_bytes else ""
        )
        rc = proc.returncode if proc.returncode is not None else 0
        return (rc, output)


def build_long_running_runner(
    workspace: Path,
    *,
    subprocess_runner: SubprocessRunner | None = None,
    registry: InFlightRegistry | None = None,
    vault_root: Path | None = None,
) -> LongRunningRunner:
    """Assemble a `LongRunningRunner` with sensible defaults.

    Defaults wire `AsyncioSubprocessRunner` (shells out via argv) and
    a fresh `InFlightRegistry`. `vault_root` is the Obsidian vault
    path threaded through to `/brief` — when `None`, `/brief` replies
    with a "not configured" hint. Tests pass fakes for all of it.
    """
    runner = (
        subprocess_runner
        if subprocess_runner is not None
        else AsyncioSubprocessRunner()
    )
    return LongRunningRunner(
        workspace, runner=runner, registry=registry, vault_root=vault_root
    )


def build_skill_arg_resolver(
    cfg: AegisConfig, *, python_executable: str | None = None,
) -> SkillArgResolver:
    """Return a resolver that turns a ``SkillDescriptor`` into a runnable argv.

    Narrow by design. Today only ``{vault_root}`` is known — pulled
    from ``cfg.vault_indexing.vault_root``. Any unknown placeholder
    produces ``None`` so the caller can tell the operator the skill
    isn't configured in this deployment instead of spawning a
    subprocess that would crash on ``KeyError`` at format time.

    The leading ``python`` token in ``argv_template`` is swapped for
    the current interpreter (``sys.executable`` by default) so dispatches
    land on the same venv as ``/brief``. Real ``python`` on PATH might
    pick a system interpreter that lacks our vendored deps.
    """
    python = python_executable if python_executable is not None else sys.executable

    def resolve(descriptor: SkillDescriptor) -> list[str] | None:
        if not descriptor.tools:
            return None
        spec = descriptor.tools[0]
        values: dict[str, str] = {}
        for name in spec.placeholders():
            if name == "vault_root":
                vr = cfg.vault_indexing.vault_root
                if vr is None:
                    return None
                values[name] = str(vr)
            else:
                return None
        resolved = [token.format_map(values) for token in spec.argv_template]
        if resolved and resolved[0] == "python":
            resolved[0] = python
        return resolved

    return resolve


def build_intent_router(catalog_dir: Path | None = None) -> IntentRouter | None:
    """Load the skill catalog and wrap it in an ``IntentRouter``.

    Returns ``None`` if the catalog directory is missing — chat just
    skips the intent short-circuit in that case. Load failures (a
    malformed YAML descriptor) propagate: a broken catalog is an
    operator error, not a silent degrade.
    """
    catalog = catalog_dir if catalog_dir is not None else _CATALOG_DIR
    if not catalog.is_dir():
        return None
    registry = SkillRegistry.from_directory(catalog)
    return IntentRouter(registry)


def build_chat_pipeline(
    cfg: AegisConfig,
    *,
    events: EventStream | None = None,
    vault_loader: VaultBodyLoader | None = None,
    local_ready: LocalReadyProbe | None = None,
) -> ChatPipeline | None:
    """Assemble a production `ChatPipeline` from config, or return `None`.

    Picks the SMART-tier chat client through `ModelRouter.route(SMART)`:

    1. `cfg.models.prefer_local=True` and Ollama reachable → local
       (`OllamaClient` with `cfg.models.smart_local`).
    2. `OPENROUTER_API_KEY` set → remote (`OpenRouterClient` with
       `cfg.models.smart`).
    3. Otherwise, if Ollama is reachable → local fallback (degraded).
    4. Otherwise → `None`, so `route_chat` falls back to its stub reply.

    Embedder follows the same liveness signal: `Bgem3Embedder`
    (bge-m3 via Ollama, dim=1024) when local is reachable, else
    `FakeEmbedder` so Tier 2 retrieval stays cheap and offline-safe.

    Always-present deps:

    * `Tier1Loader(workspace)` — tolerates missing IDENTITY/USER.md.
    * `Tier3Store()` — in-memory rolling turn store.
    * `RecallPolicy` — catches tier 2 errors internally, never blocks.
    * `ContextBuilder(tier1, tier3)` — pure byte-budget enforcement.

    `local_ready` is an injectable probe so tests can pin the
    local-availability decision without touching the network.

    Never raises — configuration faults degrade to `None`.
    """
    router = ModelRouter(cfg, local_ready=local_ready)
    local_is_up = router.is_local_ready()
    target = router.route(ModelTier.SMART)

    model: ModelClient | None
    if target.provider == "ollama":
        if not local_is_up:
            logger.info(
                "chat.pipeline.disabled", extra={"reason": "no_local_no_remote"}
            )
            return None
        try:
            model = OllamaClient(cfg)
        except OllamaHostError:
            logger.info("chat.pipeline.disabled", extra={"reason": "ollama_host"})
            return None
    elif target.provider == "openrouter":
        try:
            model = OpenRouterClient(cfg)
        except OpenRouterConfigError:
            logger.info(
                "chat.pipeline.disabled", extra={"reason": "openrouter_config"}
            )
            return None
    else:  # defensive — router currently only emits ollama/openrouter
        logger.error(
            "chat.pipeline.unknown_provider", extra={"provider": target.provider}
        )
        return None

    embedder = build_embedder(try_bgem3=local_is_up)
    try:
        tier2 = Tier2Store(cfg.storage.memory_db, embedder)
    except Exception:
        logger.exception("chat.pipeline.tier2_unavailable")
        return None
    tier1 = Tier1Loader(cfg.storage.workspace)
    tier3 = Tier3Store()
    recall = RecallPolicy(tier2=tier2, vault_loader=vault_loader)
    builder = ContextBuilder(tier1, tier3)
    logger.info(
        "chat.pipeline.enabled",
        extra={
            "provider": target.provider,
            "model": target.model,
            "degraded": target.degraded,
            "embedder": type(embedder).__name__,
        },
    )
    return ChatPipeline(
        tier1=tier1,
        tier3=tier3,
        recall=recall,
        builder=builder,
        model=model,
        model_name=target.model,
        events=events,
    )


def _startup_message_body(cfg: AegisConfig, *, now: datetime | None = None) -> str:
    """Compose the "AEGIS online" notification body. Pure function — tested.

    Includes a UTC timestamp, the configured SMART-tier model, and
    whether the conversational pipeline is wired (OpenRouter key
    present) or running in stub mode. Operator uses /status for the
    full picture; this is just a heartbeat-on-boot signal.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    chat_status = (
        "wired" if cfg.providers.openrouter_api_key else "stub (no OPENROUTER_API_KEY)"
    )
    return (
        f"🟢 AEGIS online — {stamp}\n"
        f"model: {cfg.models.smart}\n"
        f"chat: {chat_status}"
    )


async def _send_startup_message(cfg: AegisConfig, bot: Any) -> None:
    """Notify every allowlisted chat once per process start.

    Called from python-telegram-bot's `post_init` hook. Per-chat send
    failures (chat blocked, bot never DM'd) are logged and swallowed;
    startup must not hang on operator messaging. Empty allowlist is
    a no-op.
    """
    if not cfg.telegram.user_allowlist:
        return
    body = _startup_message_body(cfg)
    for chat_id in cfg.telegram.user_allowlist:
        try:
            await bot.send_message(chat_id=chat_id, text=body)
        except Exception:
            logger.exception(
                "telegram.startup.notify_failed",
                extra={"chat_id": chat_id},
            )


def _build_vault_trio(
    cfg: AegisConfig,
) -> tuple[VaultIndexer | None, Tier2Store | None, VaultState | None]:
    """Construct the `/vault` trio (indexer, tier2, state) if enabled.

    Returns `(None, None, None)` when `vault_indexing.vault_root` is
    not configured so callers can thread the values unconditionally
    and `build_read_only_handlers` will simply skip `/vault` registration.
    Constructing the store is wrapped — a broken sqlite file degrades
    to "vault disabled" rather than crashing the bot.
    """
    if not cfg.vault_indexing.is_enabled():
        return None, None, None
    try:
        tier2 = Tier2Store(cfg.storage.memory_db, build_embedder())
    except Exception:
        logger.exception("vault.tier2_unavailable")
        return None, None, None
    indexer = VaultIndexer(tier2=tier2, config=cfg.vault_indexing)
    return indexer, tier2, VaultState()


def build_scheduler(
    cfg: AegisConfig,
    *,
    long_runner: LongRunningRunner,
    skill_arg_resolver: SkillArgResolver,
    bot: Any,
    events: EventStream,
    registry: SkillRegistry | None = None,
    subprocess_runner: SubprocessRunner | None = None,
    delivery_chat_id: int | None = None,
) -> tuple[SchedulerEngine, ScheduledJobStore] | None:
    """Assemble the scheduler stack. Returns `None` when disabled.

    Scheduler is disabled when the skill catalog is missing (nothing
    to schedule) or the allowlist is empty (nowhere to deliver
    output). In both cases we skip construction entirely so
    operators see `unknown_command` for `/cron` rather than a stub
    reply — mirrors the `/recall` / `/vault` pattern.

    Delivery is pinned to a single chat id — the first entry in
    `cfg.telegram.user_allowlist` unless `delivery_chat_id` is
    explicitly supplied by the caller (tests). Matches the
    single-operator design pin: scheduled jobs push to one chat,
    not broadcast.

    `is_busy` is wired to `long_runner.registry.any_in_flight()`
    so a `/apply`, `/harness`, or `/brief` that's running when a
    cron window opens emits `scheduler.skipped_busy` and advances
    the job to the next tick rather than stomping the operator's
    in-flight subprocess.
    """
    catalog = _CATALOG_DIR
    if registry is None:
        if not catalog.is_dir():
            return None
        registry = SkillRegistry.from_directory(catalog)
    if delivery_chat_id is None:
        if not cfg.telegram.user_allowlist:
            return None
        delivery_chat_id = cfg.telegram.user_allowlist[0]
    runner = (
        subprocess_runner
        if subprocess_runner is not None
        else AsyncioSubprocessRunner()
    )

    def _resolve_argv(
        descriptor: SkillDescriptor, args: tuple[str, ...]
    ) -> list[str] | None:
        base = skill_arg_resolver(descriptor)
        if base is None:
            return None
        return base + list(args)

    async def _deliver(text: str) -> None:
        # Delivery failures (blocked bot, 403) must not crash the
        # engine — log + swallow so the tick loop keeps running.
        try:
            await bot.send_message(chat_id=delivery_chat_id, text=text)
        except Exception:
            logger.exception(
                "scheduler.delivery_failed",
                extra={"chat_id": delivery_chat_id},
            )

    job_runner = JobRunner(
        registry=registry,
        subprocess_runner=runner,
        workspace=cfg.storage.workspace,
        argv_resolver=_resolve_argv,
        deliver=_deliver,
    )
    store = ScheduledJobStore(cfg.storage.memory_db)
    seed_system_jobs(store, now=system_clock())
    engine = SchedulerEngine(
        store=store,
        events=events,
        invoker=job_runner,
        clock=system_clock,
        sleeper=system_sleeper,
        is_busy=lambda _job: long_runner.registry.any_in_flight(),
    )
    return engine, store


def build_application(  # noqa: PLR0915 - top-level assembly seam; each step is one statement
    cfg: AegisConfig,
    *,
    dispatcher: Dispatcher | None = None,
    sessions_dir: Path | None = None,
    clock: Clock | None = None,
    events: EventStream | None = None,
    recall_resolver: ColdRefResolver | None = None,
    recall_reader: ColdStorageReader | None = None,
    vault_loader: VaultBodyLoader | None = None,
    long_runner: LongRunningRunner | None = None,
    chat_pipeline: ChatPipeline | None = None,
    vault_indexer: VaultIndexer | None = None,
    vault_tier2: Tier2Store | None = None,
    vault_state: VaultState | None = None,
    intent_router: IntentRouter | None = None,
    skill_arg_resolver: SkillArgResolver | None = None,
) -> Any:
    """Construct a `telegram.ext.Application` wired to our dispatcher.

    The SDK is imported lazily so unit tests — which never call this
    function — don't need `python-telegram-bot` installed. Callers
    drive the returned `Application` with `.run_polling()` (production)
    or `.initialize()` + `.process_update(fake_update)` (integration).

    `long_runner` defaults to a production `LongRunningRunner` so
    `/apply` and `/harness` are wired out of the box. Tests that want
    to stub the subprocess layer pass their own runner.

    `chat_pipeline` default-constructs via `build_chat_pipeline(cfg, ...)`
    when `OPENROUTER_API_KEY` is configured; otherwise it stays `None`
    and `route_chat` falls back to the legacy placeholder reply. Tests
    inject a fake pipeline directly.

    A `post_init` hook sends a one-shot "🟢 AEGIS online" heartbeat to
    every allowlisted chat once the Application is ready — operator's
    signal that a restart landed. Failures per-chat are logged and
    swallowed (an operator who never DM'd the bot returns 403 and we
    move on).
    """
    # Lazy import: keeps the `telegram.ext` import out of module load
    # so unit tests don't need the SDK imported to drive FakeUpdates.
    from telegram.ext import (  # noqa: PLC0415
        ApplicationBuilder,
        MessageHandler,
        filters,
    )

    if not cfg.telegram.bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
    if events is None:
        # Default-construct a shared stream so write slashes, chat turn
        # telemetry, and the D2 verdict gate all land on the same shard.
        # Tests that want silent mode pass their own `events=None` via
        # `build_dispatcher` / `build_chat_pipeline` directly.
        events = EventStream(sessions_dir or cfg.storage.sessions_dir)
    if vault_indexer is None and vault_tier2 is None:
        vault_indexer, vault_tier2, vault_state_default = _build_vault_trio(cfg)
        if vault_state is None:
            vault_state = vault_state_default
    if (
        vault_loader is None
        and cfg.vault_indexing.is_enabled()
        and cfg.vault_indexing.vault_root is not None
    ):
        vault_loader = FilesystemVaultBodyLoader(cfg.vault_indexing.vault_root)
    if long_runner is None:
        long_runner = build_long_running_runner(
            cfg.storage.workspace,
            vault_root=cfg.vault_indexing.vault_root,
        )
    if skill_arg_resolver is None:
        skill_arg_resolver = build_skill_arg_resolver(cfg)

    # Scheduler has to run inside the same event loop as long-poll, so
    # we build the stack up-front but only wire the tick loop inside
    # `_post_init`. Construction order matters: the scheduler's
    # `ScheduledJobStore` must exist before `build_dispatcher` so
    # `/cron` can close over it. The engine task is spawned later.
    scheduler_engine: SchedulerEngine | None = None
    scheduler_store: ScheduledJobStore | None = None
    # `_DeferredBot` forwards delivery to the real SDK `bot` handle
    # once `ApplicationBuilder.build()` returns. Before that, pushes
    # are logged + dropped (should never happen in practice — the
    # engine task only starts in `_post_init`, after `bot` is bound).
    scheduler_bot_ref: list[Any] = [None]

    class _DeferredBot:
        async def send_message(self, *, chat_id: int, text: str) -> None:
            real = scheduler_bot_ref[0]
            if real is None:
                logger.warning(
                    "scheduler.delivery_skipped",
                    extra={"reason": "bot_not_ready"},
                )
                return
            await real.send_message(chat_id=chat_id, text=text)

    # Build the skill registry once so both the scheduler and the
    # dispatcher's `/cron` handler share it — lets `/cron add` reject
    # unknown/unschedulable skills at add time instead of letting the
    # engine fail silently on every tick.
    skill_registry: SkillRegistry | None = None
    if _CATALOG_DIR.is_dir():
        skill_registry = SkillRegistry.from_directory(_CATALOG_DIR)

    scheduler_stack = build_scheduler(
        cfg,
        long_runner=long_runner,
        skill_arg_resolver=skill_arg_resolver,
        bot=_DeferredBot(),
        events=events,
        registry=skill_registry,
    )
    if scheduler_stack is not None:
        scheduler_engine, scheduler_store = scheduler_stack

    if dispatcher is None:
        authorizer = Authorizer(tuple(cfg.telegram.user_allowlist))
        # Teach /help about slashes handled outside the sync dispatcher
        # (long-running set + /restart). Descriptions come from the
        # shared DEFAULT_COMMAND_HELP via help_descriptions_for, so
        # they stay identical to the in-dispatcher help.
        extra_help_commands = {*long_runner.commands, "/restart"}
        extra_help = {
            k: v
            for k, v in help_descriptions_for(extra_help_commands).items()
        }
        dispatcher = build_dispatcher(
            cfg.storage.workspace,
            authorizer=authorizer,
            sessions_dir=sessions_dir or cfg.storage.sessions_dir,
            clock=clock,
            cfg=cfg,
            router=ModelRouter(cfg),
            events=events,
            recall_resolver=recall_resolver,
            recall_reader=recall_reader,
            vault_loader=vault_loader,
            vault_indexer=vault_indexer,
            vault_tier2=vault_tier2,
            vault_state=vault_state,
            scheduler_store=scheduler_store,
            skill_registry=skill_registry,
            extra_command_help=extra_help,
        )
    if chat_pipeline is None:
        chat_pipeline = build_chat_pipeline(cfg, events=events, vault_loader=vault_loader)
    if intent_router is None:
        intent_router = build_intent_router()

    async def _post_init(application: Any) -> None:
        # Fire a one-shot "AEGIS online" notification to every
        # allowlisted chat so the operator gets a visible heartbeat
        # every time the bot restarts (config reload, OS reboot, etc.).
        await _send_startup_message(cfg, application.bot)
        # Kick off the initial vault reindex on the threadpool so we
        # don't block the long-poll event loop while embedding a large
        # corpus. Failures are logged + swallowed — a broken vault must
        # not stop the bot from accepting messages.
        if vault_indexer is not None and vault_state is not None:
            try:
                result = await asyncio.to_thread(vault_indexer.reindex)
                vault_state.last_result = result
                logger.info(
                    "vault.initial_reindex",
                    extra={
                        "added": result.added,
                        "updated": result.updated,
                        "skipped": result.skipped,
                        "pruned": result.pruned,
                        "errors": len(result.errors),
                    },
                )
            except Exception:
                logger.exception("vault.initial_reindex_failed")
        if scheduler_engine is not None:
            # Spawn the scheduler tick loop as a background task. The
            # `run()` coroutine never returns normally — it loops on
            # `asyncio.sleep(tick_interval)` and unwinds on
            # `CancelledError` at shutdown. Stash the task on the
            # application so GC doesn't collect it while Python is
            # still running.
            application.scheduler_task = asyncio.create_task(
                scheduler_engine.run()
            )
            logger.info("scheduler.started")

    app = (
        ApplicationBuilder()
        .token(cfg.telegram.bot_token)
        .post_init(_post_init)
        .build()
    )
    scheduler_bot_ref[0] = app.bot
    # Share the same Authorizer between slash + chat paths so the
    # allowlist is a single source of truth.
    chat_authorizer = dispatcher.authorizer

    async def _on_command(update: Any, context: Any) -> None:
        await route_command(
            update, context, dispatcher=dispatcher, long_runner=long_runner
        )

    async def _on_chat(update: Any, context: Any) -> None:
        await route_chat(
            update,
            context,
            pipeline=chat_pipeline,
            authorizer=chat_authorizer,
            intent_router=intent_router,
            long_runner=long_runner,
            skill_arg_resolver=skill_arg_resolver,
        )

    app.add_handler(MessageHandler(filters.COMMAND, _on_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_chat))
    return app


__all__ = [
    "MAX_TELEGRAM_CHARS",
    "AsyncioSubprocessRunner",
    "SkillArgResolver",
    "build_application",
    "build_chat_pipeline",
    "build_dispatcher",
    "build_intent_router",
    "build_long_running_runner",
    "build_skill_arg_resolver",
    "route_chat",
    "route_command",
]
