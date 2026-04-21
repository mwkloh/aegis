"""Phase 7 §4.3 — `Dispatcher` contract.

Pins:

* Denied chats never reach a handler (no leak of command surface).
* Non-command text → typed `not_a_command` reply.
* Unknown slash → typed `unknown_command` reply with the attempted
  command name (for audit).
* Handler exceptions → typed `handler_error`, never propagated.
* Handler names are case-insensitive and tolerate `/cmd` or `cmd`
  in the registry.
* `/cmd@BotName args` strips the `@BotName` suffix (Telegram routes
  group-chat commands this way).
* Shell-style quoting is honored; malformed quoting falls back to
  a whitespace split instead of raising.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.chat.telegram import (
    Authorizer,
    Dispatcher,
    DispatchOutcome,
    IncomingMessage,
    ParsedCommand,
    parse_command,
)

pytestmark = pytest.mark.unit


def _msg(
    text: str = "/ping", chat_id: int = 111, user_id: int = 222
) -> IncomingMessage:
    return IncomingMessage(chat_id=chat_id, user_id=user_id, text=text)


# --- parse_command -------------------------------------------------------


def test_parse_command_plain_slash() -> None:
    parsed = parse_command("/pending")
    assert parsed == ParsedCommand(name="/pending", args=())


def test_parse_command_with_args() -> None:
    parsed = parse_command("/approve IMP-abc12345 ship it")
    assert parsed is not None
    assert parsed.name == "/approve"
    assert parsed.args == ("IMP-abc12345", "ship", "it")


def test_parse_command_quoted_rationale() -> None:
    parsed = parse_command('/approve IMP-abc12345 "ship it — tests green"')
    assert parsed is not None
    assert parsed.args == ("IMP-abc12345", "ship it — tests green")


def test_parse_command_malformed_quote_falls_back() -> None:
    """An unclosed quote should not raise — we still hand the handler
    something, so it can emit its own typed error."""
    parsed = parse_command('/approve IMP-abc12345 "unclosed')
    assert parsed is not None
    assert parsed.name == "/approve"
    # whitespace-split fallback, quotes preserved verbatim
    assert parsed.args == ("IMP-abc12345", '"unclosed')


def test_parse_command_bot_suffix_stripped() -> None:
    parsed = parse_command("/pending@AegisBot")
    assert parsed is not None
    assert parsed.name == "/pending"


def test_parse_command_non_command_returns_none() -> None:
    assert parse_command("hello there") is None
    assert parse_command("") is None
    assert parse_command("   ") is None


def test_parse_command_lone_slash_returns_none() -> None:
    assert parse_command("/") is None
    assert parse_command("/ ") is None


def test_parse_command_is_case_insensitive() -> None:
    parsed = parse_command("/PENDING")
    assert parsed is not None
    assert parsed.name == "/pending"


# --- Dispatcher auth -----------------------------------------------------


def test_denied_chat_never_reaches_handler() -> None:
    calls: list[str] = []

    def handler(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        calls.append("boom")
        return "should not run"

    auth = Authorizer((999,))
    disp = Dispatcher(authorizer=auth, handlers={"/pending": handler})
    outcome = disp.handle(_msg(text="/pending", chat_id=111))
    assert outcome.kind == "denied"
    assert outcome.reply == "unauthorized"
    assert outcome.error == "not_allowed"
    assert calls == []


def test_empty_allowlist_denies_all_even_with_handlers() -> None:
    auth = Authorizer(())
    disp = Dispatcher(authorizer=auth, handlers={"/ping": lambda m, c: "pong"})
    outcome = disp.handle(_msg(text="/ping"))
    assert outcome.kind == "denied"
    assert outcome.error == "empty_allowlist"


# --- Dispatcher routing --------------------------------------------------


def test_ok_routes_to_handler() -> None:
    captured: list[ParsedCommand] = []

    def handler(msg: IncomingMessage, cmd: ParsedCommand) -> str:
        captured.append(cmd)
        assert msg.chat_id == 111
        return f"ran {cmd.name} with {len(cmd.args)} args"

    auth = Authorizer((111,))
    disp = Dispatcher(authorizer=auth, handlers={"/pending": handler})
    outcome = disp.handle(_msg(text="/pending foo bar"))
    assert outcome.kind == "ok"
    assert outcome.command == "/pending"
    assert outcome.reply == "ran /pending with 2 args"
    assert captured[0].args == ("foo", "bar")


def test_registry_accepts_slashless_names() -> None:
    auth = Authorizer((111,))
    disp = Dispatcher(
        authorizer=auth, handlers={"ping": lambda m, c: "pong"}
    )
    assert disp.commands == ("/ping",)
    outcome = disp.handle(_msg(text="/ping"))
    assert outcome.kind == "ok"


def test_unknown_command_kind_carries_attempted_name() -> None:
    auth = Authorizer((111,))
    disp = Dispatcher(authorizer=auth, handlers={"/pending": lambda m, c: ""})
    outcome = disp.handle(_msg(text="/nope"))
    assert outcome.kind == "unknown_command"
    assert outcome.command == "/nope"
    assert "unknown command" in outcome.reply


def test_not_a_command_for_free_text() -> None:
    auth = Authorizer((111,))
    disp = Dispatcher(authorizer=auth, handlers={})
    outcome = disp.handle(_msg(text="what's next?"))
    assert outcome.kind == "not_a_command"
    assert outcome.command is None


def test_handler_exception_becomes_typed_error_reply() -> None:
    def boom(_msg: IncomingMessage, _cmd: ParsedCommand) -> str:
        raise RuntimeError("db is gone")

    auth = Authorizer((111,))
    disp = Dispatcher(authorizer=auth, handlers={"/pending": boom})
    outcome = disp.handle(_msg(text="/pending"))
    assert outcome.kind == "handler_error"
    assert outcome.command == "/pending"
    assert "db is gone" in (outcome.error or "")
    assert "internal error" in outcome.reply


def test_outcome_is_frozen_model() -> None:
    outcome = DispatchOutcome(kind="ok", chat_id=1, reply="hi")
    with pytest.raises(ValidationError, match="frozen"):
        outcome.reply = "bye"  # type: ignore[misc]


def test_bot_suffix_routes_like_plain_command() -> None:
    auth = Authorizer((111,))
    disp = Dispatcher(authorizer=auth, handlers={"/ping": lambda m, c: "pong"})
    outcome = disp.handle(_msg(text="/ping@AegisBot"))
    assert outcome.kind == "ok"
    assert outcome.reply == "pong"
