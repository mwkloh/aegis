"""Phase 7 §4.3 — `Authorizer` contract.

Pins:

* Empty allowlist → **deny** (with reason `empty_allowlist`).
* `chat_id` in allowlist → allow.
* `chat_id` not in allowlist → deny with reason `not_allowed`.
* Dupes in allowlist are collapsed, order preserved.
* `AuthDecision` is frozen; `bool(decision) == decision.allowed`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from runtime.chat.telegram import AuthDecision, Authorizer

pytestmark = pytest.mark.unit


def test_empty_allowlist_denies_all() -> None:
    auth = Authorizer(())
    decision = auth.check(12345)
    assert decision.allowed is False
    assert decision.reason == "empty_allowlist"
    assert bool(decision) is False


def test_chat_id_in_allowlist_is_allowed() -> None:
    auth = Authorizer((12345, 67890))
    decision = auth.check(12345)
    assert decision.allowed is True
    assert decision.reason is None
    assert bool(decision) is True


def test_chat_id_not_in_allowlist_is_denied() -> None:
    auth = Authorizer((12345,))
    decision = auth.check(99999)
    assert decision.allowed is False
    assert decision.reason == "not_allowed"


def test_allowlist_dedupes_and_preserves_order() -> None:
    auth = Authorizer((1, 2, 1, 3, 2))
    assert auth.allowlist == (1, 2, 3)


def test_auth_decision_is_frozen() -> None:
    decision = AuthDecision(allowed=True, chat_id=1, reason=None)
    with pytest.raises(ValidationError, match="frozen"):
        decision.allowed = False  # type: ignore[misc]


def test_auth_decision_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AuthDecision(  # type: ignore[call-arg]
            allowed=True, chat_id=1, reason=None, bogus="x"
        )


def test_denied_decision_carries_chat_id_for_audit() -> None:
    """Caller emits `governance.denied` with this `chat_id` — so it
    must round-trip exactly what was checked, not be mutated."""
    auth = Authorizer((1,))
    decision = auth.check(42)
    assert decision.chat_id == 42
    assert decision.allowed is False
