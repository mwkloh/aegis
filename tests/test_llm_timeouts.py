"""One read-timeout override, honoured by every provider client.

If the override reached only Ollama, a cross-provider results table would mix
capability-measured local rows with budget-capped OpenRouter rows and present
them as comparable.
"""
from __future__ import annotations

import pytest

from runtime.llm.telemetry import PRODUCTION_READ_TIMEOUT_S
from runtime.llm.timeouts import read_timeout_override, resolve_read_timeout

pytestmark = pytest.mark.unit


def test_without_override_the_callers_default_is_kept() -> None:
    assert resolve_read_timeout(PRODUCTION_READ_TIMEOUT_S) == PRODUCTION_READ_TIMEOUT_S
    assert resolve_read_timeout(12.5) == 12.5


def test_override_replaces_the_default_for_any_caller() -> None:
    with read_timeout_override(300.0):
        assert resolve_read_timeout(PRODUCTION_READ_TIMEOUT_S) == 300.0
        assert resolve_read_timeout(12.5) == 300.0


def test_override_is_reverted_on_exit() -> None:
    with read_timeout_override(300.0):
        pass
    assert resolve_read_timeout(PRODUCTION_READ_TIMEOUT_S) == PRODUCTION_READ_TIMEOUT_S


def test_override_is_reverted_when_the_body_raises() -> None:
    with pytest.raises(RuntimeError), read_timeout_override(300.0):
        raise RuntimeError("boom")
    assert resolve_read_timeout(PRODUCTION_READ_TIMEOUT_S) == PRODUCTION_READ_TIMEOUT_S


def test_both_clients_resolve_through_the_same_override() -> None:
    """The point of sharing: one context manager covers every provider."""
    from runtime.llm.clients.ollama_client import _timeout_for_call as ollama_timeout
    from runtime.llm.clients.openrouter_client import (
        _timeout_for_call as openrouter_timeout,
    )

    with read_timeout_override(300.0):
        assert ollama_timeout().read == 300.0
        assert openrouter_timeout().read == 300.0

    assert ollama_timeout().read == PRODUCTION_READ_TIMEOUT_S
    assert openrouter_timeout().read == PRODUCTION_READ_TIMEOUT_S
