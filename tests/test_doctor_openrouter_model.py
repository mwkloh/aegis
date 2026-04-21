"""Doctor row that catches an ``openrouter/`` prefix on ``MODEL_CODING``.

Network paths (catalog fetch, etc.) are exercised by integration runs of
``aegis doctor``. This file pins the pure-function branches so a regression
that *removes* the prefix check fails fast.
"""
from __future__ import annotations

import pytest

from scripts.doctor import _check_openrouter_coding_model


@pytest.mark.unit
def test_returns_none_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_CODING", "x-ai/grok-4.1-fast")
    assert _check_openrouter_coding_model() is None


@pytest.mark.unit
def test_returns_none_when_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("MODEL_CODING", raising=False)
    monkeypatch.delenv("MODEL_SMART", raising=False)
    assert _check_openrouter_coding_model() is None


@pytest.mark.unit
def test_warns_on_bad_openrouter_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a network call: the bad-prefix branch should warn before probing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_CODING", "openrouter/x-ai/grok-4.1-fast")
    row = _check_openrouter_coding_model()
    assert row is not None
    label, ok, detail, severity = row
    assert label == "openrouter:coding_model:openrouter/x-ai/grok-4.1-fast"
    assert ok is False
    assert "invalid 'openrouter/' prefix" in detail
    assert severity == "warn"


@pytest.mark.unit
def test_falls_back_to_model_smart(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MODEL_CODING is unset, MODEL_SMART is the resolved fallback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("MODEL_CODING", raising=False)
    monkeypatch.setenv("MODEL_SMART", "openrouter/x-ai/grok-4.1-fast")
    row = _check_openrouter_coding_model()
    assert row is not None
    assert "openrouter/x-ai/grok-4.1-fast" in row[0]
    assert row[3] == "warn"
