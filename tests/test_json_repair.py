"""Phase 11 Track A4 — deterministic JSON repair salvage.

Pins:

* Fenced JSON (```json ... ```) is unwrapped and recovered.
* Prose wrapped around a JSON object is stripped and recovered.
* Trailing commas before `}` / `]` are removed and recovered.
* Truly broken input (no braces, unbalanced braces, garbage) returns None.
* Already-valid input passes through unchanged (parses to the same value).

`repair_json` never guesses at semantics — it only strips wrapper noise.
"""
from __future__ import annotations

import json

import pytest

from runtime.llm.structured_output import repair_json

pytestmark = pytest.mark.unit


def test_recovers_fenced_json() -> None:
    content = '```json\n{"intent": "ask", "confidence": 0.9}\n```'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == {"intent": "ask", "confidence": 0.9}


def test_recovers_fenced_json_without_language_tag() -> None:
    content = '```\n{"intent": "ask", "confidence": 0.9}\n```'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == {"intent": "ask", "confidence": 0.9}


def test_recovers_prose_wrapped_json() -> None:
    content = 'Here is the result: {"intent": "ask", "confidence": 0.9} hope that helps'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == {"intent": "ask", "confidence": 0.9}


def test_recovers_trailing_comma_in_object() -> None:
    content = '{"a": 1,}'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == {"a": 1}


def test_recovers_trailing_comma_in_nested_array() -> None:
    # Top-level extraction requires braces, so the array must live inside
    # an object.
    content = '{"items": [1, 2,]}'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == {"items": [1, 2]}


def test_no_braces_returns_none() -> None:
    assert repair_json("just some plain text, no json here") is None


def test_unbalanced_braces_returns_none() -> None:
    assert repair_json('{"intent": "ask"') is None


def test_garbage_inside_braces_returns_none() -> None:
    assert repair_json("{not json at all}") is None


def test_already_valid_json_passes_through_unchanged() -> None:
    content = '{"intent": "ask", "confidence": 0.9}'
    repaired = repair_json(content)
    assert repaired is not None
    assert json.loads(repaired) == json.loads(content)
