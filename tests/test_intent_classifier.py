from __future__ import annotations

import pytest

from runtime.intent import IntentClassifier

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("echo hello", "echo"),
        ("ECHO Hello World", "echo"),
        ("ping", "ping"),
        ("  ping  ", "ping"),
        ("hello there", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_known_intents(text: str, expected: str) -> None:
    classifier = IntentClassifier()
    result = classifier.classify(text)
    assert result.intent == expected
    if expected != "unknown":
        assert result.confidence >= 0.9
    else:
        assert result.confidence == 0.0


def test_classify_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        IntentClassifier().classify(None)  # type: ignore[arg-type]
