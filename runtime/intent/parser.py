"""Re-export of the strict intent JSON parser.

The parser implementation lives in `classifier.py` to avoid an import cycle
between the parser and `IntentClassification`. This module exists so callers
can keep importing `from runtime.intent.parser import parse_intent_json`.
"""
from __future__ import annotations

from .classifier import parse_intent_json

__all__ = ["parse_intent_json"]
