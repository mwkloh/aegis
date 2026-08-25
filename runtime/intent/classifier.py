"""Tier 0 intent classifier.

Two implementations, same surface:
  - `IntentClassifier`: deterministic prefix rules. Sync, zero I/O.
  - `ModelBackedClassifier`: rules first; falls back to a local Ollama model
    when no rule fires. Async (because Ollama is async), but degrades to the
    rule-only result if anything goes wrong.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from runtime.events.stream import EventStream
from runtime.llm.clients import ChatMessage
from runtime.llm.clients.base import ModelClient
from runtime.llm.structured_output import request_structured


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)


class ClassifierUnavailableError(Exception):
    """The model-backed classifier could not produce a classification.

    Raised for transport failures, schema/structured-output failures, and
    malformed model replies — never for a genuine model-produced
    ``intent="unknown"`` answer, which is a real classification, not a
    failure. Callers must treat this distinctly from `IntentClassification`:
    it means "we don't know whether this could be classified," not "this
    was classified as unclassifiable."
    """


# Phase 0 deterministic rules — exact triggers we want the walking skeleton
# to recognise without a model in the loop.
_PREFIX_RULES: tuple[tuple[re.Pattern[str], str, float], ...] = (
    (re.compile(r"^\s*echo(\s+|$)", re.IGNORECASE), "echo", 0.99),
    (re.compile(r"^\s*ping\s*$", re.IGNORECASE), "ping", 0.99),
)


class IntentClassifier:
    """Rule-only classifier. Sync surface used by the Phase 0 pipeline."""

    def __init__(self, known_intents: Iterable[str] | None = None) -> None:
        self._known: set[str] = set(known_intents or [])

    def classify(self, text: str) -> IntentClassification:
        if not isinstance(text, str):
            raise TypeError("classify(text): text must be str")
        for pattern, intent, conf in _PREFIX_RULES:
            if pattern.search(text):
                return IntentClassification(intent=intent, confidence=conf)
        return IntentClassification(intent="unknown", confidence=0.0)


# Hard ceiling on text we'll send to a local model — keeps prompt cost bounded
# and stops any pathological flood from triggering retries.
_MAX_INPUT_CHARS: int = 4096

_PROMPT_PATH: Path = Path(__file__).parent / "prompts" / "intent_classifier.txt"


class ModelBackedClassifier:
    """Rule-first → Ollama-fallback. Same `classify()` shape, but async.

    Model calls go through `request_structured` so one malformed JSON reply
    gets a corrective retry before we give up. A genuine model answer of
    `intent="unknown"` returns normally; a transport, schema, or validation
    failure raises `ClassifierUnavailableError` instead of degrading to a
    value — callers that need "touch nothing" on either outcome should
    catch it explicitly rather than relying on it happening to also be
    `intent="unknown"`.
    """

    def __init__(
        self,
        client: object,
        model: str,
        known_intents: Iterable[str],
        prompt_path: Path | None = None,
        events: EventStream | None = None,
        max_retries: int = 1,
    ) -> None:
        self._client = client  # ModelClient Protocol — can't import for cycles
        self._model = model
        self._known: tuple[str, ...] = tuple(sorted(set(known_intents)))
        self._prompt_template = (prompt_path or _PROMPT_PATH).read_text(encoding="utf-8")
        self._rule_only = IntentClassifier(self._known)
        self._events = events
        self._max_retries = max_retries
        self._schema: dict[str, Any] = _build_schema(self._known)

    async def classify(self, text: str) -> IntentClassification:
        if not isinstance(text, str):
            raise TypeError("classify(text): text must be str")

        rule_result = self._rule_only.classify(text)
        if rule_result.intent != "unknown":
            return rule_result

        bounded = text[:_MAX_INPUT_CHARS]
        try:
            return await self._ask_model(bounded)
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            # Transport / client failure — distinguishable from a genuine
            # model-produced "unknown" so callers can fail safe instead of
            # silently fanning out to a full-catalog planner. See
            # ClassifierUnavailableError.
            raise ClassifierUnavailableError(str(exc)) from exc

    async def _ask_model(self, user_text: str) -> IntentClassification:
        system = self._prompt_template.format(
            known_intents="\n".join(f"- {i}" for i in self._known)
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_text),
        ]
        data, outcome = await request_structured(
            cast(ModelClient, self._client),
            messages,
            self._schema,
            model=self._model,
            temperature=0.0,
            max_tokens=64,
            max_retries=self._max_retries,
            think=False,
            events=self._events,
            call_site="intent.classifier",
        )
        if outcome.error_kind != "ok":
            raise ClassifierUnavailableError(
                f"request_structured failed: {outcome.error_kind}"
            )
        try:
            return IntentClassification(
                intent=str(data["intent"]),
                confidence=float(data["confidence"]),
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise ClassifierUnavailableError(
                f"malformed structured reply: {exc}"
            ) from exc


def _build_schema(known_intents: tuple[str, ...]) -> dict[str, Any]:
    """JSON schema that pins intent to known_intents plus "unknown".

    `additionalProperties` is left unset so models are free to add a
    `rationale` field without failing schema validation — we only need
    `intent` and `confidence` to be well-typed.
    """
    allowed = sorted({*known_intents, "unknown"})
    return {
        "type": "object",
        "required": ["intent", "confidence"],
        "properties": {
            "intent": {"type": "string", "enum": allowed},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }


def parse_intent_json(raw: str, known_intents: Iterable[str]) -> IntentClassification:
    """Strict JSON → IntentClassification.

    Returns `intent="unknown", confidence=0.0` for ANY of:
      - non-JSON text
      - JSON that isn't an object
      - missing intent / confidence
      - intent not in `known_intents` (and not "unknown")
      - confidence outside [0, 1] or non-numeric
    """
    allowed = set(known_intents) | {"unknown"}

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return IntentClassification(intent="unknown", confidence=0.0)

    if not isinstance(payload, dict):
        return IntentClassification(intent="unknown", confidence=0.0)

    intent = payload.get("intent")
    confidence = payload.get("confidence")

    if not isinstance(intent, str) or intent not in allowed:
        return IntentClassification(intent="unknown", confidence=0.0)

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return IntentClassification(intent="unknown", confidence=0.0)

    try:
        return IntentClassification(intent=intent, confidence=float(confidence))
    except ValidationError:
        return IntentClassification(intent="unknown", confidence=0.0)
