"""CLI walking skeleton.

Reads lines from stdin → classifies intent → resolves skill → builds tool intent
→ harness executes → reply printed. Every step writes a JSONL event so the
Reflection plane has data to chew on later.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from typing import Protocol

from runtime.config import AegisConfig, get_config
from runtime.events import EventStream, EventType
from runtime.harness import HarnessAdapter
from runtime.intent import (
    IntentClassification,
    IntentClassifier,
    ModelBackedClassifier,
)
from runtime.llm.clients import InstrumentedModelClient, OllamaClient
from runtime.llm.clients.openrouter_client import (
    OpenRouterClient,
    OpenRouterConfigError,
)
from runtime.reasoning import SkillRunner
from runtime.reasoning.tier1_reasoner import Tier1Reasoner
from runtime.skills import SkillRegistry


class _AsyncClassifier(Protocol):
    async def classify(self, text: str) -> IntentClassification: ...


class Pipeline:
    """End-to-end glue. Pure dependency injection for testability."""

    def __init__(
        self,
        registry: SkillRegistry,
        classifier: _AsyncClassifier,
        runner: SkillRunner,
        harness: HarnessAdapter,
        events: EventStream,
    ) -> None:
        self.registry = registry
        self.classifier = classifier
        self.runner = runner
        self.harness = harness
        self.events = events

    async def handle(self, user_text: str) -> str:
        self.events.append(EventType.USER_MESSAGE, {"text": user_text})

        classification = await self.classifier.classify(user_text)
        self.events.append(
            EventType.INTENT_CLASSIFIED,
            {"intent": classification.intent, "confidence": classification.confidence},
        )

        skill = self.registry.for_intent(classification.intent)
        if skill is None:
            self.events.append(
                EventType.PATTERN_OBSERVED,
                {
                    "pattern": "intent_unknown",
                    "intent": classification.intent,
                    "confidence": classification.confidence,
                },
            )
            reply = (
                f"I don't know how to handle intent {classification.intent!r} yet. "
                "Try `echo hello`, `ping`, or `what time is it in Tokyo?`."
            )
            self.events.append(EventType.ASSISTANT_REPLY, {"text": reply})
            return reply

        self.events.append(EventType.SKILL_SELECTED, {"skill_id": skill.id})

        intent = await self.runner.build(skill, user_text)
        self.events.append(
            EventType.CONTRACT_EMITTED,
            {"tool": intent.tool, "skill_id": intent.skill_id, "args": intent.args},
        )

        self.events.append(EventType.TOOL_INVOKED, {"tool": intent.tool})
        result = self.harness.execute(intent)
        self.events.append(
            EventType.TOOL_RESULT,
            {"status": result.status, "payload": result.payload, "error": result.error},
        )

        if result.status != "ok":
            reply = f"[{intent.tool}] error: {result.error}"
        else:
            reply = self._render(intent.tool, result.payload)

        self.events.append(EventType.ASSISTANT_REPLY, {"text": reply})
        return reply

    @staticmethod
    def _render(tool: str, payload: dict[str, object]) -> str:
        if tool == "echo":
            return f"echo → {payload.get('echoed', '')}"
        if tool == "respond":
            return str(payload.get("message", ""))
        if tool == "time":
            return str(payload.get("formatted", payload))
        return f"[{tool}] {payload}"


class _SyncClassifierAdapter:
    """Wraps the sync rule classifier so Pipeline can await it uniformly."""

    def __init__(self, inner: IntentClassifier) -> None:
        self._inner = inner

    async def classify(self, text: str) -> IntentClassification:
        return self._inner.classify(text)


def build_pipeline(config: AegisConfig | None = None) -> Pipeline:
    cfg = config or get_config()
    cfg.storage.sessions_dir.mkdir(parents=True, exist_ok=True)
    events = EventStream(cfg.storage.sessions_dir)
    registry = SkillRegistry.from_directory(cfg.skills.catalog_dir)
    known_intents = [i for d in registry.all() for i in d.intents]

    classifier: _AsyncClassifier
    try:
        ollama_raw = OllamaClient(cfg)
        ollama = InstrumentedModelClient(
            inner=ollama_raw, events=events, tier="fast", provider="ollama"
        )
        classifier = ModelBackedClassifier(
            client=ollama, model=cfg.models.fast, known_intents=known_intents
        )
    except Exception:
        classifier = _SyncClassifierAdapter(IntentClassifier(known_intents=known_intents))

    tier1: Tier1Reasoner | None = None
    try:
        openrouter_raw = OpenRouterClient(cfg)
        openrouter = InstrumentedModelClient(
            inner=openrouter_raw, events=events, tier="smart", provider="openrouter"
        )
        tier1 = Tier1Reasoner(
            client=openrouter,
            model=cfg.models.smart,
            think=cfg.think_for(cfg.models.smart),
        )
    except OpenRouterConfigError:
        tier1 = None
        events.append(
            EventType.PATTERN_OBSERVED,
            {"pattern": "tier1_missing", "reason": "OPENROUTER_API_KEY not configured"},
        )

    runner = SkillRunner(tier1=tier1)

    from runtime.files.client import FilesClient  # noqa: PLC0415
    from runtime.harness import DEFAULT_TOOLS  # noqa: PLC0415
    from runtime.harness.tools.files_tool import make_files_tools  # noqa: PLC0415

    try:
        _files_client = FilesClient(cfg.files.allowed_roots)
        _file_tools = make_files_tools(_files_client)
    except ValueError:
        _file_tools = {}
    # file tools namespaced with "files_" so no collision risk with builtins
    harness = HarnessAdapter(tools={**DEFAULT_TOOLS, **_file_tools})
    return Pipeline(registry, classifier, runner, harness, events)


def main(argv: list[str] | None = None) -> int:
    """Read-eval-print loop. Ctrl-D / 'quit' / 'exit' to stop."""
    argv = argv or sys.argv[1:]
    pipeline = build_pipeline()
    pipeline.events.append(EventType.SESSION_START, {"surface": "cli"})

    print("AEGIS CLI — Phase 1. Type 'echo <text>', 'ping', or ask a question. Ctrl-D to exit.")
    try:
        for line in _read_lines():
            text = line.strip()
            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                break
            try:
                reply = asyncio.run(pipeline.handle(text))
            except Exception as exc:
                pipeline.events.append(
                    EventType.ERROR, {"where": "pipeline.handle", "error": str(exc)}
                )
                reply = f"error: {exc}"
            print(reply)
    finally:
        pipeline.events.append(EventType.SESSION_END, {})
    return 0


def _read_lines() -> Iterator[str]:
    """Indirection so tests can monkeypatch stdin without process trickery."""
    while True:
        try:
            yield input("> ")
        except EOFError:
            return


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
