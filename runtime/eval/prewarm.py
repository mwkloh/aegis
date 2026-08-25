"""Make a model resident before the benchmark starts measuring it.

Ollama loads weights lazily on first use. Measured 2026-08-24 on
`qwen3-vl:4b`: `load_duration` 23.48 s against a 30 s read timeout, with
`/api/ps` reporting nothing resident. Whichever variant happens to run first
absorbs that, and running nine configs back-to-back forces evictions, so the
tax reappears unpredictably.

Warming up front does not make the cost disappear -- it moves it outside the
measured window and reports it once, as the hardware fact it is, rather than
smearing it across whichever task drew the short straw.
"""
from __future__ import annotations

import logging

from runtime.llm.clients.base import ChatMessage, ChatRequest, ModelClient
from runtime.llm.telemetry import CallTelemetry, collect_calls

logger = logging.getLogger(__name__)

_WARMUP_PROMPT = "ok"


async def prewarm(client: ModelClient, model: str) -> CallTelemetry | None:
    """Issue one throwaway generation. Returns its telemetry, or None if it failed.

    The call is wrapped in its own collector, so its cost is captured here and
    never reaches an enclosing per-variant measurement.

    Failure is not fatal: a model that cannot be warmed will fail loudly enough
    in the run itself, and aborting the whole benchmark over a warm-up would
    lose the other configs.
    """
    with collect_calls() as calls:
        try:
            await client.chat(
                ChatRequest(
                    model=model,
                    messages=[ChatMessage(role="user", content=_WARMUP_PROMPT)],
                    max_tokens=1,
                )
            )
        except Exception:
            logger.warning("eval.prewarm.failed", extra={"model": model})
            return None
    return calls[-1] if calls else None


__all__ = ["prewarm"]
