"""Shared contract for model clients.

Every provider implementation (Ollama, OpenRouter, …) exposes the same
async `chat()` surface so the runtime never branches on provider.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str = Field(min_length=0, max_length=64_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    response_format: Literal["text", "json"] = "text"
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON schema for decoder-enforced structured output. When set, "
            "takes precedence over response_format on clients that support "
            "it (Ollama). Clients that do not support it fall back to "
            "response_format."
        ),
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class ModelClient(Protocol):
    """Async chat surface — every client implements this."""

    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    async def health(self) -> bool: ...
