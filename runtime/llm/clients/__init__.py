"""Model clients (Ollama / OpenRouter) sitting behind a single Protocol."""

from .base import ChatMessage, ChatRequest, ChatResponse, ModelClient
from .instrumented import InstrumentedModelClient
from .ollama_client import OllamaClient
from .openrouter_client import OpenRouterClient

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "InstrumentedModelClient",
    "ModelClient",
    "OllamaClient",
    "OpenRouterClient",
]
