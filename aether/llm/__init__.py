"""LLM provider abstraction: Strategy + Adapter over multiple LLM vendors."""

from aether.llm.contracts import LLMProvider, LLMRequest, LLMResponse
from aether.llm.ask import ask

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ask",
]