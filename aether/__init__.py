"""
Aether — a Python framework for building AI-native applications.

Blends classical software engineering patterns with the demands of
LLM-based execution: provider abstraction, prompt pipelines, tool
registries, memory, context budgeting, reasoning strategies, and
observability.
"""

from aether.client import Aether
from aether.llm import LLMProvider, LLMRequest, LLMResponse, ask
from aether.providers.llm.registry import register_provider
from aether.registry import register, register_lazy

__all__ = [
    # Entry point
    "Aether",
    # Generic extension API (any subsystem: LLM, vector store, DB, ...)
    "register",
    "register_lazy",
    # LLM-specific convenience
    "register_provider",
    # LLM contracts
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ask",
]

__version__ = "0.0.1"
