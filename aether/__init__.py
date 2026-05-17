"""
Aether — a Python framework for building AI-native applications.

Blends classical software engineering patterns with the demands of
LLM-based execution: provider abstraction, prompt pipelines, tool
registries, memory, context budgeting, reasoning strategies, and
observability.
"""

from aether.llm import LLMProvider, LLMRequest, LLMResponse, ask

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ask",
]

__version__ = "0.0.1"