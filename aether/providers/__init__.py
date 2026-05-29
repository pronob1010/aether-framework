"""All plugin implementations for Aether's extension points.

Each subdirectory corresponds to a `kind` in the generic registry:
  - `aether.providers.llm` — LLM provider implementations
  - `aether.providers.vector` — (future) vector store implementations
  - `aether.providers.database` — (future) database implementations

Plugin authors add to this namespace via decorators like `@register_provider`.
"""
