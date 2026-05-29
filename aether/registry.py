"""Top-level plugin registry. Knows nothing about LLMs, DBs, or vectors —
each subsystem registers under its own `kind`.

Layout:
    REGISTRY[kind][name] -> PluginSpec

`kind` is a free-form string namespacing extension types ("llm_provider",
"vector_store", "database", ...). `metadata` on the spec is whatever the
owning subsystem needs to interpret — the registry never reads it.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class PluginSpec:
    """Everything the framework stores about one registered plugin."""
    _factory: Callable[[], type]            # returns the CLASS (lazy or eager)
    metadata: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, dict[str, PluginSpec]] = defaultdict(dict)


def register(kind: str, name: str, **metadata: Any):
    """Class decorator — the generic extension point.

    `**metadata` is forwarded into the spec untouched. The subsystem that
    owns `kind` decides what keys are meaningful.

    Example (LLM):
        @register("llm_provider", "ollama", api_key_env="OLLAMA_KEY")
        class OllamaProvider: ...

    Example (hypothetical vector store):
        @register("vector_store", "pinecone", dimension=1536)
        class PineconeStore: ...
    """
    def decorator(cls):
        if not isinstance(cls, type):
            raise TypeError(
                f"register() expects a class, got {type(cls).__name__}"
            )
        REGISTRY[kind][name] = PluginSpec(
            _factory=lambda: cls,
            metadata=dict(metadata),
        )
        return cls
    return decorator


def register_lazy(
    kind: str,
    name: str,
    factory: Callable[[], type],
    **metadata: Any,
) -> None:
    """Imperative registration for lazy-imported built-ins.

    `factory` is called by `make_*` helpers when they actually need the class —
    so heavy SDK imports (openai, google.genai, psycopg, etc.) stay deferred
    until first use.
    """
    REGISTRY[kind][name] = PluginSpec(
        _factory=factory,
        metadata=dict(metadata),
    )


def get(kind: str, name: str) -> PluginSpec:
    """Lookup with a helpful error if the kind or name is unknown."""
    if kind not in REGISTRY:
        raise KeyError(
            f"unknown kind: {kind!r}. Known kinds: {sorted(REGISTRY)}."
        )
    if name not in REGISTRY[kind]:
        raise KeyError(
            f"unknown {kind} {name!r}. "
            f"Known {kind}s: {sorted(REGISTRY[kind])}."
        )
    return REGISTRY[kind][name]


def list_kind(kind: str) -> list[str]:
    """Names registered under `kind`, sorted. Empty list if kind is unknown."""
    return sorted(REGISTRY.get(kind, {}))
