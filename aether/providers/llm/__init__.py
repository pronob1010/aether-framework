"""LLM provider implementations and the LLM-specific registration helper.

Public surface:
  - `register_provider` — decorator for third-party providers
  - `make_provider`     — name → instance lookup
  - `build_provider`    — config → composed (provider + decorators) stack
  - `ProviderConfig`, `RetryConfig`, `CircuitBreakerConfig` — config models

Concrete adapters (`OpenAIProvider`, `GeminiProvider`, `FakeProvider`) and
decorators (`RetryingProvider`, `CircuitBreakerProvider`) live as submodules
and are imported lazily to keep optional SDK dependencies out of the cold path.
"""

from aether.providers.llm.registry import register_provider, LLM_PROVIDER_KIND
from aether.providers.llm.factory import make_provider
from aether.providers.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    build_provider,
)

__all__ = [
    "register_provider",
    "LLM_PROVIDER_KIND",
    "make_provider",
    "build_provider",
    "ProviderConfig",
    "RetryConfig",
    "CircuitBreakerConfig",
]
