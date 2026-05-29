import os
from aether.llm.contracts import LLMProvider, LLMRequest
from aether.providers.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    build_provider,
)
from aether.registry import REGISTRY, list_kind
from aether.providers.llm.registry import LLM_PROVIDER_KIND


class Aether:
    """Top-level entry point. Hides provider construction, resilience wiring,
    and request/response plumbing."""

    def __init__(self, provider: LLMProvider):
        self._provider = provider

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "Aether":
        return cls(build_provider(config))

    @classmethod
    def from_env(
        cls,
        *,
        with_retry: bool = True,
        with_circuit_breaker: bool = True,
    ) -> "Aether":
        name = os.getenv("LLM_PROVIDER", "openai")
        specs = REGISTRY[LLM_PROVIDER_KIND]
        if name not in specs:
            raise ValueError(
                f"Unknown LLM_PROVIDER={name!r}. "
                f"Known: {list_kind(LLM_PROVIDER_KIND)}."
            )
        meta = specs[name].metadata

        api_key = None
        if env := meta.get("api_key_env"):
            api_key = os.getenv(env)
            if not api_key:
                raise RuntimeError(
                    f"Set {env} to use the {name!r} provider."
                )

        default_model = None
        if env := meta.get("model_env"):
            default_model = os.getenv(env)

        return cls.from_config(ProviderConfig(
            name=name,
            api_key=api_key,
            default_model=default_model,
            retry=RetryConfig() if with_retry else None,
            circuit_breaker=CircuitBreakerConfig() if with_circuit_breaker else None,
        ))

    async def ask(self, question: str) -> str:
        response = await self._provider.complete(LLMRequest(prompt=question))
        return response.text
