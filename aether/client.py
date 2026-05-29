import os
from aether.llm.contracts import LLMProvider, LLMRequest
from aether.llm.providers.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    build_provider,
)

_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "fake":   None,
}
_MODEL_ENV = {
    "openai": "OPENAI_MODEL",
    "gemini": "GEMINI_MODEL",
    "fake":   None,
}

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
        if name not in _API_KEY_ENV:
            raise ValueError(
                f"Unknown LLM_PROVIDER={name!r}. "
                f"Known: {list(_API_KEY_ENV)}."
            )

        api_key = None
        if env := _API_KEY_ENV[name]:
            api_key = os.getenv(env)
            if not api_key:
                raise RuntimeError(f"Set {env} to use the {name!r} provider.")

        default_model = None
        if env := _MODEL_ENV[name]:
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
