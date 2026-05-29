import os
from typing import AsyncIterator
from aether.llm.contracts import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
)
from aether.extensions.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    CostTrackingConfig,
    build_provider,
)
from aether.extensions.llm.cost_tracking import CostTrackingProvider, UsageStats
from aether.registry import REGISTRY, list_kind
from aether.extensions.llm.registry import LLM_PROVIDER_KIND


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
        with_cost_tracking: bool = True,
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
            cost_tracking=CostTrackingConfig() if with_cost_tracking else None,
        ))

    @property
    def usage(self) -> UsageStats:
        """Cumulative token usage and cost across all calls made via this client.

        Returns an empty `UsageStats` if cost tracking is not enabled in the
        decorator stack — so the property is always callable, you just see zeros.
        """
        # Walk the decorator chain looking for a CostTrackingProvider.
        provider = self._provider
        while True:
            if isinstance(provider, CostTrackingProvider):
                return provider.stats
            inner = getattr(provider, "inner_provider", None)
            if inner is None:
                return UsageStats()
            provider = inner

    @staticmethod
    def _to_messages(prompt: str | list[Message]) -> list[Message]:
        """Accept either a string (single user turn) or a full message list."""
        if isinstance(prompt, str):
            return [Message(role="user", content=prompt)]
        return prompt

    async def complete(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Full response — text, model, token counts.

        `prompt` accepts either a string (treated as a single user turn) or
        a list of `Message` objects for multi-turn conversations.
        """
        return await self._provider.complete(LLMRequest(
            messages=self._to_messages(prompt),
            model=model,
            temperature=temperature,
        ))

    async def ask(self, question: str | list[Message]) -> str:
        """Text-only convenience over `complete()`. Returns just the answer."""
        response = await self.complete(question)
        return response.text

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream of rich chunks — delta text + metadata."""
        request = LLMRequest(
            messages=self._to_messages(prompt),
            model=model,
            temperature=temperature,
        )
        async for chunk in self._provider.stream(request):
            yield chunk

    async def stream_text(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Text-only convenience over `stream()`. Yields just text deltas."""
        async for chunk in self.stream(prompt, model=model, temperature=temperature):
            if chunk.text:
                yield chunk.text
