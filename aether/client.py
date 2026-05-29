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
from aether.tools.registry import dispatch_tool
from aether.config import get_default_temperature, get_max_tool_iterations


def _config_from_env(
    *,
    with_retry: bool,
    with_circuit_breaker: bool,
    with_cost_tracking: bool,
) -> ProviderConfig:
    """Resolve LLM_PROVIDER + per-provider env vars into a ProviderConfig.

    Used by `Aether()` when no explicit provider or config is given.
    """
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

    return ProviderConfig(
        name=name,
        api_key=api_key,
        default_model=default_model,
        retry=RetryConfig() if with_retry else None,
        circuit_breaker=CircuitBreakerConfig() if with_circuit_breaker else None,
        cost_tracking=CostTrackingConfig() if with_cost_tracking else None,
    )


class Aether:
    """Top-level entry point. Hides provider construction, resilience wiring,
    and request/response plumbing.

    Three construction modes through one constructor:

        Aether()
            Auto-detect from env. Reads LLM_PROVIDER and the matching
            *_API_KEY / *_MODEL vars. Wraps the provider in retry +
            circuit-breaker + cost-tracking by default.

        Aether(config=ProviderConfig(...))
            Build the same decorator stack from an explicit config object.

        Aether(provider=some_provider)
            Use a pre-built provider as-is. You're in control of any
            decorator wrapping; the `with_*` flags are ignored.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        config: ProviderConfig | None = None,
        with_retry: bool = True,
        with_circuit_breaker: bool = True,
        with_cost_tracking: bool = True,
    ):
        if provider is not None and config is not None:
            raise ValueError("Pass either `provider` or `config`, not both.")

        if provider is not None:
            self._provider = provider
            return

        if config is None:
            config = _config_from_env(
                with_retry=with_retry,
                with_circuit_breaker=with_circuit_breaker,
                with_cost_tracking=with_cost_tracking,
            )
        self._provider = build_provider(config)

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
        temperature: float | None = None,
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> LLMResponse:
        """Full response — text, model, token counts.

        `prompt` accepts either a string (treated as a single user turn) or
        a list of `Message` objects for multi-turn conversations.

        If `tools` is provided, runs the tool-calling loop: the LLM may
        request tool invocations, which Aether dispatches and feeds back
        as new messages, up to `max_tool_iterations` round-trips before
        returning the most recent response.

        Unspecified `temperature` reads `AETHER_DEFAULT_TEMPERATURE`
        (falls back to 0.7). Unspecified `max_tool_iterations` reads
        `AETHER_MAX_TOOL_ITERATIONS` (falls back to 10).
        """
        if temperature is None:
            temperature = get_default_temperature()
        if max_tool_iterations is None:
            max_tool_iterations = get_max_tool_iterations()
        messages = self._to_messages(prompt)

        # Fast path: no tools → single round-trip.
        if not tools:
            return await self._provider.complete(LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
            ))

        # Tool loop: each iteration is one LLM call. If the LLM emits
        # tool_calls, dispatch them, append the results as messages, and
        # call again. Stop when the LLM produces a response with no more
        # tool calls, or when the iteration cap is hit.
        response: LLMResponse | None = None
        for _ in range(max_tool_iterations + 1):
            response = await self._provider.complete(LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                tools=tools,
            ))
            if not response.tool_calls:
                return response

            messages.append(Message(
                role="assistant",
                content=response.text or None,
                tool_calls=response.tool_calls,
            ))
            for tc in response.tool_calls:
                try:
                    result = await dispatch_tool(tc.name, tc.arguments)
                    content = str(result)
                except Exception as e:
                    content = f"Error executing {tc.name}: {e}"
                messages.append(Message(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                ))

        # Hit the iteration cap — return the last response (likely still
        # asking for tools, but the caller said "give up after N").
        assert response is not None
        return response

    async def ask(
        self,
        question: str | list[Message],
        *,
        tools: list[str] | None = None,
    ) -> str:
        """Text-only convenience over `complete()`. Returns just the answer.

        Supports tool calling — pass `tools=["name", ...]` and Aether runs
        the loop, returning the final assistant text after all tool calls
        are resolved.
        """
        response = await self.complete(question, tools=tools)
        return response.text

    async def stream(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream of rich chunks — delta text + metadata."""
        if temperature is None:
            temperature = get_default_temperature()
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
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Text-only convenience over `stream()`. Yields just text deltas."""
        async for chunk in self.stream(prompt, model=model, temperature=temperature):
            if chunk.text:
                yield chunk.text
