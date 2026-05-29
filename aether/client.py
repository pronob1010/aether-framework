import os
import time
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
from aether.events import (
    EventBus,
    Handler,
    REQUEST_START, REQUEST_COMPLETE, REQUEST_ERROR,
    STREAM_START, STREAM_CHUNK, STREAM_COMPLETE, STREAM_ERROR,
    TOOL_START, TOOL_COMPLETE, TOOL_ERROR,
    RequestStartEvent, RequestCompleteEvent, RequestErrorEvent,
    StreamStartEvent, StreamChunkEvent, StreamCompleteEvent, StreamErrorEvent,
    ToolStartEvent, ToolCompleteEvent, ToolErrorEvent,
)


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
        events: EventBus | None = None,
    ):
        if provider is not None and config is not None:
            raise ValueError("Pass either `provider` or `config`, not both.")

        # Share an EventBus across clients by passing the same instance.
        # Default: each client gets its own.
        self.events = events or EventBus()

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

    # --- Observability shortcuts -----------------------------------------

    def on(self, event: str, handler: Handler | None = None):
        """Subscribe a handler (sync or async) to a lifecycle event.

        Two forms:

            client.on("request.complete", my_handler)        # direct call
            @client.on("request.complete")                   # decorator
            def my_handler(event): ...
        """
        return self.events.on(event, handler)

    def off(self, event: str, handler: Handler) -> None:
        """Unsubscribe a handler from an event. No-op if it wasn't registered."""
        self.events.off(event, handler)

    # --- Internal emit helpers ------------------------------------------

    async def _complete_with_events(self, request: LLMRequest) -> LLMResponse:
        """Wrap one provider.complete() call in request.start/complete/error events."""
        await self.events.emit(REQUEST_START, RequestStartEvent(request=request))
        start = time.perf_counter()
        try:
            response = await self._provider.complete(request)
        except BaseException as e:
            duration = time.perf_counter() - start
            await self.events.emit(REQUEST_ERROR, RequestErrorEvent(
                request=request, error=e, duration_seconds=duration,
            ))
            raise
        duration = time.perf_counter() - start
        await self.events.emit(REQUEST_COMPLETE, RequestCompleteEvent(
            request=request, response=response, duration_seconds=duration,
        ))
        return response

    async def _dispatch_with_events(self, tc) -> str:
        """Wrap one tool dispatch in tool.start/complete/error events.

        Returns a *content* string suitable for a tool message. Errors are
        converted to content (so the LLM can recover) — they don't propagate.
        """
        await self.events.emit(TOOL_START, ToolStartEvent(call=tc))
        start = time.perf_counter()
        try:
            result = await dispatch_tool(tc.name, tc.arguments)
        except Exception as e:
            duration = time.perf_counter() - start
            await self.events.emit(TOOL_ERROR, ToolErrorEvent(
                call=tc, error=e, duration_seconds=duration,
            ))
            return f"Error executing {tc.name}: {e}"
        duration = time.perf_counter() - start
        await self.events.emit(TOOL_COMPLETE, ToolCompleteEvent(
            call=tc, result=result, duration_seconds=duration,
        ))
        return str(result)

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
            return await self._complete_with_events(LLMRequest(
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
            response = await self._complete_with_events(LLMRequest(
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
                content = await self._dispatch_with_events(tc)
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
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream of rich chunks — delta text + metadata.

        Fires `stream.start` ONCE at the beginning, `stream.chunk` per
        text chunk yielded to the user, and `stream.complete` when the
        whole user-facing stream ends. If `tools` is provided, the LLM
        may call tools mid-stream; those happen invisibly to the user
        (no chunks emitted for tool-call control flow). Tool dispatch
        fires the normal `tool.*` events.

        Errors propagate; `stream.error` fires with `chunk_count` set
        to however many user-facing text chunks made it out first.
        """
        if temperature is None:
            temperature = get_default_temperature()
        if max_tool_iterations is None:
            max_tool_iterations = get_max_tool_iterations()
        messages = self._to_messages(prompt)

        # `initial_request` is what the user-facing stream events reference.
        # Internal tool-loop iterations build their own LLMRequest objects.
        initial_request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
        )

        await self.events.emit(STREAM_START, StreamStartEvent(request=initial_request))
        start = time.perf_counter()
        chunk_count = 0

        try:
            if not tools:
                # Fast path — single provider session, no tool loop.
                async for chunk in self._provider.stream(initial_request):
                    chunk_count += 1
                    await self.events.emit(STREAM_CHUNK, StreamChunkEvent(
                        request=initial_request, chunk=chunk,
                    ))
                    yield chunk
            else:
                # Tool-aware loop: each iteration is one provider.stream()
                # session. Text chunks pass through to the user; chunks
                # carrying tool_calls are consumed internally (dispatched,
                # results appended to messages, next iteration starts).
                for _ in range(max_tool_iterations + 1):
                    request = LLMRequest(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        tools=tools,
                    )

                    session_tool_calls: list = []
                    session_text = ""

                    async for chunk in self._provider.stream(request):
                        if chunk.tool_calls:
                            # Control-flow chunk — accumulate, don't show user.
                            session_tool_calls.extend(chunk.tool_calls)
                            continue
                        chunk_count += 1
                        session_text += chunk.text
                        await self.events.emit(STREAM_CHUNK, StreamChunkEvent(
                            request=initial_request, chunk=chunk,
                        ))
                        yield chunk

                    if not session_tool_calls:
                        # LLM produced its final answer; user-facing stream is done.
                        break

                    # Append the assistant turn (text it streamed + tool_calls
                    # it requested), then dispatch each tool and append results.
                    messages.append(Message(
                        role="assistant",
                        content=session_text or None,
                        tool_calls=session_tool_calls,
                    ))
                    for tc in session_tool_calls:
                        content = await self._dispatch_with_events(tc)
                        messages.append(Message(
                            role="tool",
                            content=content,
                            tool_call_id=tc.id,
                        ))
        except BaseException as e:
            duration = time.perf_counter() - start
            await self.events.emit(STREAM_ERROR, StreamErrorEvent(
                request=initial_request, error=e,
                duration_seconds=duration, chunk_count=chunk_count,
            ))
            raise

        duration = time.perf_counter() - start
        await self.events.emit(STREAM_COMPLETE, StreamCompleteEvent(
            request=initial_request,
            duration_seconds=duration,
            chunk_count=chunk_count,
        ))

    async def stream_text(
        self,
        prompt: str | list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[str] | None = None,
        max_tool_iterations: int | None = None,
    ) -> AsyncIterator[str]:
        """Text-only convenience over `stream()`. Yields just text deltas.

        Supports tool calling: pass `tools=["name", ...]` and Aether runs
        the tool loop invisibly, yielding only the LLM's final-answer text.
        """
        async for chunk in self.stream(
            prompt,
            model=model,
            temperature=temperature,
            tools=tools,
            max_tool_iterations=max_tool_iterations,
        ):
            if chunk.text:
                yield chunk.text
