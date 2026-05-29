"""Observer-pattern hooks for runtime introspection.

Subscribers (sync or async callables) hook into lifecycle events emitted
by `Aether` — request start/complete/error, stream chunks, tool dispatch.
Use them for logging, tracing, metrics, debugging, audit trails.

Subscriber errors are caught and logged, never propagated. Observability
must not break the request path. If your logger throws, your agent keeps
running and you see the exception in stderr.

Quick start:

    client = Aether()
    client.on("request.complete", lambda e: print(f"{e.duration_seconds:.2f}s"))
    client.on("tool.error", lambda e: alert(e.error))
    await client.ask("hi")
"""
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union
from aether.llm.contracts import LLMRequest, LLMResponse, LLMStreamChunk, ToolCall


logger = logging.getLogger(__name__)


# --- Event name constants -------------------------------------------------
# Use these instead of bare strings — typo-proof, easy to grep.

REQUEST_START    = "request.start"
REQUEST_COMPLETE = "request.complete"
REQUEST_ERROR    = "request.error"

STREAM_START     = "stream.start"
STREAM_CHUNK     = "stream.chunk"
STREAM_COMPLETE  = "stream.complete"
STREAM_ERROR     = "stream.error"

TOOL_START       = "tool.start"
TOOL_COMPLETE    = "tool.complete"
TOOL_ERROR       = "tool.error"


# --- Event payloads -----------------------------------------------------
# Each event class is what handlers receive — typed so IDEs autocomplete.

@dataclass
class RequestStartEvent:
    request: LLMRequest


@dataclass
class RequestCompleteEvent:
    request: LLMRequest
    response: LLMResponse
    duration_seconds: float


@dataclass
class RequestErrorEvent:
    request: LLMRequest
    error: BaseException
    duration_seconds: float


@dataclass
class StreamStartEvent:
    request: LLMRequest


@dataclass
class StreamChunkEvent:
    request: LLMRequest
    chunk: LLMStreamChunk


@dataclass
class StreamCompleteEvent:
    request: LLMRequest
    duration_seconds: float
    chunk_count: int


@dataclass
class StreamErrorEvent:
    request: LLMRequest
    error: BaseException
    duration_seconds: float
    chunk_count: int


@dataclass
class ToolStartEvent:
    call: ToolCall


@dataclass
class ToolCompleteEvent:
    call: ToolCall
    result: Any
    duration_seconds: float


@dataclass
class ToolErrorEvent:
    call: ToolCall
    error: BaseException
    duration_seconds: float


# --- The bus -----------------------------------------------------------

Handler = Callable[[Any], Union[None, Awaitable[None]]]


class EventBus:
    """A small synchronous-or-async pub/sub for lifecycle events.

    Pass an instance to multiple `Aether` clients if you want them to share
    subscribers — useful for a single global metrics/logging stack.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)

    def on(self, event: str, handler: Handler | None = None):
        """Register a subscriber. Two forms:

            bus.on("request.complete", my_handler)        # direct call
            @bus.on("request.complete")                   # decorator
            def my_handler(event): ...
        """
        if handler is None:
            def decorator(h: Handler) -> Handler:
                self._subscribers[event].append(h)
                return h
            return decorator
        self._subscribers[event].append(handler)
        return handler

    def off(self, event: str, handler: Handler) -> None:
        """Unregister a subscriber. No-op if it wasn't registered."""
        try:
            self._subscribers[event].remove(handler)
        except (KeyError, ValueError):
            pass

    async def emit(self, event: str, payload: Any) -> None:
        """Fire `payload` to every subscriber. Sync handlers run inline;
        async handlers are awaited in registration order. Exceptions in
        any one subscriber are logged and swallowed — observability is
        never allowed to break the request path.
        """
        # Snapshot the list so a handler that subscribes/unsubscribes during
        # dispatch doesn't mutate what we're iterating over.
        for handler in list(self._subscribers.get(event, [])):
            try:
                result = handler(payload)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    f"Subscriber for event {event!r} raised; ignoring."
                )
