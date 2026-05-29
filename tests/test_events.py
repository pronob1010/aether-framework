"""Observability hooks — Observer pattern via Aether's EventBus.

Verifies all 10 lifecycle events fire at the right moments, that sync
and async handlers both work, that subscriber errors are isolated, and
that buses can be shared across clients.
"""
import asyncio
import pytest
from aether import Aether, EventBus, register_tool
from aether.llm.contracts import LLMResponse, ToolCall
from aether.extensions.llm.fake import FakeProvider
from aether.events import (
    REQUEST_START, REQUEST_COMPLETE, REQUEST_ERROR,
    STREAM_START, STREAM_CHUNK, STREAM_COMPLETE, STREAM_ERROR,
    TOOL_START, TOOL_COMPLETE, TOOL_ERROR,
    RequestStartEvent, RequestCompleteEvent, RequestErrorEvent,
    StreamStartEvent, StreamChunkEvent, StreamCompleteEvent, StreamErrorEvent,
    ToolStartEvent, ToolCompleteEvent, ToolErrorEvent,
)
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# Helpers for scripted multi-turn provider responses (tool tests)
def _assistant_with_call(name: str, arguments: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        text="", model="fake-model", input_tokens=1, output_tokens=1,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )

def _assistant_final(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="fake-model", input_tokens=1, output_tokens=1)


# --- Request lifecycle ---------------------------------------------------

@pytest.mark.asyncio
async def test_request_start_and_complete_fire_in_order():
    seen = []
    client = Aether(FakeProvider(canned_response="ok"))
    client.on(REQUEST_START,    lambda e: seen.append(("start", type(e).__name__)))
    client.on(REQUEST_COMPLETE, lambda e: seen.append(("complete", type(e).__name__)))
    await client.ask("hi")
    assert seen == [
        ("start", "RequestStartEvent"),
        ("complete", "RequestCompleteEvent"),
    ]


@pytest.mark.asyncio
async def test_request_complete_event_has_response_and_duration():
    received: list[RequestCompleteEvent] = []
    client = Aether(FakeProvider(canned_response="42"))
    client.on(REQUEST_COMPLETE, lambda e: received.append(e))
    await client.ask("hi")
    assert received[0].response.text == "42"
    assert received[0].duration_seconds >= 0


@pytest.mark.asyncio
async def test_request_error_event_fires_and_exception_propagates():
    class Boom:
        async def complete(self, request):
            raise ValueError("kaboom")
        async def stream(self, request):
            yield  # unreachable

    received: list[RequestErrorEvent] = []
    client = Aether(Boom())
    client.on(REQUEST_ERROR, lambda e: received.append(e))
    with pytest.raises(ValueError, match="kaboom"):
        await client.ask("hi")
    assert len(received) == 1
    assert "kaboom" in str(received[0].error)


# --- Handler shapes ------------------------------------------------------

@pytest.mark.asyncio
async def test_async_handler_is_awaited():
    seen = []
    async def slow_handler(e):
        await asyncio.sleep(0)
        seen.append(e)
    client = Aether(FakeProvider())
    client.on(REQUEST_COMPLETE, slow_handler)
    await client.ask("hi")
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_multiple_handlers_all_called_in_registration_order():
    seen = []
    client = Aether(FakeProvider())
    client.on(REQUEST_COMPLETE, lambda e: seen.append("a"))
    client.on(REQUEST_COMPLETE, lambda e: seen.append("b"))
    client.on(REQUEST_COMPLETE, lambda e: seen.append("c"))
    await client.ask("hi")
    assert seen == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_subscriber_error_does_not_break_request():
    """Critical contract: observability must never break the request path."""
    def bad_handler(e):
        raise RuntimeError("logger died")
    client = Aether(FakeProvider(canned_response="still ok"))
    client.on(REQUEST_COMPLETE, bad_handler)
    answer = await client.ask("hi")
    assert answer == "still ok"


@pytest.mark.asyncio
async def test_subscriber_error_isolation_per_handler():
    """One handler's exception doesn't prevent later handlers from running."""
    seen = []
    client = Aether(FakeProvider())
    client.on(REQUEST_COMPLETE, lambda e: (_ for _ in ()).throw(RuntimeError("die")))
    client.on(REQUEST_COMPLETE, lambda e: seen.append("survived"))
    await client.ask("hi")
    assert seen == ["survived"]


@pytest.mark.asyncio
async def test_off_unsubscribes():
    seen = []
    handler = lambda e: seen.append(e)  # noqa: E731
    client = Aether(FakeProvider())
    client.on(REQUEST_COMPLETE, handler)
    client.off(REQUEST_COMPLETE, handler)
    await client.ask("hi")
    assert seen == []


@pytest.mark.asyncio
async def test_off_for_unsubscribed_handler_is_noop():
    client = Aether(FakeProvider())
    client.off(REQUEST_COMPLETE, lambda e: None)  # should not raise
    await client.ask("hi")


@pytest.mark.asyncio
async def test_on_returns_handler_for_decorator_usage():
    """`client.on(...)` returns the handler so it can be used as a decorator."""
    seen = []
    client = Aether(FakeProvider())

    @client.on(REQUEST_COMPLETE)
    def my_handler(event):
        seen.append(event)

    await client.ask("hi")
    assert len(seen) == 1
    assert my_handler is not None  # still bound


# --- Stream lifecycle ---------------------------------------------------

@pytest.mark.asyncio
async def test_stream_events_fire_for_each_chunk_and_at_boundaries():
    starts: list[StreamStartEvent] = []
    chunks: list[StreamChunkEvent] = []
    completes: list[StreamCompleteEvent] = []
    client = Aether(FakeProvider(canned_response="a b c d"))
    client.on(STREAM_START,    lambda e: starts.append(e))
    client.on(STREAM_CHUNK,    lambda e: chunks.append(e))
    client.on(STREAM_COMPLETE, lambda e: completes.append(e))
    async for _ in client.stream("hi"):
        pass
    assert len(starts) == 1
    assert len(chunks) == 4
    assert len(completes) == 1
    assert completes[0].chunk_count == 4


@pytest.mark.asyncio
async def test_stream_error_event_fires_and_includes_chunk_count():
    class HalfwayFail:
        async def complete(self, request):
            raise NotImplementedError
        async def stream(self, request):
            from aether.llm.contracts import LLMStreamChunk
            yield LLMStreamChunk(text="ok")
            raise ConnectionError("dropped")

    errors: list[StreamErrorEvent] = []
    client = Aether(HalfwayFail())
    client.on(STREAM_ERROR, lambda e: errors.append(e))
    with pytest.raises(ConnectionError):
        async for _ in client.stream("hi"):
            pass
    assert len(errors) == 1
    assert errors[0].chunk_count == 1  # one chunk made it out before the error


# --- Tool lifecycle -----------------------------------------------------

@pytest.mark.asyncio
async def test_tool_start_and_complete_fire(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    starts: list[ToolStartEvent] = []
    completes: list[ToolCompleteEvent] = []
    client = Aether(FakeProvider(responses=[
        _assistant_with_call("add", {"a": 2, "b": 3}),
        _assistant_final("5"),
    ]))
    client.on(TOOL_START,    lambda e: starts.append(e))
    client.on(TOOL_COMPLETE, lambda e: completes.append(e))
    await client.ask("calc", tools=["add"])
    assert len(starts) == 1
    assert starts[0].call.name == "add"
    assert len(completes) == 1
    assert completes[0].result == 5


@pytest.mark.asyncio
async def test_tool_error_event_fires_and_loop_recovers(cleanup_registry):
    """Tool errors become content for the LLM (don't propagate), but
    the error event still fires so observers can record the failure."""
    @register_tool()
    def broken() -> str:
        raise ValueError("kaboom")

    errors: list[ToolErrorEvent] = []
    client = Aether(FakeProvider(responses=[
        _assistant_with_call("broken", {}),
        _assistant_final("recovered"),
    ]))
    client.on(TOOL_ERROR, lambda e: errors.append(e))
    answer = await client.ask("call broken", tools=["broken"])
    assert answer == "recovered"
    assert len(errors) == 1
    assert "kaboom" in str(errors[0].error)


# --- Shared bus across clients -----------------------------------------

@pytest.mark.asyncio
async def test_external_bus_shared_across_clients():
    """One bus, two clients — both clients' events flow to the same subscriber."""
    bus = EventBus()
    seen = []
    bus.on(REQUEST_COMPLETE, lambda e: seen.append(e))

    client_a = Aether(FakeProvider(canned_response="a"), events=bus)
    client_b = Aether(FakeProvider(canned_response="b"), events=bus)
    await client_a.ask("hi")
    await client_b.ask("hi")
    assert len(seen) == 2
