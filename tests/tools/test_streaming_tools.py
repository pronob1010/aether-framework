"""Streaming with tool calls — the LLM streams text, requests a tool,
gets the result, streams more text, and the user sees one logical stream.

Uses FakeProvider's `streaming_responses` mode to script each provider
stream session (text + tool_calls per session). The facade hides the
multi-session machinery from the caller.
"""
import pytest
from aether import Aether, register_tool, EventBus
from aether.llm.contracts import LLMStreamChunk, ToolCall
from aether.extensions.llm.fake import FakeProvider
from aether.events import (
    STREAM_START, STREAM_CHUNK, STREAM_COMPLETE,
    TOOL_START, TOOL_COMPLETE, TOOL_ERROR,
)
from aether.registry import REGISTRY


@pytest.fixture
def cleanup_registry():
    original = {kind: dict(specs) for kind, specs in REGISTRY.items()}
    yield
    REGISTRY.clear()
    for kind, specs in original.items():
        REGISTRY[kind] = specs


# --- Single tool call in stream -----------------------------------------

@pytest.mark.asyncio
async def test_stream_with_one_tool_call_round_trip(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    fake = FakeProvider(streaming_responses=[
        # Session 1: stream some text, then a tool_calls chunk
        [
            LLMStreamChunk(text="Let me "),
            LLMStreamChunk(text="add those."),
            LLMStreamChunk(
                text="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
            ),
        ],
        # Session 2: stream the final answer after tool result
        [
            LLMStreamChunk(text="The answer "),
            LLMStreamChunk(text="is 5."),
            LLMStreamChunk(text="", finish_reason="stop"),
        ],
    ])

    client = Aether(fake)
    parts = []
    async for chunk in client.stream("calc", tools=["add"]):
        parts.append(chunk.text)

    # User-visible stream: text from BOTH sessions, no tool-call chunks.
    full = "".join(parts)
    assert "Let me add those." in full
    assert "The answer is 5." in full
    # Provider was called twice (two sessions)
    assert len(fake.calls) == 2
    # Second call's messages include the assistant turn + the tool result.
    second = fake.calls[1].messages
    tool_msg = next(m for m in second if m.role == "tool")
    assert tool_msg.content == "5"
    assert tool_msg.tool_call_id == "c1"


@pytest.mark.asyncio
async def test_stream_text_with_tool_call_yields_only_text(cleanup_registry):
    @register_tool()
    def echo(msg: str) -> str:
        return msg

    fake = FakeProvider(streaming_responses=[
        [
            LLMStreamChunk(text="thinking "),
            LLMStreamChunk(
                text="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"msg": "hi"})],
            ),
        ],
        [
            LLMStreamChunk(text="done"),
            LLMStreamChunk(text="", finish_reason="stop"),
        ],
    ])

    client = Aether(fake)
    text = "".join([t async for t in client.stream_text("go", tools=["echo"])])
    assert text == "thinking done"


# --- Multiple parallel tool calls in one session -----------------------

@pytest.mark.asyncio
async def test_stream_with_multiple_parallel_tool_calls(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    @register_tool()
    def mul(a: int, b: int) -> int:
        return a * b

    fake = FakeProvider(streaming_responses=[
        # Session 1: two tool calls at once
        [
            LLMStreamChunk(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
                    ToolCall(id="c2", name="mul", arguments={"a": 4, "b": 5}),
                ],
            ),
        ],
        # Session 2: final answer
        [
            LLMStreamChunk(text="Sum=5, product=20.", finish_reason="stop"),
        ],
    ])

    client = Aether(fake)
    text = "".join([t async for t in client.stream_text("both", tools=["add", "mul"])])
    assert text == "Sum=5, product=20."
    # Second call has both tool results
    tool_msgs = [m for m in fake.calls[1].messages if m.role == "tool"]
    assert len(tool_msgs) == 2
    assert {m.content for m in tool_msgs} == {"5", "20"}


# --- Multi-iteration tool loop -----------------------------------------

@pytest.mark.asyncio
async def test_stream_with_multiple_iterations(cleanup_registry):
    @register_tool()
    def step() -> str:
        return "ok"

    fake = FakeProvider(streaming_responses=[
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c1", name="step", arguments={})])],
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c2", name="step", arguments={})])],
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c3", name="step", arguments={})])],
        [LLMStreamChunk(text="done", finish_reason="stop")],
    ])

    client = Aether(fake)
    text = "".join([t async for t in client.stream_text("loop", tools=["step"])])
    assert text == "done"
    assert len(fake.calls) == 4


# --- Iteration cap ------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_respects_max_tool_iterations(cleanup_registry):
    @register_tool()
    def forever() -> str:
        return "again"

    fake = FakeProvider(streaming_responses=[
        # 100 tool-call sessions; cap should stop at 2
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id=f"c{i}", name="forever", arguments={})])]
        for i in range(100)
    ])

    client = Aether(fake)
    _ = [t async for t in client.stream_text("loop", tools=["forever"], max_tool_iterations=2)]
    # 2 iterations + 1 initial = 3 sessions
    assert len(fake.calls) == 3


# --- Tool error during streaming ---------------------------------------

@pytest.mark.asyncio
async def test_tool_error_during_stream_recovers(cleanup_registry):
    @register_tool()
    def broken() -> str:
        raise ValueError("kaboom")

    fake = FakeProvider(streaming_responses=[
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c1", name="broken", arguments={})])],
        [LLMStreamChunk(text="recovered", finish_reason="stop")],
    ])

    client = Aether(fake)
    text = "".join([t async for t in client.stream_text("go", tools=["broken"])])
    assert text == "recovered"
    # Tool error was reported back to LLM as content, not raised
    tool_msg = next(m for m in fake.calls[1].messages if m.role == "tool")
    assert "kaboom" in tool_msg.content


# --- Event semantics ---------------------------------------------------

@pytest.mark.asyncio
async def test_stream_events_fire_once_across_tool_loop(cleanup_registry):
    """stream.start and stream.complete fire ONCE for the user-facing stream,
    even though there are multiple internal sessions."""
    @register_tool()
    def step() -> str:
        return "ok"

    fake = FakeProvider(streaming_responses=[
        [LLMStreamChunk(text="working", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c1", name="step", arguments={})])],
        [LLMStreamChunk(text=" done", finish_reason="stop")],
    ])

    bus = EventBus()
    starts, completes = [], []
    bus.on(STREAM_START, lambda e: starts.append(e))
    bus.on(STREAM_COMPLETE, lambda e: completes.append(e))

    client = Aether(fake, events=bus)
    _ = [c async for c in client.stream("go", tools=["step"])]

    assert len(starts) == 1
    assert len(completes) == 1


@pytest.mark.asyncio
async def test_stream_chunk_event_does_not_fire_for_tool_call_chunks(cleanup_registry):
    """stream.chunk should only fire for user-facing text chunks,
    not for internal control-flow tool_call chunks."""
    @register_tool()
    def step() -> str:
        return "ok"

    fake = FakeProvider(streaming_responses=[
        [LLMStreamChunk(text="hi"),
         LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c1", name="step", arguments={})])],
        [LLMStreamChunk(text=" bye", finish_reason="stop")],
    ])

    chunks = []
    client = Aether(fake)
    client.on(STREAM_CHUNK, lambda e: chunks.append(e.chunk))

    _ = [c async for c in client.stream("go", tools=["step"])]
    # Should see "hi" and " bye" — NOT the tool_calls chunk.
    assert len(chunks) == 2
    assert all(not c.tool_calls for c in chunks)


@pytest.mark.asyncio
async def test_tool_events_fire_during_stream(cleanup_registry):
    @register_tool()
    def add(a: int, b: int) -> int:
        return a + b

    fake = FakeProvider(streaming_responses=[
        [LLMStreamChunk(text="", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})])],
        [LLMStreamChunk(text="5", finish_reason="stop")],
    ])

    starts, completes, errors = [], [], []
    client = Aether(fake)
    client.on(TOOL_START,    lambda e: starts.append(e))
    client.on(TOOL_COMPLETE, lambda e: completes.append(e))
    client.on(TOOL_ERROR,    lambda e: errors.append(e))

    _ = [c async for c in client.stream("go", tools=["add"])]

    assert len(starts) == 1
    assert len(completes) == 1
    assert completes[0].result == 5
    assert len(errors) == 0
