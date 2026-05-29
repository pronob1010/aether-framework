import pytest
from typing import AsyncIterator
from aether import Aether
from aether.llm.contracts import LLMRequest, LLMStreamChunk, Message
from aether.extensions.llm.fake import FakeProvider
from aether.extensions.llm.retrying import RetryingProvider
from aether.extensions.llm.circuit_breaker import (
    CircuitBreakerProvider,
    CircuitState,
    CircuitBreakerOpenException,
)


# --- FakeProvider streaming -----------------------------------------------

@pytest.mark.asyncio
async def test_fake_provider_stream_yields_word_deltas():
    fake = FakeProvider(canned_response="hello world how are you")
    chunks = [c async for c in fake.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    # 5 words → 5 chunks
    assert len(chunks) == 5
    # First chunk has no leading space, rest do
    assert chunks[0].text == "hello"
    assert chunks[1].text == " world"


@pytest.mark.asyncio
async def test_fake_provider_stream_final_chunk_carries_metadata():
    fake = FakeProvider(canned_response="one two")
    chunks = [c async for c in fake.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    # Final chunk has model, finish_reason, token counts
    assert chunks[-1].model == "fake-model"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].output_tokens == 2
    # Non-final chunks don't
    assert chunks[0].model is None
    assert chunks[0].finish_reason is None


# --- Aether facade streaming ---------------------------------------------

@pytest.mark.asyncio
async def test_aether_stream_yields_rich_chunks():
    fake = FakeProvider(canned_response="alpha beta")
    client = Aether(fake)
    chunks = [c async for c in client.stream("hi")]
    assert len(chunks) == 2
    assert chunks[0].text == "alpha"
    assert chunks[1].text == " beta"
    assert chunks[1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_aether_stream_text_yields_only_strings():
    fake = FakeProvider(canned_response="alpha beta gamma")
    client = Aether(fake)
    text_parts = [t async for t in client.stream_text("hi")]
    assert "".join(text_parts) == "alpha beta gamma"


@pytest.mark.asyncio
async def test_aether_stream_forwards_model_and_temperature():
    fake = FakeProvider()
    client = Aether(fake)
    _ = [c async for c in client.stream("hi", model="gpt-4o", temperature=0.0)]
    sent = fake.calls[0]
    assert sent.model == "gpt-4o"
    assert sent.temperature == 0.0


# --- Retry + streaming ---------------------------------------------------

class FlakyStreamProvider:
    """Fails opening the stream `fail_times` times, then streams normally."""
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.attempts = 0

    async def complete(self, request): ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError(f"attempt {self.attempts} fails before yielding")
        yield LLMStreamChunk(text="one")
        yield LLMStreamChunk(text=" two", finish_reason="stop")


@pytest.mark.asyncio
async def test_retry_recovers_when_handshake_fails():
    """Failures BEFORE the first chunk trigger retry, eventually succeeding."""
    flaky = FlakyStreamProvider(fail_times=2)
    retrying = RetryingProvider(flaky, max_attempts=3, min_wait=0.01, max_wait=0.02)
    chunks = [c async for c in retrying.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    assert flaky.attempts == 3       # 2 fails + 1 success
    assert chunks[0].text == "one"


@pytest.mark.asyncio
async def test_retry_gives_up_after_max_attempts():
    flaky = FlakyStreamProvider(fail_times=5)
    retrying = RetryingProvider(flaky, max_attempts=3, min_wait=0.01, max_wait=0.02)
    with pytest.raises(ConnectionError):
        _ = [c async for c in retrying.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    assert flaky.attempts == 3       # stopped at max


class MidStreamFailureProvider:
    """Yields one chunk successfully, then errors mid-stream."""
    def __init__(self):
        self.attempts = 0

    async def complete(self, request): ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        self.attempts += 1
        yield LLMStreamChunk(text="first")
        raise ConnectionError("died mid-stream")


@pytest.mark.asyncio
async def test_retry_does_not_apply_after_first_chunk():
    """Once a chunk has yielded, mid-stream errors propagate — no retry."""
    midfail = MidStreamFailureProvider()
    retrying = RetryingProvider(midfail, max_attempts=3, min_wait=0.01, max_wait=0.02)
    yielded = []
    with pytest.raises(ConnectionError, match="died mid-stream"):
        async for chunk in retrying.stream(LLMRequest(messages=[Message(role="user", content="hi")])):
            yielded.append(chunk)
    assert midfail.attempts == 1     # NOT retried
    assert len(yielded) == 1         # caller saw the first chunk before the failure


# --- Circuit breaker + streaming -----------------------------------------

class AlwaysFailStreamProvider:
    async def complete(self, request): ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        raise RuntimeError("backend down")
        yield  # noqa: unreachable — marks this as an async generator


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failed_streams():
    breaker = CircuitBreakerProvider(
        AlwaysFailStreamProvider(),
        failure_threshold=3,
        recovery_timeout=60,
    )
    for _ in range(3):
        with pytest.raises(RuntimeError):
            _ = [c async for c in breaker.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_fails_fast_when_open():
    breaker = CircuitBreakerProvider(
        AlwaysFailStreamProvider(),
        failure_threshold=1,
        recovery_timeout=60,
    )
    # Trip the breaker
    with pytest.raises(RuntimeError):
        _ = [c async for c in breaker.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]

    # Next stream should fail fast — no chunks yielded
    yielded = []
    with pytest.raises(CircuitBreakerOpenException):
        async for chunk in breaker.stream(LLMRequest(messages=[Message(role="user", content="hi")])):
            yielded.append(chunk)
    assert yielded == []


@pytest.mark.asyncio
async def test_circuit_breaker_counts_successful_stream():
    fake = FakeProvider(canned_response="ok")
    breaker = CircuitBreakerProvider(fake, failure_threshold=3, recovery_timeout=60)
    _ = [c async for c in breaker.stream(LLMRequest(messages=[Message(role="user", content="hi")]))]
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
