import pytest
from typing import AsyncIterator
from aether import Aether, UsageStats
from aether.llm.contracts import LLMRequest, LLMResponse, LLMStreamChunk, Message
from aether.extensions.llm.fake import FakeProvider
from aether.extensions.llm.cost_tracking import (
    CostTrackingProvider,
    DEFAULT_PRICING,
    ModelPricing,
    TokenUsage,
)
from aether.extensions.llm.retrying import RetryingProvider
from aether.extensions.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CostTrackingConfig,
    build_provider,
)


# --- complete() recording ------------------------------------------------

@pytest.mark.asyncio
async def test_complete_records_tokens_per_model():
    tracker = CostTrackingProvider(FakeProvider(canned_response="one two three"))
    await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi there")]))
    assert tracker.stats.total_requests == 1
    assert tracker.stats.total_input_tokens == 2     # "hi there"
    assert tracker.stats.total_output_tokens == 3    # "one two three"
    assert tracker.stats.by_model["fake-model"].requests == 1


@pytest.mark.asyncio
async def test_complete_accumulates_across_calls():
    tracker = CostTrackingProvider(FakeProvider(canned_response="ok"))
    for _ in range(5):
        await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert tracker.stats.total_requests == 5
    assert tracker.stats.by_model["fake-model"].requests == 5


class TwoModelProvider:
    """Returns a different model name on each call — to test per-model split."""
    def __init__(self):
        self.toggle = False

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.toggle = not self.toggle
        model = "gpt-4o" if self.toggle else "gpt-4o-mini"
        return LLMResponse(text="x", model=model, input_tokens=10, output_tokens=5)

    async def stream(self, request): ...


@pytest.mark.asyncio
async def test_per_model_breakdown_is_correct():
    tracker = CostTrackingProvider(TwoModelProvider())
    for _ in range(4):
        await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert tracker.stats.by_model["gpt-4o"].requests == 2
    assert tracker.stats.by_model["gpt-4o-mini"].requests == 2
    assert tracker.stats.by_model["gpt-4o"].input_tokens == 20


# --- stream() recording --------------------------------------------------

@pytest.mark.asyncio
async def test_stream_records_only_final_chunk_token_counts():
    tracker = CostTrackingProvider(FakeProvider(canned_response="alpha beta gamma"))
    async for _ in tracker.stream(LLMRequest(messages=[Message(role="user", content="hi")])):
        pass
    # FakeProvider only emits tokens on the final chunk (3 words)
    assert tracker.stats.total_output_tokens == 3
    assert tracker.stats.total_requests == 1


class AbortMidStreamProvider:
    async def complete(self, request): ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(text="started")
        raise RuntimeError("aborted")


@pytest.mark.asyncio
async def test_aborted_stream_records_nothing():
    """If a stream errors before the final chunk's usage data, don't count it."""
    tracker = CostTrackingProvider(AbortMidStreamProvider())
    with pytest.raises(RuntimeError):
        async for _ in tracker.stream(LLMRequest(messages=[Message(role="user", content="hi")])):
            pass
    assert tracker.stats.total_requests == 0


# --- Cost calculation ----------------------------------------------------

@pytest.mark.asyncio
async def test_total_cost_is_none_without_pricing():
    tracker = CostTrackingProvider(FakeProvider())
    await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert tracker.stats.total_cost_usd is None


@pytest.mark.asyncio
async def test_total_cost_uses_default_pricing():
    tracker = CostTrackingProvider(FakeProvider(canned_response="x"), pricing=DEFAULT_PRICING)
    await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    # fake-model is priced at $0 → total = 0
    assert tracker.stats.total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_total_cost_uses_custom_pricing():
    custom = {"fake-model": ModelPricing(input_per_1m=1_000_000, output_per_1m=2_000_000)}
    tracker = CostTrackingProvider(
        FakeProvider(canned_response="one two three"),
        pricing=custom,
    )
    # 1 token of "hi" prompt → 1 input token, "one two three" → 3 output tokens
    await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    # Cost = (1/1M * 1M) + (3/1M * 2M) = 1 + 6 = 7
    assert tracker.stats.total_cost_usd == 7.0


def test_total_cost_skips_models_without_pricing():
    """If we observe an unpriced model, partial sum > 0 only from priced ones."""
    stats = UsageStats(
        by_model={
            "gpt-4o": TokenUsage(input_tokens=1_000_000, output_tokens=0, requests=1),
            "mystery-model": TokenUsage(input_tokens=1_000_000, output_tokens=0, requests=1),
        },
        pricing=DEFAULT_PRICING,
    )
    # gpt-4o has pricing ($2.50/1M input), mystery-model doesn't → only gpt-4o counted
    assert stats.total_cost_usd == 2.50


# --- Aether facade integration ------------------------------------------

@pytest.mark.asyncio
async def test_aether_usage_exposes_stats_when_tracking_on():
    client = Aether.from_config(ProviderConfig(
        name="fake",
        cost_tracking=CostTrackingConfig(),
    ))
    await client.ask("hi")
    assert client.usage.total_requests == 1


@pytest.mark.asyncio
async def test_aether_usage_returns_empty_when_tracking_off():
    """No surprises — property always callable, just shows zeros."""
    client = Aether.from_config(ProviderConfig(name="fake"))
    await client.ask("hi")
    assert client.usage.total_requests == 0
    assert isinstance(client.usage, UsageStats)


@pytest.mark.asyncio
async def test_aether_usage_walks_through_retry_wrapper():
    """Cost tracking sits OUTSIDE retry — usage should be reachable through it."""
    client = Aether.from_config(ProviderConfig(
        name="fake",
        retry=RetryConfig(max_attempts=2, min_wait=0.01, max_wait=0.02),
        cost_tracking=CostTrackingConfig(),
    ))
    await client.ask("hi")
    assert client.usage.total_requests == 1


# --- Decorator-order semantics ------------------------------------------

class FlakyProvider:
    """Fails N times before succeeding — to verify cost tracker is outermost."""
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.attempts = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError(f"attempt {self.attempts}")
        return LLMResponse(text="ok", model="fake-model", input_tokens=10, output_tokens=5)

    async def stream(self, request): ...


@pytest.mark.asyncio
async def test_cost_tracking_only_counts_successful_final_outcome():
    """Retry fails twice then succeeds; cost tracker should record ONCE."""
    flaky = FlakyProvider(fail_times=2)
    retry = RetryingProvider(flaky, max_attempts=3, min_wait=0.01, max_wait=0.02)
    # Builder puts cost tracking OUTERMOST — same shape here.
    tracker = CostTrackingProvider(retry)
    await tracker.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
    assert flaky.attempts == 3                  # 2 fails + 1 success at the base
    assert tracker.stats.total_requests == 1    # but ONLY 1 success counted
    assert tracker.stats.total_input_tokens == 10
