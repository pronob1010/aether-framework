import pytest
import time
from typing import Optional
from aether.llm.contracts import LLMRequest, LLMResponse, LLMProvider, Message
from aether.extensions.llm.retrying import RetryingProvider
from aether.extensions.llm.circuit_breaker import CircuitBreakerProvider, CircuitState, CircuitBreakerOpenException

class FailingProvider(LLMProvider):
    """A mock provider that fails a specific number of times before succeeding."""
    def __init__(self, fail_count: int = 1):
        self.fail_count = fail_count
        self.attempts = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise ValueError(f"Simulated transient failure on attempt {self.attempts}")
        return LLMResponse(
            text="Success",
            model="test-model",
            input_tokens=10,
            output_tokens=10
        )

@pytest.mark.asyncio
async def test_retrying_provider_recovers_from_failures():
    # Will fail twice and succeed on the third try
    inner = FailingProvider(fail_count=2)
    # Using small wait times for fast tests
    provider = RetryingProvider(inner, max_attempts=3, min_wait=0.01, max_wait=0.1)
    
    request = LLMRequest(messages=[Message(role="user", content="test")])
    response = await provider.complete(request)
    
    assert response.text == "Success"
    assert inner.attempts == 3

@pytest.mark.asyncio
async def test_retrying_provider_bubbles_up_after_max_retries():
    inner = FailingProvider(fail_count=4)
    provider = RetryingProvider(inner, max_attempts=3, min_wait=0.01, max_wait=0.1)
    
    request = LLMRequest(messages=[Message(role="user", content="test")])
    with pytest.raises(ValueError, match="Simulated transient failure"):
        await provider.complete(request)
    
    assert inner.attempts == 3

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    inner = FailingProvider(fail_count=10) # Fails consistently
    provider = CircuitBreakerProvider(inner, failure_threshold=2, recovery_timeout=0.1)
    
    request = LLMRequest(messages=[Message(role="user", content="test")])
    
    # First failure
    with pytest.raises(ValueError):
        await provider.complete(request)
    assert provider.state == CircuitState.CLOSED
    
    # Second failure triggers threshold
    with pytest.raises(ValueError):
        await provider.complete(request)
    assert provider.state == CircuitState.OPEN
    
    # Third call fails fast without hitting inner provider
    with pytest.raises(CircuitBreakerOpenException):
        await provider.complete(request)
        
    assert inner.attempts == 2 # Inner provider only hit twice

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    class RecoveringProvider(LLMProvider):
        def __init__(self):
            self.should_fail = True
            
        async def complete(self, request: LLMRequest) -> LLMResponse:
            if self.should_fail:
                raise ValueError("Failing")
            return LLMResponse(text="Recovered", model="m", input_tokens=0, output_tokens=0)
            
    inner = RecoveringProvider()
    provider = CircuitBreakerProvider(inner, failure_threshold=1, recovery_timeout=0.1)
    request = LLMRequest(messages=[Message(role="user", content="test")])
    
    # Trip the breaker
    with pytest.raises(ValueError):
        await provider.complete(request)
    assert provider.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(0.15)
    
    # Make inner provider succeed now
    inner.should_fail = False
    
    # This call should transition to HALF_OPEN, succeed, and go CLOSED
    response = await provider.complete(request)
    assert response.text == "Recovered"
    assert provider.state == CircuitState.CLOSED
