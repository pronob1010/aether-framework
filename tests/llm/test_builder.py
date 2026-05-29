import pytest
from aether.providers.llm.builder import (
    ProviderConfig,
    RetryConfig,
    CircuitBreakerConfig,
    build_provider,
)
from aether.providers.llm.fake import FakeProvider
from aether.providers.llm.retrying import RetryingProvider
from aether.providers.llm.circuit_breaker import CircuitBreakerProvider


def test_build_provider_bare_returns_concrete_provider():
    provider = build_provider(ProviderConfig(name="fake"))
    assert isinstance(provider, FakeProvider)


def test_build_provider_with_retry_only():
    provider = build_provider(ProviderConfig(name="fake", retry=RetryConfig()))
    assert isinstance(provider, RetryingProvider)
    assert isinstance(provider.inner_provider, FakeProvider)


def test_build_provider_with_circuit_breaker_only():
    provider = build_provider(
        ProviderConfig(name="fake", circuit_breaker=CircuitBreakerConfig())
    )
    assert isinstance(provider, CircuitBreakerProvider)
    assert isinstance(provider.inner_provider, FakeProvider)


def test_build_provider_full_stack_orders_retry_inside_breaker():
    """Order is load-bearing: retry must sit INSIDE the breaker so one
    retry-exhausted call counts as ONE breaker failure, not N."""
    provider = build_provider(ProviderConfig(
        name="fake",
        retry=RetryConfig(),
        circuit_breaker=CircuitBreakerConfig(),
    ))
    assert isinstance(provider, CircuitBreakerProvider)
    assert isinstance(provider.inner_provider, RetryingProvider)
    assert isinstance(provider.inner_provider.inner_provider, FakeProvider)


def test_build_provider_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider(ProviderConfig(name="bogus"))
