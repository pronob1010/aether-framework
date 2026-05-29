from pydantic import BaseModel
from aether.llm.contracts import LLMProvider
from aether.providers.llm.factory import make_provider

class RetryConfig(BaseModel):
    max_attempts: int = 3
    min_wait: float = 1.0
    max_wait: float = 10.0

class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    recovery_timeout: float = 30.0

class ProviderConfig(BaseModel):
    name: str
    api_key: str | None = None
    default_model: str | None = None
    retry: RetryConfig | None = None
    circuit_breaker: CircuitBreakerConfig | None = None

def build_provider(config: ProviderConfig) -> LLMProvider:
    kwargs = {}
    if config.api_key:       kwargs["api_key"] = config.api_key
    if config.default_model: kwargs["default_model"] = config.default_model
    provider = make_provider(config.name, **kwargs)

    # Order is load-bearing: retry sits INSIDE the breaker so one
    # retry-exhausted call counts as ONE breaker failure, not N.
    if config.retry:
        from aether.providers.llm.retrying import RetryingProvider
        provider = RetryingProvider(provider, **config.retry.model_dump())
    if config.circuit_breaker:
        from aether.providers.llm.circuit_breaker import CircuitBreakerProvider
        provider = CircuitBreakerProvider(provider, **config.circuit_breaker.model_dump())

    return provider
