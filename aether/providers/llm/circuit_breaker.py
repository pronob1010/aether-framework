import time
import logging
from enum import Enum
from aether.llm.contracts import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"         # Normal operation
    OPEN = "OPEN"             # Failing fast
    HALF_OPEN = "HALF_OPEN"   # Testing if recovered

class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is open and requests are failing fast."""
    pass

class CircuitBreakerProvider:
    """
    A decorator provider implementing the Circuit Breaker pattern.
    Transitions between CLOSED, OPEN, and HALF_OPEN states based on failure thresholds.
    """
    def __init__(
        self,
        inner_provider: LLMProvider,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.inner_provider = inner_provider
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._check_state_transition()

        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenException("Circuit breaker is OPEN. Failing fast.")

        try:
            response = await self.inner_provider.complete(request)
            self._on_success()
            return response
        except Exception as e:
            if not isinstance(e, CircuitBreakerOpenException):
                self._on_failure()
            raise

    def _check_state_transition(self) -> None:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN to test recovery.")
                self.state = CircuitState.HALF_OPEN

    def _on_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker successful in HALF_OPEN. Transitioning to CLOSED.")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker failed in HALF_OPEN. Reverting to OPEN.")
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker failure threshold ({self.failure_threshold}) reached. Transitioning to OPEN.")
            self.state = CircuitState.OPEN
