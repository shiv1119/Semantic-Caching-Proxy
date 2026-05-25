from enum import Enum
import time
import asyncio
from collections import deque
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation — all requests go through
    OPEN = "open"           # Too many failures — requests are rejected immediately
    HALF_OPEN = "half_open" # Recovery probe — a few test requests are allowed through

class CircuitBreaker:
    """
    Implements the circuit breaker pattern to protect the Ollama LLM API
    from cascading failures.

    The idea: if Ollama starts failing repeatedly, stop hammering it and
    give it time to recover. After recovery_timeout seconds, allow a small
    number of test requests through (HALF_OPEN). If they succeed, close
    the circuit and resume normal traffic. If they fail, reopen and wait again.

    States:
        CLOSED   → normal, requests pass through
        OPEN     → breaker tripped, requests fail fast
        HALF_OPEN → recovery probe, limited requests allowed
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
        self.recent_failures = deque(maxlen=10)  # rolling window for diagnostics
        self.success_count = 0
        self.lock = asyncio.Lock()  # protects state transitions under async concurrency
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Wraps an async function call with circuit breaker protection.

        If the circuit is OPEN and the recovery timeout hasn't elapsed yet,
        the call is rejected immediately without touching the LLM.
        Once the timeout passes, we move to HALF_OPEN and let a few calls
        through to test if Ollama has recovered.

        On success in HALF_OPEN: close the circuit after half_open_max_calls succeed.
        On failure anywhere: record it and potentially trip the breaker.
        In CLOSED state, a single success slightly decays the failure count
        to avoid tripping on isolated blips.
        """
        async with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                else:
                    remaining = self.recovery_timeout - (time.time() - self.last_failure_time)
                    raise Exception(f"Circuit {self.name} is OPEN (recovery in {remaining:.0f}s)")
        
        try:
            result = await func(*args, **kwargs)
            
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.half_open_max_calls:
                        logger.info(f"Circuit {self.name} recovered, closing")
                        self._reset()
                elif self.state == CircuitState.CLOSED:
                    # Gradually decay failure_count on success so isolated errors
                    # don't accumulate and trip the breaker unfairly
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            async with self.lock:
                self._record_failure(e)
            raise
    
    def _record_failure(self, error: Exception):
        """
        Increments the failure count and trips the circuit if we've crossed
        the threshold. In HALF_OPEN, any single failure immediately reopens
        the circuit — we're not confident enough yet to tolerate more failures.
        """
        self.failure_count += 1
        self.recent_failures.append({
            "timestamp": time.time(),
            "error": str(error)
        })
        
        logger.warning(f"Circuit {self.name} recorded failure {self.failure_count}/{self.failure_threshold}")
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit {self.name} opening due to {self.failure_count} failures")
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
        
        elif self.state == CircuitState.HALF_OPEN:
            # One failure during recovery is enough to send us back to OPEN —
            # the service clearly isn't stable yet
            logger.error(f"Circuit {self.name} reopening due to failure in half-open state")
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
    
    def _reset(self):
        """
        Fully resets the breaker to CLOSED. Called after a successful
        HALF_OPEN probe run — clears all failure history so we start fresh.
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.recent_failures.clear()
        self.success_count = 0
    
    def get_state(self) -> dict:
        """
        Snapshot of the current breaker state — exposed via the /metrics endpoint.
        time_since_last_failure helps understand how close we are to recovery timeout.
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "recent_failures": list(self.recent_failures),
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else 0
        }


class CircuitBreakerRegistry:
    """
    A shared registry so different parts of the app can look up the same
    circuit breaker instance by name rather than passing objects around.
    Useful when you want separate breakers for different downstream services
    (e.g. one for Ollama generate, one for Ollama embeddings).
    """
    
    def __init__(self):
        self.breakers = {}
        self.lock = asyncio.Lock()
    
    async def get(self, name: str, **kwargs) -> CircuitBreaker:
        """
        Returns an existing breaker by name or creates a new one if it doesn't exist.
        Thread-safe — the lock ensures two concurrent callers don't both try
        to create the same breaker simultaneously.
        """
        async with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self.breakers[name]
    
    async def get_all_states(self) -> dict:
        """Returns a state snapshot for every registered breaker — useful for health checks."""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}