from enum import Enum
import time
import asyncio
from collections import deque
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    """Circuit breaker pattern for LLM API calls"""
    
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
        self.recent_failures = deque(maxlen=10)
        self.success_count = 0
        self.lock = asyncio.Lock()
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
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
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.half_open_max_calls:
                        logger.info(f"Circuit {self.name} recovered, closing")
                        self._reset()
                elif self.state == CircuitState.CLOSED:
                    # Reset failure count on success in closed state
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            async with self.lock:
                self._record_failure(e)
            raise
    
    def _record_failure(self, error: Exception):
        """Record failure and potentially open circuit"""
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
            logger.error(f"Circuit {self.name} reopening due to failure in half-open state")
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_calls = 0
        self.recent_failures.clear()
        self.success_count = 0
    
    def get_state(self) -> dict:
        """Get current circuit state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "recent_failures": list(self.recent_failures),
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else 0
        }

class CircuitBreakerRegistry:
    """Manage multiple circuit breakers"""
    
    def __init__(self):
        self.breakers = {}
        self.lock = asyncio.Lock()
    
    async def get(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker"""
        async with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self.breakers[name]
    
    async def get_all_states(self) -> dict:
        """Get states of all circuit breakers"""
        return {name: breaker.get_state() for name, breaker in self.breakers.items()}