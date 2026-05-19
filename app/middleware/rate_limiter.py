from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict, Tuple
import asyncio
from collections import defaultdict
import redis.asyncio as redis
from app.config import get_settings

class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens, return True if allowed"""
        async with self.lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class SlidingWindowCounter:
    """Sliding window counter for more accurate rate limiting"""
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        self.window_size = window_size
        self.max_requests = max_requests
        self.counters: Dict[int, int] = defaultdict(int)
        self.lock = asyncio.Lock()
    
    async def allow_request(self, current_time: int = None) -> Tuple[bool, int]:
        """Check if request is allowed, return (allowed, retry_after)"""
        if current_time is None:
            current_time = int(time.time())
        
        async with self.lock:
            window_start = current_time - self.window_size
            
            # Clean old counters
            old_windows = [t for t in self.counters.keys() if t < window_start]
            for t in old_windows:
                del self.counters[t]
            
            # Calculate current total
            total = sum(self.counters.values())
            
            if total < self.max_requests:
                self.counters[current_time] += 1
                return True, 0
            else:
                # Calculate retry after
                oldest_window = min(self.counters.keys())
                retry_after = (oldest_window + self.window_size) - current_time
                return False, retry_after

class DistributedRateLimiter:
    """Redis-based distributed rate limiter"""
    
    def __init__(self, redis_client, requests_per_minute: int, burst: int):
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.rate_per_second = requests_per_minute / 60
        
    async def check(self, client_id: str) -> Tuple[bool, int]:
        """Check rate limit using Redis sorted sets (distributed)"""
        key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local burst = tonumber(ARGV[4])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
        
        -- Get current count
        local count = redis.call('ZCARD', key)
        
        -- Check burst limit first
        if count >= burst then
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = window - (now - oldest[2])
            return {0, math.ceil(retry_after)}
        end
        
        -- Check sustained rate
        if count < max_requests then
            redis.call('ZADD', key, now, now .. ':' .. math.random())
            redis.call('EXPIRE', key, window + 10)
            return {1, max_requests - count - 1}
        else
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = window - (now - oldest[2])
            return {0, math.ceil(retry_after)}
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            now,
            60,  # window size
            self.requests_per_minute,
            self.burst
        )
        
        return bool(result[0]), result[1]

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.local_limiters = defaultdict(lambda: TokenBucket(
            rate=self.settings.rate_limit_per_minute / 60,
            capacity=self.settings.rate_limit_burst
        ))
        
    async def dispatch(self, request: Request, call_next):
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if hasattr(request.app.state, 'rate_limiter'):
            # Distributed mode
            allowed, retry_after = await request.app.state.rate_limiter.check(client_id)
        else:
            # Local mode
            limiter = self.local_limiters[client_id]
            allowed = await limiter.consume()
            retry_after = 60
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit": self.settings.rate_limit_per_minute
                },
                headers={
                    "X-RateLimit-Limit": str(self.settings.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                    "Retry-After": str(retry_after)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_per_minute)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier (IP or API key)"""
        if self.settings.rate_limit_by_api_key and request.headers.get("X-API-Key"):
            return f"apikey:{request.headers['X-API-Key']}"
        
        # Use IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"