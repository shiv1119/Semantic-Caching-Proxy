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
    """
    Classic token bucket rate limiter for a single client.
    Tokens refill at a steady rate up to capacity — bursts are allowed
    as long as there are tokens available. Once empty, requests are rejected
    until enough tokens accumulate again.
    Good for: allowing short bursts while enforcing a sustained average rate.
    """
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate        # tokens added per second
        self.capacity = capacity
        self.tokens = capacity  # start full
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Tries to consume tokens from the bucket.
        Refills first based on how much time has passed since the last call,
        then checks if there are enough tokens for this request.
        Returns True if allowed, False if the bucket doesn't have enough.
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            # Add tokens for elapsed time, but don't exceed capacity
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class SlidingWindowCounter:
    """
    Sliding window rate limiter — more accurate than a fixed window because
    it doesn't allow a full burst right at the window boundary.

    Tracks request counts per second bucket and evicts stale buckets on each check.
    Returns a retry_after value so the caller knows exactly how long to wait.
    """
    
    def __init__(self, window_size: int = 60, max_requests: int = 100):
        self.window_size = window_size
        self.max_requests = max_requests
        self.counters: Dict[int, int] = defaultdict(int)  # second_timestamp -> request_count
        self.lock = asyncio.Lock()
    
    async def allow_request(self, current_time: int = None) -> Tuple[bool, int]:
        """
        Checks whether a new request fits within the sliding window.
        Prunes timestamps older than window_size seconds on every call —
        keeps memory bounded without a separate cleanup task.
        Returns (allowed, retry_after_seconds).
        """
        if current_time is None:
            current_time = int(time.time())
        
        async with self.lock:
            window_start = current_time - self.window_size
            
            # Drop buckets that have slid out of the window
            old_windows = [t for t in self.counters.keys() if t < window_start]
            for t in old_windows:
                del self.counters[t]
            
            total = sum(self.counters.values())
            
            if total < self.max_requests:
                self.counters[current_time] += 1
                return True, 0
            else:
                # Tell the caller exactly when the oldest bucket will slide out —
                # that's the earliest point a new request would be accepted
                oldest_window = min(self.counters.keys())
                retry_after = (oldest_window + self.window_size) - current_time
                return False, retry_after


class DistributedRateLimiter:
    """
    Redis-based rate limiter for multi-instance deployments.
    Uses a sorted set per client to track request timestamps,
    which allows accurate sliding window counting across multiple app replicas.

    The Lua script runs atomically on the Redis server — this avoids race conditions
    that would occur if we did the read-increment-write in Python across multiple awaits.
    Two limits are enforced: a sustained per-minute rate and a burst cap.
    """
    
    def __init__(self, redis_client, requests_per_minute: int, burst: int):
        self.redis = redis_client
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.rate_per_second = requests_per_minute / 60
        
    async def check(self, client_id: str) -> Tuple[bool, int]:
        """
        Runs the rate limit check atomically via a Lua script.
        The script trims stale entries, checks the burst cap first (stricter),
        then checks the sustained rate. Returns (allowed, retry_after_seconds).

        Using a sorted set with timestamp as score lets us efficiently remove
        old entries with ZREMRANGEBYSCORE instead of scanning everything.
        """
        key = f"ratelimit:{client_id}"
        now = time.time()
        window_start = now - 60
        
        # Atomic Lua script — runs as a single Redis command so no two
        # concurrent requests can interleave their read-modify-write steps
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
            60,  # 1 minute window
            self.requests_per_minute,
            self.burst
        )
        
        return bool(result[0]), result[1]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces rate limits before requests reach route handlers.
    Supports two modes:
    - Distributed: uses Redis (DistributedRateLimiter) when app.state.rate_limiter is set
    - Local: falls back to in-process TokenBucket per client when Redis isn't available

    On rejection, returns 429 with standard rate limit headers so clients
    know exactly when to retry.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        # One TokenBucket per client_id — created lazily on first request
        self.local_limiters = defaultdict(lambda: TokenBucket(
            rate=self.settings.rate_limit_per_minute / 60,
            capacity=self.settings.rate_limit_burst
        ))
        
    async def dispatch(self, request: Request, call_next):
        client_id = self._get_client_id(request)
        
        # Use distributed Redis limiter if available, otherwise fall back to local
        if hasattr(request.app.state, 'rate_limiter'):
            allowed, retry_after = await request.app.state.rate_limiter.check(client_id)
        else:
            limiter = self.local_limiters[client_id]
            allowed = await limiter.consume()
            retry_after = 60  # conservative estimate — local mode doesn't know exact window
        
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
        
        response = await call_next(request)
        
        # Attach rate limit headers to every successful response too —
        # clients can use these to self-throttle before hitting the limit
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_per_minute)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Identifies the client by API key (if present) or IP address.
        API key takes priority — it lets us apply per-key limits rather
        than per-IP, which is more useful when multiple users share a NAT.
        X-Forwarded-For is checked first for clients behind a proxy/load balancer.
        """
        if self.settings.rate_limit_by_api_key and request.headers.get("X-API-Key"):
            return f"apikey:{request.headers['X-API-Key']}"
        
        # X-Forwarded-For can contain a chain of IPs — take the first one (original client)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return f"ip:{client_ip}"