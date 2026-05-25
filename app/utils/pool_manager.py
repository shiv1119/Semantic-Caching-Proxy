import httpx
import redis.asyncio as redis
from typing import Dict, Any
from app.config import Settings
import logging

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """Instead of opening a brand new connection every time we talk to Ollama or Redis,
    we keep a pool of reusable connections ready to go — saves a lot of overhead under load"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pools: Dict[str, Any] = {}  # all our pools live here, keyed by name
        
    async def initialize_all(self):
        """Sets up all the connection pools when the server starts.
        Right now that's just HTTP for Ollama — Redis handles its own connection separately"""
        
        # httpx.AsyncClient acts as our HTTP connection pool for talking to Ollama
        # instead of a new TCP handshake every request, we reuse existing connections
        self.pools["http"] = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=self.settings.connection_pool_size,           # idle connections to keep warm
                max_connections=self.settings.connection_pool_size + self.settings.connection_pool_overflow,  # hard ceiling including burst traffic
                keepalive_expiry=self.settings.keepalive_timeout                        # drop idle connections after this long
            ),
            timeout=httpx.Timeout(self.settings.connection_timeout),  # give up if Ollama doesn't respond in time
            http2=True  # HTTP/2 lets us multiplex requests over one connection — faster than HTTP/1.1
        )
        
        # Redis sets up its own pool inside RedisVectorClient, so we skip it here
        logger.info(f"Initialized connection pools with {self.settings.connection_pool_size} connections")
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """Hands out the shared HTTP client — everyone uses the same pool, not their own client"""
        return self.pools.get("http")
    
    async def close_all(self):
        """Cleanly shuts down every pool when the server is stopping.
        We check for both aclose() and close() because different libraries use different names"""
        for name, pool in self.pools.items():
            if hasattr(pool, 'aclose'):
                await pool.aclose()   # async close — for httpx and similar async libraries
            elif hasattr(pool, 'close'):
                await pool.close()    # regular close — fallback for sync-style clients
        
        logger.info("Closed all connection pools")