import httpx
import redis.asyncio as redis
from typing import Dict, Any
from app.config import Settings
import logging

logger = logging.getLogger(__name__)

class ConnectionPoolManager:
    """Manage all connection pools for optimal resource usage"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pools: Dict[str, Any] = {}
        
    async def initialize_all(self):
        """Initialize all connection pools"""
        # HTTP pool for Ollama
        self.pools["http"] = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=self.settings.connection_pool_size,
                max_connections=self.settings.connection_pool_size + self.settings.connection_pool_overflow,
                keepalive_expiry=self.settings.keepalive_timeout
            ),
            timeout=httpx.Timeout(self.settings.connection_timeout),
            http2=True
        )
        
        # Redis connection will be initialized separately
        logger.info(f"Initialized connection pools with {self.settings.connection_pool_size} connections")
    
    async def get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client from pool"""
        return self.pools.get("http")
    
    async def close_all(self):
        """Close all connection pools"""
        for name, pool in self.pools.items():
            if hasattr(pool, 'aclose'):
                await pool.aclose()
            elif hasattr(pool, 'close'):
                await pool.close()
        
        logger.info("Closed all connection pools")