import redis.asyncio as redis
import numpy as np
from typing import List, Optional, Tuple, Dict
import json
import asyncio
import time
from app.config import Settings
import logging
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)

class RedisVectorClient:
    def __init__(self, redis_url: str, index_name: str, settings: Settings):
        self.redis_url = redis_url
        self.index_name = index_name
        self.settings = settings
        self.redis = None
        self.pipeline_batch_size = 100
        self._batch_queue = asyncio.Queue(maxsize=settings.cache_async_write_queue_size)
        self._batch_worker_task = None
        self.dimension = settings.redis_vector_dimension
        
    async def connect(self):
        """Establish connection pool with optimized settings"""
        self.redis = await redis.from_url(
            self.redis_url,
            decode_responses=True,
            max_connections=self.settings.redis_pool_size,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError]
        )
        
        # Test connection
        await self.redis.ping()
        
        # Create index
        await self._create_index()
        
        # Start batch worker for async writes
        if self.settings.cache_write_behind:
            self._batch_worker_task = asyncio.create_task(self._batch_write_worker())
        
        logger.info(f"Redis client connected to {self.redis_url}")
    
    async def _create_index(self):
        """Create Redis vector similarity index with HNSW (safe for multi-worker startup)"""
        try:
            # Try to drop index if exists (ignore if doesn't exist)
            try:
                await self.redis.execute_command(f"FT.DROPINDEX {self.index_name}")
                logger.info(f"Dropped existing index {self.index_name}")
            except ResponseError as e:
                if "Unknown index name" not in str(e):
                    logger.warning(f"Drop index warning: {e}")

            # Create index with correct HNSW parameters
            create_command = (
                f"FT.CREATE {self.index_name} "
                f"ON HASH PREFIX 1 cache: "
                f"SCHEMA "
                f"embedding VECTOR HNSW 6 "
                f"TYPE FLOAT32 "
                f"DIM {self.dimension} "
                f"DISTANCE_METRIC COSINE "
                f"response TEXT "
                f"timestamp NUMERIC "
                f"ttl NUMERIC "
                f"hit_count NUMERIC "
                f"last_accessed NUMERIC"
            )

            try:
                await self.redis.execute_command(create_command)
                logger.info(f"Created index {self.index_name} successfully")
            except ResponseError as e:
                if "Index already exists" in str(e):
                    logger.info(f"Index {self.index_name} already exists")
                else:
                    raise

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    async def store_batch(self, entries: List[Tuple[str, List[float], str, int, int]]):
        """Store multiple entries in pipeline"""
        if not entries:
            return
            
        pipeline = self.redis.pipeline()
        
        for query_hash, embedding, response, ttl, hit_count in entries:
            key = f"cache:{query_hash}"
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            pipeline.hset(
                key,
                mapping={
                    "embedding": embedding_bytes,
                    "response": response,
                    "timestamp": time.time(),
                    "ttl": ttl,
                    "hit_count": hit_count,
                    "last_accessed": time.time()
                }
            )
            pipeline.expire(key, ttl)
        
        await pipeline.execute()
    
    async def _batch_write_worker(self):
        """Background worker for batch writes"""
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                # Wait for item with timeout
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(), 
                        timeout=0.5
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Flush if batch is full or timeout reached
                if len(batch) >= self.pipeline_batch_size or \
                   (batch and time.time() - last_flush > 0.1):
                    await self.store_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                logger.error(f"Batch write error: {e}")
                await asyncio.sleep(1)
    
    async def store_with_embedding_async(
        self, 
        query_hash: str, 
        embedding: List[float], 
        response: str,
        ttl: int,
        hit_count: int = 0
    ):
        """Store asynchronously using queue"""
        await self._batch_queue.put((query_hash, embedding, response, ttl, hit_count))
    
    async def semantic_search(
        self, 
        query_embedding: List[float], 
        threshold: float,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for semantically similar queries with optimized KNN"""
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        try:
            # Correct syntax for Redis vector search
            # Using FT.SEARCH with KNN predicate
            results = await self.redis.execute_command(
                "FT.SEARCH",
                self.index_name,
                f"*=>[KNN {k} @embedding $vec AS vector_score]",
                "PARAMS",
                "2",
                "vec",
                embedding_bytes,
                "RETURN",
                "2",
                "response",
                "vector_score",
                "DIALECT",
                "4",
                "LIMIT",
                "0",
                str(k)
            )
            
            matches = []
            if results and len(results) > 0:
                # First element is the number of results
                num_results = results[0] if isinstance(results[0], int) else 0
                
                # Parse results
                idx = 1
                for _ in range(num_results):
                    if idx >= len(results):
                        break
                    
                    # Document ID (skip)
                    idx += 1
                    
                    # Fields
                    if idx >= len(results):
                        break
                    
                    fields = results[idx]
                    idx += 1
                    
                    response_text = ""
                    similarity = 0.0
                    
                    # Parse fields (they come as pairs)
                    if isinstance(fields, list):
                        for j in range(0, len(fields), 2):
                            if j + 1 < len(fields):
                                field_name = fields[j]
                                field_value = fields[j + 1]
                                
                                if field_name == "response":
                                    response_text = field_value
                                elif field_name == "vector_score":
                                    try:
                                        # Convert distance to similarity
                                        distance = float(field_value)
                                        similarity = 1 - distance
                                    except:
                                        pass
                    
                    # Only include if similarity meets threshold and we have a response
                    if similarity >= threshold and response_text:
                        matches.append((response_text, similarity))
            
            if matches:
                logger.info(f"Found {len(matches)} semantic matches, best similarity: {matches[0][1]:.3f}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def get_by_exact_match(self, query_hash: str) -> Optional[str]:
        """Get exact match by hash (fallback for exact matches)"""
        try:
            key = f"cache:{query_hash}"
            response = await self.redis.hget(key, "response")
            return response
        except Exception as e:
            logger.error(f"Exact match error: {e}")
            return None
    
    async def increment_hit_count(self, query_hash: str):
        """Atomically increment hit count for popularity tracking"""
        try:
            key = f"cache:{query_hash}"
            await self.redis.hincrby(key, "hit_count", 1)
            await self.redis.hset(key, "last_accessed", time.time())
        except Exception as e:
            logger.error(f"Failed to increment hit count: {e}")
    
    async def get_size(self) -> int:
        """Get total number of cached entries"""
        try:
            info = await self.redis.execute_command(f"FT.INFO {self.index_name}")
            # Parse INFO response
            for i, val in enumerate(info):
                if val == "num_docs":
                    return int(info[i + 1])
        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
        return 0
    
    async def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics"""
        try:
            info = await self.redis.execute_command(f"FT.INFO {self.index_name}")
            stats = {}
            for i, val in enumerate(info):
                if isinstance(val, str):
                    if i + 1 < len(info):
                        stats[val] = info[i + 1]
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    async def clear_all(self):
        """Clear all cache entries (for testing)"""
        try:
            # Get all keys with prefix
            keys = await self.redis.keys("cache:*")
            if keys:
                await self.redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def cleanup_expired(self):
        """Clean up expired entries"""
        # Redis handles TTL automatically, but we can log stats
        size = await self.get_size()
        logger.debug(f"Current cache size: {size} entries")
    
    async def close(self):
        """Close connections"""
        if self._batch_worker_task:
            self._batch_worker_task.cancel()
            try:
                await self._batch_worker_task
            except asyncio.CancelledError:
                pass
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")