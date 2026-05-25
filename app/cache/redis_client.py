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
        """
        Establishes a connection pool with keepalive and retry support.
        Also creates the vector index and starts the background write worker if write-behind is enabled.
        """
        self.redis = await redis.from_url(
            self.redis_url,
            decode_responses=True,
            max_connections=self.settings.redis_pool_size,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError]
        )
        
        await self.redis.ping()
        await self._create_index()
        
        # Write-behind mode batches cache writes asynchronously to avoid
        # blocking the main request path on every cache store operation.
        if self.settings.cache_write_behind:
            self._batch_worker_task = asyncio.create_task(self._batch_write_worker())
        
        logger.info(f"Redis client connected to {self.redis_url}")
    
    async def _create_index(self):
        """
        Creates the HNSW vector index in Redis Stack.

        We drop and recreate on startup to avoid stale index configs across deployments.
        HNSW is used over flat search because it gives O(log n) lookup vs O(n),
        which matters as the cache grows.
        """
        try:
            try:
                await self.redis.execute_command(f"FT.DROPINDEX {self.index_name}")
                logger.info(f"Dropped existing index {self.index_name}")
            except ResponseError as e:
                if "Unknown index name" not in str(e):
                    logger.warning(f"Drop index warning: {e}")

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
        """
        Writes multiple cache entries in a single Redis pipeline.
        Batching reduces round-trips significantly under concurrent load.
        """
        if not entries:
            return
            
        pipeline = self.redis.pipeline()
        
        for query_hash, embedding, response, ttl, hit_count in entries:
            key = f"cache:{query_hash}"
            # Redis vector search requires raw bytes in FLOAT32 format
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
        """
        Background task that drains the write queue in batches.

        Flushes when the batch hits pipeline_batch_size OR after 100ms of inactivity —
        whichever comes first. The timeout prevents entries from sitting in the queue
        too long during low traffic periods.
        """
        batch = []
        last_flush = time.time()
        
        while True:
            try:
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(), 
                        timeout=0.5
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
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
        """Enqueue a cache write — actual storage happens in the background worker."""
        await self._batch_queue.put((query_hash, embedding, response, ttl, hit_count))
    
    async def semantic_search(
        self, 
        query_embedding: List[float], 
        threshold: float,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        KNN vector search using Redis HNSW index.

        Returns matches above the similarity threshold, sorted by best match first.
        Redis returns cosine DISTANCE (0 = identical, 2 = opposite), so we convert
        to similarity with: similarity = 1 - distance.

        Threshold of 0.75 was chosen empirically — low enough to catch paraphrases,
        high enough to avoid returning wrong cached answers.
        """
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        try:
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
                num_results = results[0] if isinstance(results[0], int) else 0
                
                idx = 1
                for _ in range(num_results):
                    if idx >= len(results):
                        break
                    
                    idx += 1  # skip document ID
                    
                    if idx >= len(results):
                        break
                    
                    fields = results[idx]
                    idx += 1
                    
                    response_text = ""
                    similarity = 0.0
                    
                    if isinstance(fields, list):
                        for j in range(0, len(fields), 2):
                            if j + 1 < len(fields):
                                field_name = fields[j]
                                field_value = fields[j + 1]
                                
                                if field_name == "response":
                                    response_text = field_value
                                elif field_name == "vector_score":
                                    try:
                                        distance = float(field_value)
                                        similarity = 1 - distance  # convert distance → similarity
                                    except:
                                        pass
                    
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
        """
        Tier-1 cache lookup by exact query hash before falling back to vector search.
        Faster and cheaper than HNSW for identical repeated queries.
        """
        try:
            key = f"cache:{query_hash}"
            response = await self.redis.hget(key, "response")
            return response
        except Exception as e:
            logger.error(f"Exact match error: {e}")
            return None
    
    async def increment_hit_count(self, query_hash: str):
        """Tracks how often a cached entry is served — useful for cache analytics and eviction tuning."""
        try:
            key = f"cache:{query_hash}"
            await self.redis.hincrby(key, "hit_count", 1)
            await self.redis.hset(key, "last_accessed", time.time())
        except Exception as e:
            logger.error(f"Failed to increment hit count: {e}")
    
    async def get_size(self) -> int:
        """Returns the number of documents currently indexed in the vector store."""
        try:
            info = await self.redis.execute_command(f"FT.INFO {self.index_name}")
            for i, val in enumerate(info):
                if val == "num_docs":
                    return int(info[i + 1])
        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
        return 0
    
    async def get_cache_stats(self) -> Dict:
        """Returns raw FT.INFO stats — used by the /metrics endpoint."""
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
        """Wipes all cache entries. Intended for testing — not safe to call in production."""
        try:
            keys = await self.redis.keys("cache:*")
            if keys:
                await self.redis.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    async def cleanup_expired(self):
        """
        Redis handles TTL expiry automatically, so no manual cleanup is needed.
        This exists as a hook for logging cache size over time.
        """
        size = await self.get_size()
        logger.debug(f"Current cache size: {size} entries")
    
    async def close(self):
        """Gracefully shuts down the batch worker and closes the Redis connection."""
        if self._batch_worker_task:
            self._batch_worker_task.cancel()
            try:
                await self._batch_worker_task
            except asyncio.CancelledError:
                pass
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")