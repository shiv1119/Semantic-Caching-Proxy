import hashlib
import time
from typing import Optional, Tuple, List
from collections import defaultdict
import asyncio
from app.cache.redis_client import RedisVectorClient
from app.cache.ttl_strategies import TTLCalculator
from app.llm.embedding import EmbeddingGenerator, BatchEmbeddingGenerator
from app.utils.metrics import MetricsCollector
from app.config import Settings
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, redis_client: RedisVectorClient, settings: Settings, metrics: MetricsCollector):
        self.redis = redis_client
        self.settings = settings
        self.metrics = metrics
        self.embedding_gen = EmbeddingGenerator(settings)
        self.batch_embedding_gen = BatchEmbeddingGenerator(settings)
        self.ttl_calculator = TTLCalculator()
        self.hit_count_cache = defaultdict(int)
        
        # L1 cache for hot items (in-memory)
        self.local_cache = {}  # query_hash -> (response, timestamp)
        self.local_cache_size = 1000
        self.local_cache_ttl = 60  # 1 minute for L1 cache
        
        # Track cache statistics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.l1_hits = 0
        
    async def get(self, query: str) -> Optional[Tuple[str, float]]:
        """Retrieve cached response with multi-level caching"""
        start_time = time.time()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Level 1: Check exact match first (fastest path)
        exact_match = await self.redis.get_by_exact_match(query_hash)
        if exact_match:
            self.exact_hits += 1
            self.metrics.record_cache_hit("exact")
            await self.redis.increment_hit_count(query_hash)
            self.hit_count_cache[query_hash] += 1
            latency = (time.time() - start_time) * 1000
            logger.debug(f"Exact cache hit in {latency:.0f}ms")
            return exact_match, 1.0
        
        # Level 2: Check L1 cache (hot items in memory)
        if query_hash in self.local_cache:
            cached_item, timestamp = self.local_cache[query_hash]
            if time.time() - timestamp < self.local_cache_ttl:
                self.l1_hits += 1
                self.metrics.record_cache_hit("l1")
                await self.redis.increment_hit_count(query_hash)
                self.hit_count_cache[query_hash] += 1
                latency = (time.time() - start_time) * 1000
                logger.debug(f"L1 cache hit in {latency:.0f}ms")
                return cached_item, 1.0
        
        # Level 3: Generate embedding and search semantically
        embedding = await self.embedding_gen.generate(query)
        
        # Search in Redis for semantically similar queries
        matches = await self.redis.semantic_search(
            embedding, 
            self.settings.similarity_threshold,
            k=3
        )
        
        if matches:
            response, similarity = matches[0]  # Best match
            self.semantic_hits += 1
            
            # Update hit count for popularity tracking
            await self.redis.increment_hit_count(query_hash)
            self.hit_count_cache[query_hash] += 1
            
            # Add to L1 cache if this query becomes popular
            if self.hit_count_cache[query_hash] >= 5:  # Lower threshold for L1
                self.local_cache[query_hash] = (response, time.time())
                # Maintain L1 cache size (LRU-like eviction)
                if len(self.local_cache) > self.local_cache_size:
                    # Remove oldest 10% of entries
                    remove_count = self.local_cache_size // 10
                    oldest_keys = sorted(self.local_cache.keys(), 
                                       key=lambda k: self.local_cache[k][1])[:remove_count]
                    for key in oldest_keys:
                        del self.local_cache[key]
                logger.debug(f"Added query to L1 cache: {query[:50]}...")
            
            latency = (time.time() - start_time) * 1000
            self.metrics.record_cache_hit("redis", latency, similarity)
            
            logger.info(f"Semantic cache hit! Similarity: {similarity:.3f}, Latency: {latency:.0f}ms")
            return response, similarity
        
        # Cache miss
        self.metrics.record_cache_miss()
        logger.info(f"Cache miss for query: {query[:50]}...")
        return None
    
    async def set(self, query: str, response: str, query_hash: str = None):
        """Store response with intelligent TTL"""
        if not query_hash:
            query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Generate embedding for semantic search
        embedding = await self.embedding_gen.generate(query)
        
        # Calculate dynamic TTL based on various factors
        hit_count = self.hit_count_cache.get(query_hash, 0)
        base_ttl = self.settings.default_ttl
        
        ttl = self.ttl_calculator.calculate(
            hit_count=hit_count,
            query_length=len(query),
            base_ttl=base_ttl,
            time_of_day=time.localtime().tm_hour
        )
        
        # Store asynchronously (write-behind caching)
        await self.redis.store_with_embedding_async(
            query_hash, embedding, response, ttl, hit_count
        )
        
        # Update metrics
        self.metrics.record_cache_write()
        logger.debug(f"Stored query in cache with TTL {ttl}s: {query[:50]}...")
        
    async def get_or_set(self, query: str, llm_func) -> Tuple[str, bool, float]:
        """Get from cache or generate and store"""
        # Try to get from cache
        cached = await self.get(query)
        
        if cached:
            response, similarity = cached
            return response, True, similarity
        
        # Generate from LLM
        response = await llm_func(query)
        
        # Store in cache
        await self.set(query, response)
        
        return response, False, 0.0
        
    async def batch_set(self, queries: List[str], responses: List[str]):
        """Store multiple items in batch for efficiency"""
        # Generate embeddings for all queries in batch
        embeddings = await self.batch_embedding_gen.generate_batch(queries)
        
        entries = []
        for query, response, embedding in zip(queries, responses, embeddings):
            query_hash = hashlib.md5(query.encode()).hexdigest()
            ttl = self.ttl_calculator.calculate(
                hit_count=0,
                query_length=len(query),
                base_ttl=self.settings.default_ttl,
                time_of_day=time.localtime().tm_hour
            )
            entries.append((query_hash, embedding, response, ttl, 0))
        
        # Store all at once
        await self.redis.store_batch(entries)
        self.metrics.record_batch_write(len(entries))
        logger.info(f"Batch stored {len(entries)} items in cache")
    
    async def warmup_cache(self):
        """Pre-populate cache with common queries (async background task)"""
        warmup_queries = [
            "What is Python programming language?",
            "How to reverse a list in Python?",
            "Explain machine learning concepts",
            "What is an API and how does it work?",
            "How to sort an array efficiently?",
            "What is Docker container?",
            "Explain asynchronous programming",
            "What is Redis cache?",
            "How to use FastAPI framework?",
            "What is semantic caching?"
        ]
        
        logger.info(f"Starting cache warmup with {len(warmup_queries)} queries")
        # Run warmup in background to not block startup
        asyncio.create_task(self._warmup_worker(warmup_queries))
    
    async def _warmup_worker(self, queries: List[str]):
        """Background warmup worker that generates and caches responses"""
        try:
            # Import here to avoid circular imports
            from app.llm.ollama_client import OllamaClient
            
            llm_client = OllamaClient(self.settings)
            
            # Generate responses for warmup queries
            for query in queries:
                try:
                    response = await llm_client.generate(query)
                    await self.set(query, response)
                    logger.info(f"Warmed up cache for: {query[:50]}...")
                    await asyncio.sleep(0.5)  # Small delay to avoid overwhelming
                except Exception as e:
                    logger.error(f"Failed to warmup query '{query}': {e}")
            
            await llm_client.close()
            logger.info(f"Cache warmup complete! Added {len(queries)} entries")
        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")
    
    async def invalidate(self, query: str):
        """Invalidate a specific query from cache"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        key = f"cache:{query_hash}"
        
        # Remove from Redis
        await self.redis.redis.delete(key)
        
        # Remove from L1 cache if present
        if query_hash in self.local_cache:
            del self.local_cache[query_hash]
        
        # Reset hit count
        if query_hash in self.hit_count_cache:
            self.hit_count_cache[query_hash] = 0
            
        logger.info(f"Invalidated cache entry: {query[:50]}...")
    
    async def clear_all(self):
        """Clear all cache entries"""
        await self.redis.clear_all()
        self.local_cache.clear()
        self.hit_count_cache.clear()
        logger.info("Cleared all cache entries")
    
    async def get_detailed_stats(self) -> dict:
        """Get detailed cache statistics"""
        cache_size = await self.redis.get_size()
        redis_stats = await self.redis.get_cache_stats()
        
        total_requests = self.exact_hits + self.semantic_hits + self.l1_hits + self.metrics.cache_misses
        hit_rate = (self.exact_hits + self.semantic_hits + self.l1_hits) / max(total_requests, 1)
        
        return {
            "total_entries": cache_size,
            "l1_cache_size": len(self.local_cache),
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "l1_hits": self.l1_hits,
            "total_hits": self.exact_hits + self.semantic_hits + self.l1_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "popular_queries": dict(sorted(self.hit_count_cache.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]),
            "redis_stats": redis_stats,
            "metrics": self.metrics.get_summary(),
            "embedding_cache_stats": self.embedding_gen.get_cache_stats()
        }
    
    async def get_hit_rate(self) -> float:
        """Get current cache hit rate"""
        total_requests = self.exact_hits + self.semantic_hits + self.l1_hits + self.metrics.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits + self.l1_hits) / total_requests