import httpx
import asyncio
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from app.config import Settings
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for text using local Ollama with caching"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.model = settings.ollama_embedding_model
        self.cache = {}  # Simple embedding cache
        self.cache_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with local optimization"""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(
                    timeout=30.0,
                    connect=5.0
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=50,
                    max_connections=100
                ),
                http2=True
            )
        return self.client
    
    @retry(
        stop=stop_after_attempt(2),  # Fewer retries for local
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3)
    )
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for single text with aggressive caching"""
        # Check cache first
        if text in self.cache:
            self.cache_hits += 1
            logger.debug(f"Embedding cache hit: {text[:50]}...")
            return self.cache[text]
        
        self.cache_misses += 1
        client = await self._get_client()
        
        try:
            start_time = asyncio.get_event_loop().time()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embedding"]
            
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(f"Generated embedding in {elapsed:.0f}ms")
            
            # Cache with size management
            if len(self.cache) >= self.cache_size:
                # Remove 10% oldest (simple FIFO)
                remove_count = self.cache_size // 10
                for _ in range(remove_count):
                    if self.cache:
                        self.cache.pop(next(iter(self.cache)))
            
            self.cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with optimized batching"""
        # Check which ones are cached
        results = []
        to_generate = []
        
        for text in texts:
            if text in self.cache:
                results.append(self.cache[text])
            else:
                to_generate.append(text)
                results.append(None)  # Placeholder
        
        # Generate missing embeddings in parallel
        if to_generate:
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(10)
            
            async def generate_with_limit(text: str):
                async with semaphore:
                    return await self.generate(text)
            
            new_embeddings = await asyncio.gather(*[
                generate_with_limit(text) for text in to_generate
            ])
            
            # Fill in results
            new_idx = 0
            for i, result in enumerate(results):
                if result is None:
                    results[i] = new_embeddings[new_idx]
                    new_idx += 1
        
        return results
    
    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "max_cache_size": self.cache_size
        }
    
    async def close(self):
        """Close HTTP client"""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.debug("Embedding client closed")

class BatchEmbeddingGenerator:
    """Generate embeddings in batch for efficiency"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_gen = EmbeddingGenerator(settings)
        self.batch_size = settings.batch_embedding_size
        self.queue = asyncio.Queue()
        self.worker_task = None
        
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching"""
        # Process in batches for better performance
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = await self.embedding_gen.generate_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def close(self):
        """Cleanup"""
        await self.embedding_gen.close()