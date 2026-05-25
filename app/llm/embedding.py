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
    """
    Calls the local Ollama API to convert text into vector embeddings.
    Keeps an in-memory cache so we never re-embed the same text twice
    within the same process lifetime — embeddings are deterministic,
    so there's no reason to recompute them.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.model = settings.ollama_embedding_model
        self.cache = {}  # text -> embedding vector
        self.cache_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def _get_client(self) -> httpx.AsyncClient:
        """
        Lazily creates the HTTP client on first use and reuses it across calls.
        HTTP/2 is enabled to multiplex concurrent embedding requests over a
        single connection — helps a lot under batch load.
        """
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
        stop=stop_after_attempt(2),   # only 2 retries — Ollama is local so failures are usually real
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3)
    )
    async def generate(self, text: str) -> List[float]:
        """
        Returns the embedding vector for a single piece of text.
        Hits the in-memory cache first — if it's a miss, calls Ollama
        and stores the result before returning.

        Cache eviction is simple FIFO: when we hit the size limit,
        we drop the oldest 10% of entries. Not perfect LRU, but cheap
        and good enough for an embedding cache.
        """
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
            
            # Evict oldest 10% when cache is full.
            # dict preserves insertion order in Python 3.7+, so iter() gives us the oldest key.
            if len(self.cache) >= self.cache_size:
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
        """
        Generates embeddings for a list of texts, skipping any that are
        already in cache. The uncached ones are fired off in parallel,
        limited to 10 concurrent Ollama requests via a semaphore —
        enough to keep throughput high without overwhelming the local model.

        Results are stitched back into the original order using None placeholders.
        """
        results = []
        to_generate = []
        
        for text in texts:
            if text in self.cache:
                results.append(self.cache[text])
            else:
                to_generate.append(text)
                results.append(None)  # placeholder — filled in below
        
        if to_generate:
            # Cap concurrency at 10 so we don't flood Ollama with simultaneous requests
            semaphore = asyncio.Semaphore(10)
            
            async def generate_with_limit(text: str):
                async with semaphore:
                    return await self.generate(text)
            
            new_embeddings = await asyncio.gather(*[
                generate_with_limit(text) for text in to_generate
            ])
            
            # Fill placeholders with the newly generated embeddings, in order
            new_idx = 0
            for i, result in enumerate(results):
                if result is None:
                    results[i] = new_embeddings[new_idx]
                    new_idx += 1
        
        return results
    
    def get_cache_stats(self) -> dict:
        """Returns hit rate and size info — exposed via the /metrics endpoint."""
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
        """Closes the underlying HTTP client. Call this on app shutdown."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.debug("Embedding client closed")


class BatchEmbeddingGenerator:
    """
    Thin wrapper around EmbeddingGenerator that splits large lists into
    fixed-size chunks before generating. Useful for bulk operations like
    cache warmup where sending hundreds of texts at once would be unwieldy.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_gen = EmbeddingGenerator(settings)
        self.batch_size = settings.batch_embedding_size
        self.queue = asyncio.Queue()
        self.worker_task = None
        
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Splits texts into chunks of batch_size and processes each chunk
        through EmbeddingGenerator.generate_batch. Results are concatenated
        back into a single flat list in the original order.
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = await self.embedding_gen.generate_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def close(self):
        """Propagates shutdown to the underlying EmbeddingGenerator."""
        await self.embedding_gen.close()