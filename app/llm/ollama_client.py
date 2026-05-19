import httpx
import asyncio
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import logging
from app.config import Settings
import socket
import aiohttp

logger = logging.getLogger(__name__)

class OllamaClient:
    """Async client for local Ollama API with optimized connection pooling"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.model = settings.ollama_model
        self.timeout = httpx.Timeout(
            timeout=settings.ollama_timeout,
            connect=settings.ollama_connection_timeout
        )
        self.base_url = settings.ollama_base_url
        self._check_local_ollama()
        
    def _check_local_ollama(self):
        """Check if local Ollama is accessible"""
        try:
            # Extract host and port from URL
            url = self.base_url.replace('http://', '')
            host, port = url.split(':')
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            
            if result == 0:
                logger.info(f"Local Ollama accessible at {self.base_url}")
            else:
                logger.warning(f"Local Ollama not accessible at {self.base_url}")
        except Exception as e:
            logger.warning(f"Could not check Ollama connectivity: {e}")
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with optimized connection pooling for local"""
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=50,
                    max_connections=100,
                    keepalive_expiry=30
                ),
                http2=True,
                follow_redirects=True
            )
        return self.client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),  # Faster retries for local
        retry=retry_if_exception(lambda e: isinstance(e, (httpx.TimeoutException, httpx.ConnectError)))
    )
    async def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str:
        """Generate response from local Ollama with optimized settings"""
        client = await self._get_client()
        
        try:
            # Optimized payload for local Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "options": {
                    "num_ctx": 2048,  # Context window
                    "num_predict": max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            start_time = asyncio.get_event_loop().time()
            response = await client.post(
                "/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(f"Ollama response in {elapsed:.0f}ms")
            
            data = response.json()
            return data.get("response", "")
            
        except httpx.TimeoutException:
            logger.error(f"Ollama timeout after {self.settings.ollama_timeout}s")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error {e.response.status_code}: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    @retry(stop=stop_after_attempt(2))
    async def generate_batch(
        self, 
        prompts: List[str],
        temperature: float = 0.7,
        max_concurrent: int = 5  # Lower concurrency for local to avoid overloading
    ) -> List[str]:
        """Generate responses for multiple prompts concurrently with semaphore"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, temperature)
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        results = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Batch generation error for prompt {i}: {resp}")
                results.append("Error generating response")
            else:
                results.append(resp)
        
        return results
    
    async def check_model_availability(self) -> bool:
        """Check if the model is available in local Ollama"""
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            available = self.model in models
            if not available:
                logger.warning(f"Model {self.model} not found. Available: {models}")
            return available
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            logger.debug("Ollama client closed")