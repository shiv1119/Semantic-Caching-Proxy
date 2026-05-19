from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import time
import hashlib
import asyncio
from typing import Optional
from app.models import ChatRequest, ChatResponse
from app.llm.ollama_client import OllamaClient
from app.middleware.circuit_breaker import CircuitBreakerRegistry
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["proxy"])
circuit_breaker_registry = CircuitBreakerRegistry()

class ProxyHandler:
    """Handle all proxy operations with optimized async patterns"""
    
    def __init__(self, app_state):
        self.app_state = app_state
        self.llm_client = OllamaClient(app_state.semantic_cache.settings)
        
    async def handle_chat_request(self, request: Request, chat_request: ChatRequest):
        """Handle chat completion request with semantic caching"""
        start_time = time.time()
        
        # Extract prompt
        prompt = chat_request.prompt
        
        # Check semantic cache
        cached_response = await self.app_state.semantic_cache.get(prompt)
        
        if cached_response:
            response_text, similarity = cached_response
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.app_state.metrics.record_request(
                cached=True,
                latency_ms=latency_ms,
                similarity=similarity
            )
            
            return ChatResponse(
                response=response_text,
                cached=True,
                similarity_score=similarity,
                latency_ms=latency_ms,
                tokens_saved=len(prompt.split()) + len(response_text.split()),
                cost_saved_usd=self._calculate_cost_saved(len(prompt) + len(response_text))
            )
        
        # Cache miss - get circuit breaker for LLM
        circuit_breaker = await circuit_breaker_registry.get("ollama")
        
        try:
            # Call LLM with circuit breaker protection
            response_text = await circuit_breaker.call(
                self.llm_client.generate,
                prompt,
                chat_request.temperature,
                chat_request.max_tokens
            )
            
            # Store in cache asynchronously (don't wait)
            asyncio.create_task(
                self.app_state.semantic_cache.set(prompt, response_text)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.app_state.metrics.record_request(
                cached=False,
                latency_ms=latency_ms
            )
            
            return ChatResponse(
                response=response_text,
                cached=False,
                similarity_score=None,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            self.app_state.metrics.record_error()
            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable: {str(e)}"
            )
    
    def _calculate_cost_saved(self, tokens: int) -> float:
        """Calculate estimated cost saved"""
        # Assume $0.001 per 1000 tokens
        return (tokens / 1000) * 0.001

@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Main endpoint for chat completions with semantic caching"""
    proxy_handler = ProxyHandler(request.app.state)
    return await proxy_handler.handle_chat_request(request, chat_request)

@router.post("/v1/batch/completions")
async def batch_completion(
    request: Request,
    prompts: list[str],
    background_tasks: BackgroundTasks
):
    """Batch endpoint for multiple prompts"""
    start_time = time.time()
    
    # Process all prompts
    results = []
    for prompt in prompts:
        cached = await request.app.state.semantic_cache.get(prompt)
        if cached:
            results.append({"prompt": prompt, "response": cached[0], "cached": True})
        else:
            # Queue for LLM processing
            results.append({"prompt": prompt, "cached": False})
    
    # Process non-cached in batch
    non_cached = [r for r in results if not r["cached"]]
    if non_cached:
        llm_client = OllamaClient(request.app.state.semantic_cache.settings)
        prompts_to_generate = [r["prompt"] for r in non_cached]
        responses = await llm_client.generate_batch(prompts_to_generate)
        
        # Store responses
        for i, result in enumerate(non_cached):
            result["response"] = responses[i]
            asyncio.create_task(
                request.app.state.semantic_cache.set(result["prompt"], responses[i])
            )
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "results": results,
        "total_latency_ms": latency_ms,
        "cached_count": len([r for r in results if r.get("cached")]),
        "total_count": len(results)
    }