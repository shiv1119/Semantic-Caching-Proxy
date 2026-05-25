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
# Single registry instance shared across all requests — ensures the same
# circuit breaker state is reused rather than resetting on every call
circuit_breaker_registry = CircuitBreakerRegistry()

class ProxyHandler:
    """
    Handles the core request flow: check semantic cache → call LLM on miss →
    store result asynchronously → return response.
    Separated from the route functions so the logic is testable independently
    of FastAPI's request/response cycle.
    """
    
    def __init__(self, app_state):
        self.app_state = app_state
        self.llm_client = OllamaClient(app_state.semantic_cache.settings)
        
    async def handle_chat_request(self, request: Request, chat_request: ChatRequest):
        """
        Main request handler. Cache hit path is fast — no LLM call at all.
        On a miss, the circuit breaker wraps the Ollama call to prevent
        cascading failures if the model is slow or unavailable.
        Cache writes after an LLM response are fire-and-forget (create_task)
        so they don't add latency to the response the caller is waiting for.
        """
        start_time = time.time()
        prompt = chat_request.prompt
        
        cached_response = await self.app_state.semantic_cache.get(prompt)
        
        if cached_response:
            response_text, similarity = cached_response
            latency_ms = (time.time() - start_time) * 1000
            
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
                # tokens_saved and cost_saved give callers visibility into
                # how much value the cache provided on this request
                tokens_saved=len(prompt.split()) + len(response_text.split()),
                cost_saved_usd=self._calculate_cost_saved(len(prompt) + len(response_text))
            )
        
        # Cache miss — go to the LLM, but wrap it with the circuit breaker
        # so repeated Ollama failures trip the breaker and fast-fail instead of queuing
        circuit_breaker = await circuit_breaker_registry.get("ollama")
        
        try:
            response_text = await circuit_breaker.call(
                self.llm_client.generate,
                prompt,
                chat_request.temperature,
                chat_request.max_tokens
            )
            
            # Fire-and-forget cache write — we don't await this because
            # the caller shouldn't have to wait for storage to complete
            asyncio.create_task(
                self.app_state.semantic_cache.set(prompt, response_text)
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
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
            # 503 rather than 500 — the service itself is fine,
            # but the upstream LLM is unavailable
            raise HTTPException(
                status_code=503,
                detail=f"LLM service unavailable: {str(e)}"
            )
    
    def _calculate_cost_saved(self, tokens: int) -> float:
        """
        Rough estimate of API cost avoided by serving from cache.
        Based on $0.001 per 1000 tokens — a conservative approximation.
        Not meant to be exact, just gives callers a useful ballpark figure.
        """
        return (tokens / 1000) * 0.001


@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(
    request: Request,
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Single prompt endpoint. Delegates entirely to ProxyHandler so the
    cache-check and LLM-call logic stays in one testable place.
    """
    proxy_handler = ProxyHandler(request.app.state)
    return await proxy_handler.handle_chat_request(request, chat_request)


@router.post("/v1/batch/completions")
async def batch_completion(
    request: Request,
    prompts: list[str],
    background_tasks: BackgroundTasks
):
    """
    Batch endpoint for processing multiple prompts in one request.
    Strategy: check all prompts against cache first, then send only the
    uncached ones to the LLM in a single batch call — much cheaper than
    N individual LLM requests. Cache writes for new responses are
    fire-and-forget, same as the single-prompt endpoint.
    """
    start_time = time.time()
    
    results = []
    for prompt in prompts:
        cached = await request.app.state.semantic_cache.get(prompt)
        if cached:
            results.append({"prompt": prompt, "response": cached[0], "cached": True})
        else:
            results.append({"prompt": prompt, "cached": False})
    
    # Only hit the LLM for prompts that weren't in cache
    non_cached = [r for r in results if not r["cached"]]
    if non_cached:
        llm_client = OllamaClient(request.app.state.semantic_cache.settings)
        prompts_to_generate = [r["prompt"] for r in non_cached]
        responses = await llm_client.generate_batch(prompts_to_generate)
        
        for i, result in enumerate(non_cached):
            result["response"] = responses[i]
            # Store each new response asynchronously — don't block the batch response
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