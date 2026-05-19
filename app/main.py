import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
import httpx
import logging
from app.config import Settings
from app.cache.redis_client import RedisVectorClient
from app.cache.semantic_cache import SemanticCache
from app.routes.proxy import router as proxy_router
from app.middleware.rate_limiter import DistributedRateLimiter, RateLimitMiddleware
from app.middleware.request_logger import LoggingMiddleware
from app.utils.metrics import MetricsCollector
from app.utils.pool_manager import ConnectionPoolManager
from app.llm.ollama_client import OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with optimized connection pooling"""
    logger.info("Starting Semantic Caching Proxy with Local Ollama...")
    
    # Verify local Ollama is accessible
    test_client = OllamaClient(settings)
    model_available = await test_client.check_model_availability()
    if not model_available:
        logger.warning(f"Model {settings.ollama_model} not available in local Ollama")
        logger.info(f"Please run: ollama pull {settings.ollama_model}")
        logger.info(f"and: ollama pull {settings.ollama_embedding_model}")
    else:
        logger.info(f"✓ Local Ollama model {settings.ollama_model} is available")
    await test_client.close()
    
    # Initialize connection pools
    app.state.pool_manager = ConnectionPoolManager(settings)
    await app.state.pool_manager.initialize_all()
    
    # Initialize Redis client
    redis_client = RedisVectorClient(
        settings.redis_url,
        settings.redis_vector_index,
        settings
    )
    await redis_client.connect()
    app.state.redis_client = redis_client
    logger.info("✓ Redis connected")
    
    # Initialize rate limiter
    rate_limiter = DistributedRateLimiter(
        redis_client.redis, 
        settings.rate_limit_per_minute,
        settings.rate_limit_burst
    )
    app.state.rate_limiter = rate_limiter
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    app.state.metrics = metrics
    
    # Initialize semantic cache
    semantic_cache = SemanticCache(
        redis_client, 
        settings,
        metrics
    )
    app.state.semantic_cache = semantic_cache
    
    # Warm up cache if enabled
    if settings.cache_warmup_enabled:
        await semantic_cache.warmup_cache()
    
    logger.info(f"✓ Server ready on {settings.host}:{settings.port} with {settings.workers} workers")
    logger.info(f"✓ Local Ollama endpoint: {settings.ollama_base_url}")
    logger.info(f"✓ Model: {settings.ollama_model}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    await app.state.pool_manager.close_all()
    await redis_client.close()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Semantic Caching Proxy",
    description="High-performance semantic caching layer for local LLM APIs",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(proxy_router, prefix="/api")

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with Ollama status"""
    # Check Ollama status
    ollama_client = OllamaClient(settings)
    model_available = await ollama_client.check_model_availability()
    await ollama_client.close()
    
    return {
        "status": "healthy",
        "cache_size": await request.app.state.redis_client.get_size(),
        "metrics": request.app.state.metrics.get_summary(),
        "ollama": {
            "available": model_available,
            "url": settings.ollama_base_url,
            "model": settings.ollama_model
        }
    }

@app.get("/metrics")
async def get_metrics(request: Request):
    """Prometheus metrics endpoint"""
    return request.app.state.metrics.get_prometheus_metrics()

@app.get("/cache/stats")
async def cache_stats(request: Request):
    """Detailed cache statistics"""
    return await request.app.state.semantic_cache.get_detailed_stats()

@app.get("/ollama/status")
async def ollama_status():
    """Check Ollama status and available models"""
    client = OllamaClient(settings)
    try:
        # Check if Ollama is running
        test_client = httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=5.0)
        response = await test_client.get("/api/tags")
        await test_client.aclose()
        
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return {
                "status": "running",
                "url": settings.ollama_base_url,
                "models": models,
                "default_model": settings.ollama_model,
                "embedding_model": settings.ollama_embedding_model,
                "model_available": settings.ollama_model in models,
                "embedding_available": settings.ollama_embedding_model in models
            }
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        await client.close()