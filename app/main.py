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

# setting up logs so we can see what's happening when the app runs
# the format shows time, which part of the code logged it, and the message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs when the server starts up and when it shuts down.
    Think of it like the "opening and closing shift" of the app —
    we set everything up before requests come in, and clean up after we're done.
    """
    logger.info("Starting Semantic Caching Proxy with Local Ollama...")
    
    # first, let's make sure Ollama is actually running on this machine
    # no point starting if the AI model isn't even available
    test_client = OllamaClient(settings)
    model_available = await test_client.check_model_availability()
    if not model_available:
        # warn the user and tell them exactly what command to run to fix it
        logger.warning(f"Model {settings.ollama_model} not available in local Ollama")
        logger.info(f"Please run: ollama pull {settings.ollama_model}")
        logger.info(f"and: ollama pull {settings.ollama_embedding_model}")
    else:
        logger.info(f"✓ Local Ollama model {settings.ollama_model} is available")
    await test_client.close()
    
    # connection pools let us reuse existing connections instead of opening
    # a new one every single request — much faster under load
    app.state.pool_manager = ConnectionPoolManager(settings)
    await app.state.pool_manager.initialize_all()
    
    # connect to Redis — this is where we store cached responses
    # Redis is basically a super fast in-memory key-value store
    redis_client = RedisVectorClient(
        settings.redis_url,
        settings.redis_vector_index,
        settings
    )
    await redis_client.connect()
    app.state.redis_client = redis_client
    logger.info("✓ Redis connected")
    
    # rate limiter makes sure one user can't spam the API and slow it down for everyone
    # it's stored in Redis so it works across multiple server instances too
    rate_limiter = DistributedRateLimiter(
        redis_client.redis, 
        settings.rate_limit_per_minute,
        settings.rate_limit_burst
    )
    app.state.rate_limiter = rate_limiter
    
    # metrics collector keeps track of things like cache hit rate, response times etc.
    # useful for knowing if the cache is actually helping
    metrics = MetricsCollector()
    app.state.metrics = metrics
    
    # the semantic cache is the main feature here — instead of exact string matching,
    # it finds responses to questions that *mean* the same thing even if worded differently
    semantic_cache = SemanticCache(
        redis_client, 
        settings,
        metrics
    )
    app.state.semantic_cache = semantic_cache
    
    # optionally pre-fill the cache with common queries so the first users
    # don't have to wait for cold cache responses
    if settings.cache_warmup_enabled:
        await semantic_cache.warmup_cache()
    
    logger.info(f"✓ Server ready on {settings.host}:{settings.port} with {settings.workers} workers")
    logger.info(f"✓ Local Ollama endpoint: {settings.ollama_base_url}")
    logger.info(f"✓ Model: {settings.ollama_model}")
    
    yield  # everything above runs on startup, everything below runs on shutdown
    
    # cleanup time — close all open connections properly
    # skipping this can cause weird errors or resource leaks
    logger.info("Shutting down...")
    await app.state.pool_manager.close_all()
    await redis_client.close()
    logger.info("Shutdown complete")

# spinning up the actual FastAPI app with our lifespan handler attached
app = FastAPI(
    title="Semantic Caching Proxy",
    description="High-performance semantic caching layer for local LLM APIs",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse  # faster JSON serialization than the default
)

# CORS lets browsers on other domains call our API
# in production you'd want to lock down allowed_origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# these two run on every single request, in this order
app.add_middleware(RateLimitMiddleware)   # block if they're sending too many requests
app.add_middleware(LoggingMiddleware)     # log the request for debugging/monitoring

# attach our routes — all proxy routes will be under /api/...
app.include_router(proxy_router, prefix="/api")

@app.get("/health")
async def health_check(request: Request):
    """
    Quick way to check if the server is alive and Ollama is reachable.
    Load balancers and monitoring tools usually ping this endpoint.
    """
    # check if the Ollama model is still up — it might have gone down after startup
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
    """Exposes stats in Prometheus format — plug this into Grafana for nice dashboards"""
    return request.app.state.metrics.get_prometheus_metrics()

@app.get("/cache/stats")
async def cache_stats(request: Request):
    """Breakdown of how the cache is doing — hit rate, size, evictions, that kind of thing"""
    return await request.app.state.semantic_cache.get_detailed_stats()

@app.get("/ollama/status")
async def ollama_status():
    """
    Checks what models are currently loaded in Ollama.
    Handy for debugging when you're not sure if the right model got pulled.
    """
    client = OllamaClient(settings)
    try:
        # hit Ollama's tags endpoint — it lists all downloaded models
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
                # these two tell you if the models we actually need are present
                "model_available": settings.ollama_model in models,
                "embedding_available": settings.ollama_embedding_model in models
            }
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        # if we can't even reach Ollama, surface the error message directly
        return {"status": "error", "message": str(e)}
    finally:
        await client.close()