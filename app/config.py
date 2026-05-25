from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    worker_connections: int = 1000
    
    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_pool_size: int = 50
    redis_vector_index: str = "semantic_cache:index"
    redis_vector_dimension: int = 768  # must match the embedding model output size
    redis_max_retries: int = 3
    
    # --- Cache strategy ---
    # 0.85 is stricter than the 0.75 used in vector search — change carefully,
    # lower values increase hit rate but risk returning wrong cached answers
    similarity_threshold: float = 0.85
    default_ttl: int = 3600           # 1 hour baseline, adjusted dynamically by TTLCalculator
    semantic_ttl_enabled: bool = True
    cache_max_size: int = 50000
    cache_warmup_enabled: bool = True
    cache_write_behind: bool = True   # async writes — see _batch_write_worker in redis_client
    cache_async_write_queue_size: int = 2000
    
    # --- Rate limiting ---
    rate_limit_per_minute: int = 10000
    rate_limit_burst: int = 1000      # max requests allowed in a short spike above the sustained rate
    rate_limit_by_ip: bool = True
    rate_limit_by_api_key: bool = True
    
    # --- Ollama ---
    # Defaults assume Ollama is running locally. In Docker, this becomes
    # http://host.docker.internal:11434 via the docker-compose env override.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:latest"
    ollama_embedding_model: str = "nomic-embed-text"  # produces 768-dim vectors
    ollama_timeout: int = 60          # local models can be slow on CPU — keep this generous
    ollama_max_retries: int = 3
    ollama_batch_size: int = 32
    ollama_connection_timeout: int = 10
    ollama_max_connections: int = 50
    
    # --- LLM fallback ---
    # Optional secondary LLM if Ollama is completely unavailable.
    # Disabled by default — enable and configure for production resilience.
    llm_fallback_enabled: bool = False
    llm_fallback_url: Optional[str] = None
    llm_fallback_api_key: Optional[str] = None
    
    # --- Circuit breaker ---
    # Tuned for a local LLM: lower failure threshold and faster recovery
    # than you'd use for a remote API, since local failures tend to be brief.
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout: int = 30  # seconds before moving to HALF_OPEN
    circuit_half_open_max_calls: int = 2
    
    # --- Connection pooling ---
    max_concurrent_requests: int = 500
    connection_pool_size: int = 100
    connection_pool_overflow: int = 50  # extra connections allowed above pool_size under load
    connection_timeout: int = 30
    keepalive_timeout: int = 5
    
    # --- Logging & monitoring ---
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090          # Prometheus scrape target
    prometheus_enabled: bool = True
    
    # --- Security ---
    api_key_required: bool = False    # set to True and provide api_key in production
    api_key: Optional[str] = None
    allowed_origins: str = "*"        # tighten this in production
    max_request_size: int = 1000000   # 1MB — guards against oversized prompt payloads
    
    # --- Performance ---
    enable_response_compression: bool = True
    batch_embedding_size: int = 32    # how many texts to embed in one Ollama call
    prefetch_cache_enabled: bool = True
    enable_semantic_deduplication: bool = True
    
    # --- Advanced features ---
    enable_ttl_adaptation: bool = True      # lets TTLCalculator adjust TTLs dynamically
    enable_popularity_tracking: bool = True  # tracks hit counts for L1 promotion logic
    enable_cost_tracking: bool = True
    estimated_monthly_requests: int = 1000000
    
    # --- Cluster ---
    # cluster_mode enables multi-replica behaviour — each node needs a unique node_id
    cluster_mode: bool = False
    node_id: str = "node-1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # silently drops unknown env vars rather than erroring


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    lru_cache ensures we only read and parse the .env file once per process —
    safe to call anywhere without worrying about repeated disk reads.
    """
    return Settings()