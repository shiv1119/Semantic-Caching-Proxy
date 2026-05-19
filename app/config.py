from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    worker_connections: int = 1000
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_pool_size: int = 50
    redis_vector_index: str = "semantic_cache:index"
    redis_vector_dimension: int = 768
    redis_max_retries: int = 3
    
    # Cache Strategy
    similarity_threshold: float = 0.85
    default_ttl: int = 3600
    semantic_ttl_enabled: bool = True
    cache_max_size: int = 50000
    cache_warmup_enabled: bool = True
    cache_write_behind: bool = True
    cache_async_write_queue_size: int = 2000
    
    # Rate Limiting
    rate_limit_per_minute: int = 10000
    rate_limit_burst: int = 1000
    rate_limit_by_ip: bool = True
    rate_limit_by_api_key: bool = True
    
    # Ollama Configuration - Modified for local access
    ollama_base_url: str = "http://localhost:11434"  # Local Ollama
    ollama_model: str = "gemma3:latest"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout: int = 60  # Increased timeout for local
    ollama_max_retries: int = 3
    ollama_batch_size: int = 32
    ollama_connection_timeout: int = 10
    ollama_max_connections: int = 50
    
    # LLM Fallback
    llm_fallback_enabled: bool = False
    llm_fallback_url: Optional[str] = None
    llm_fallback_api_key: Optional[str] = None
    
    # Circuit Breaker - Adjusted for local LLM
    circuit_failure_threshold: int = 3  # Lower threshold for local
    circuit_recovery_timeout: int = 30  # Faster recovery
    circuit_half_open_max_calls: int = 2
    
    # Connection Pooling
    max_concurrent_requests: int = 500
    connection_pool_size: int = 100
    connection_pool_overflow: int = 50
    connection_timeout: int = 30
    keepalive_timeout: int = 5
    
    # Logging & Monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    prometheus_enabled: bool = True
    
    # Security
    api_key_required: bool = False
    api_key: Optional[str] = None
    allowed_origins: str = "*"
    max_request_size: int = 1000000  # 1MB
    
    # Performance Tuning
    enable_response_compression: bool = True
    batch_embedding_size: int = 32
    prefetch_cache_enabled: bool = True
    enable_semantic_deduplication: bool = True
    
    # Advanced Features
    enable_ttl_adaptation: bool = True
    enable_popularity_tracking: bool = True
    enable_cost_tracking: bool = True
    estimated_monthly_requests: int = 1000000
    
    # Cluster Configuration
    cluster_mode: bool = False
    node_id: str = "node-1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars

@lru_cache()
def get_settings() -> Settings:
    return Settings()