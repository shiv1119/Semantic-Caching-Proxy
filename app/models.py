from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class RequestType(str, Enum):
    CHAT_COMPLETION = "chat_completion"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

class CacheEntry(BaseModel):
    query_hash: str
    query_text: str
    embedding: List[float]
    response: str
    timestamp: float
    ttl: int
    hit_count: int = 0
    last_accessed: float = 0.0

class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False
    
class ChatResponse(BaseModel):
    response: str
    cached: bool
    similarity_score: Optional[float] = None
    latency_ms: float
    tokens_saved: Optional[int] = None
    cost_saved_usd: Optional[float] = None

class CacheStats(BaseModel):
    total_entries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_latency_cache_hit_ms: float
    avg_latency_cache_miss_ms: float
    estimated_cost_saved: float
    memory_usage_mb: float
    ttl_distribution: Dict[str, int]

class RateLimitInfo(BaseModel):
    client_id: str
    allowed: bool
    remaining: int
    retry_after: Optional[int] = None
    reset_at: datetime

class MetricsSnapshot(BaseModel):
    timestamp: datetime
    requests_per_second: float
    cache_hit_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    concurrent_requests: int