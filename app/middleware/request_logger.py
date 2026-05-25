from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Dict
import json

logger = logging.getLogger("api")

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request as a structured JSON line after it completes.
    Structured logging (vs plain strings) makes it easy to filter and
    aggregate in tools like Grafana Loki or CloudWatch — you can query
    by status_code, path, or cache_hit directly.

    Log level is chosen based on status code:
        5xx → error, 4xx → warning, everything else → info
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # These fields are set by the route handler when a cache lookup happens.
        # Not present on every request (e.g. health checks), so we check first.
        if hasattr(request.state, "cache_hit"):
            log_data["cache_hit"] = request.state.cache_hit
        if hasattr(request.state, "similarity_score"):
            log_data["similarity_score"] = request.state.similarity_score
        
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
        
        # Expose response time as a header so clients and load balancers
        # can observe latency without parsing logs
        response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))
        
        return response