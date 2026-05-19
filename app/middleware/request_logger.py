from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Dict
import json

logger = logging.getLogger("api")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing information"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Add cache hit info if available
        if hasattr(request.state, "cache_hit"):
            log_data["cache_hit"] = request.state.cache_hit
        if hasattr(request.state, "similarity_score"):
            log_data["similarity_score"] = request.state.similarity_score
        
        # Log based on status code
        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))
        
        # Add timing header
        response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))
        
        return response