import time
from collections import deque
from typing import Dict, List
import threading
from datetime import datetime
import numpy as np

class MetricsCollector:
    """Collect and expose performance metrics"""
    
    def __init__(self, window_size: int = 3600):  # 1 hour window
        self.window_size = window_size
        self.latencies = deque(maxlen=10000)
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.request_timestamps = deque(maxlen=window_size)
        self.cache_write_ops = 0
        self.batch_writes = 0
        
    def record_request(self, cached: bool, latency_ms: float, similarity: float = None):
        """Record a request with its latency and cache status"""
        self.latencies.append(latency_ms)
        self.request_timestamps.append(time.time())
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_cache_hit(self, level: str, latency_ms: float = 0, similarity: float = 1.0):
        """Record cache hit with level (L1 or L2)"""
        if level == "l1":
            self.l1_hits += 1
        else:
            self.l2_hits += 1
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def record_cache_write(self):
        """Record cache write operation"""
        self.cache_write_ops += 1
    
    def record_batch_write(self, count: int):
        """Record batch write operation"""
        self.batch_writes += count
    
    def record_error(self):
        """Record error"""
        self.errors += 1
    
    def get_summary(self) -> dict:
        """Get metrics summary"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)
        
        # Calculate percentiles
        latencies_list = list(self.latencies)
        p95 = np.percentile(latencies_list, 95) if latencies_list else 0
        p99 = np.percentile(latencies_list, 99) if latencies_list else 0
        
        # Calculate requests per second
        now = time.time()
        recent_requests = [t for t in self.request_timestamps if now - t <= 60]
        rps = len(recent_requests) / 60
        
        return {
            "total_requests": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "avg_latency_ms": round(np.mean(latencies_list), 2) if latencies_list else 0,
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "requests_per_second": round(rps, 2),
            "error_rate": round(self.errors / max(total_requests, 1), 4),
            "cache_write_ops": self.cache_write_ops,
            "batch_writes": self.batch_writes
        }
    
    def get_prometheus_metrics(self) -> dict:
        """Format metrics for Prometheus"""
        summary = self.get_summary()
        
        return {
            "cache_hits_total": summary["cache_hits"],
            "cache_misses_total": summary["cache_misses"],
            "cache_hit_rate": summary["hit_rate"],
            "request_latency_avg_ms": summary["avg_latency_ms"],
            "request_latency_p95_ms": summary["p95_latency_ms"],
            "requests_per_second": summary["requests_per_second"],
            "error_rate": summary["error_rate"],
            "cache_operations_total": summary["cache_write_ops"]
        }
    
    def reset(self):
        """Reset all metrics"""
        self.latencies.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.request_timestamps.clear()
        self.cache_write_ops = 0