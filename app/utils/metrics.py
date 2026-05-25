import time
from collections import deque
from typing import Dict, List
import threading
from datetime import datetime
import numpy as np

class MetricsCollector:
    """Keeps track of how the server is performing over time — things like cache hit rate,
    how slow or fast responses are, and how many errors we're seeing"""
    
    def __init__(self, window_size: int = 3600):  # keeps data for the last 1 hour
        self.window_size = window_size
        self.latencies = deque(maxlen=10000)  # stores last 10k response times, older ones get dropped automatically
        self.cache_hits = 0       # how many times we served a response from cache
        self.cache_misses = 0     # how many times we had to actually call the model
        self.errors = 0           # anything that went wrong
        self.l1_hits = 0          # hits from the faster in-memory cache (level 1)
        self.l2_hits = 0          # hits from Redis (level 2, slightly slower but bigger)
        self.request_timestamps = deque(maxlen=window_size)  # used to calculate requests/sec
        self.cache_write_ops = 0  # how many times we wrote a new entry to cache
        self.batch_writes = 0     # writes that were grouped together for efficiency
        
    def record_request(self, cached: bool, latency_ms: float, similarity: float = None):
        """Called after every request — saves how long it took and whether cache helped or not"""
        self.latencies.append(latency_ms)
        self.request_timestamps.append(time.time())
        
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_cache_hit(self, level: str, latency_ms: float = 0, similarity: float = 1.0):
        """Logs a cache hit and also tracks *which* cache level served it (L1 or L2)
        — helps us know if the faster cache is pulling its weight"""
        if level == "l1":
            self.l1_hits += 1
        else:
            self.l2_hits += 1
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Nothing in cache matched — we'll have to do the actual work this time"""
        self.cache_misses += 1
    
    def record_cache_write(self):
        """Called when we store a new response in the cache for future reuse"""
        self.cache_write_ops += 1
    
    def record_batch_write(self, count: int):
        """Same as above but for when multiple entries are written in one go"""
        self.batch_writes += count
    
    def record_error(self):
        """Something broke — bump the error counter so we can track how often this happens"""
        self.errors += 1
    
    def get_summary(self) -> dict:
        """Crunches all the numbers and returns a snapshot of how things are going right now"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)  # max(..., 1) avoids dividing by zero on startup
        
        # p95 and p99 latency tell you about worst-case performance —
        # p95 means 95% of requests were faster than this number
        latencies_list = list(self.latencies)
        p95 = np.percentile(latencies_list, 95) if latencies_list else 0
        p99 = np.percentile(latencies_list, 99) if latencies_list else 0
        
        # count requests from the last 60 seconds to get a per-second rate
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
        """Reshapes our summary into the format Prometheus expects —
        Prometheus is a monitoring tool that scrapes this endpoint on a schedule"""
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
        """Wipes everything back to zero — useful in tests or if you want a fresh start
        without actually restarting the server"""
        self.latencies.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.request_timestamps.clear()
        self.cache_write_ops = 0