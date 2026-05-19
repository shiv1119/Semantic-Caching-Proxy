import math
from typing import Dict
from datetime import datetime

class TTLCalculator:
    """Adaptive TTL based on multiple factors"""
    
    def __init__(self):
        self.popularity_decay = 0.95
        self.complexity_threshold = 200
        
    def calculate(
        self, 
        hit_count: int, 
        query_length: int, 
        base_ttl: int, 
        time_of_day: int
    ) -> int:
        """Calculate dynamic TTL"""
        ttl = base_ttl
        
        # Factor 1: Popularity (exponential scaling)
        if hit_count > 100:
            ttl = int(ttl * 4)
        elif hit_count > 50:
            ttl = int(ttl * 2)
        elif hit_count > 20:
            ttl = int(ttl * 1.5)
        elif hit_count < 5:
            ttl = int(ttl * 0.8)
        
        # Factor 2: Query complexity (longer queries are more specific)
        if query_length > self.complexity_threshold:
            ttl = int(ttl * 0.7)  # Shorter TTL for complex queries
        elif query_length < 50:
            ttl = int(ttl * 1.2)  # Longer TTL for simple queries
        
        # Factor 3: Time of day (business hours get shorter TTL)
        if 9 <= time_of_day <= 17:  # Business hours
            ttl = int(ttl * 0.6)  # More frequent updates during work
        elif 22 <= time_of_day <= 6:  # Night time
            ttl = int(ttl * 1.5)  # Longer TTL at night
        
        # Factor 4: Day of week
        if datetime.now().weekday() >= 5:  # Weekend
            ttl = int(ttl * 1.3)  # Longer TTL on weekends
        
        # Ensure bounds
        min_ttl = 300  # 5 minutes minimum
        max_ttl = 86400  # 24 hours maximum
        
        return max(min_ttl, min(ttl, max_ttl))

class AdaptiveTTLStrategy:
    """Self-learning TTL based on access patterns"""
    
    def __init__(self):
        self.access_patterns: Dict[str, list] = {}
        self.learning_window = 3600  # 1 hour
        
    def record_access(self, key: str, timestamp: float):
        """Record access timestamp for learning"""
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(timestamp)
        
        # Clean old entries
        cutoff = timestamp - self.learning_window
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def get_optimal_ttl(self, key: str, base_ttl: int) -> int:
        """Calculate optimal TTL based on access frequency"""
        if key not in self.access_patterns:
            return base_ttl
        
        accesses = len(self.access_patterns[key])
        
        if accesses > 100:
            # Frequently accessed, longer TTL
            return min(base_ttl * 4, 86400)
        elif accesses > 50:
            return min(base_ttl * 2, 43200)
        elif accesses < 5:
            return base_ttl // 2
        
        return base_ttl