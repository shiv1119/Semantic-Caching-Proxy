import math
from typing import Dict
from datetime import datetime

class TTLCalculator:
    """
    Calculates a dynamic TTL for each cache entry based on four factors:
    popularity, query complexity, time of day, and day of week.
    The goal is to keep hot/simple queries cached longer and
    expire specific or time-sensitive queries faster.
    """
    
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
        """
        Applies four multipliers to base_ttl in sequence and clamps
        the result between 5 minutes and 24 hours.
        """
        ttl = base_ttl
        
        # Popular queries stay cached longer — they're clearly being reused.
        # Queries with fewer than 5 hits get a shorter TTL since we don't
        # yet know if they'll be asked again.
        if hit_count > 100:
            ttl = int(ttl * 4)
        elif hit_count > 50:
            ttl = int(ttl * 2)
        elif hit_count > 20:
            ttl = int(ttl * 1.5)
        elif hit_count < 5:
            ttl = int(ttl * 0.8)
        
        # Long queries tend to be highly specific, so their cached answers
        # go stale faster. Short generic queries are safe to cache longer.
        if query_length > self.complexity_threshold:
            ttl = int(ttl * 0.7)
        elif query_length < 50:
            ttl = int(ttl * 1.2)
        
        # During business hours content changes more frequently,
        # so we shorten TTL to avoid serving outdated responses.
        # The night window condition is intentionally written as >= 22 OR <= 6
        # since it wraps around midnight.
        if 9 <= time_of_day <= 17:
            ttl = int(ttl * 0.6)
        elif 22 <= time_of_day <= 6:
            ttl = int(ttl * 1.5)
        
        # Lower traffic on weekends means less content churn,
        # so we can afford to hold entries longer.
        if datetime.now().weekday() >= 5:
            ttl = int(ttl * 1.3)
        
        # Hard bounds — never expire in under 5 min or over 24 hours
        min_ttl = 300    # 5 minutes
        max_ttl = 86400  # 24 hours
        
        return max(min_ttl, min(ttl, max_ttl))


class AdaptiveTTLStrategy:
    """
    Learns optimal TTL from real access patterns rather than using fixed multipliers.
    Maintains a rolling 1-hour window of access timestamps per key and
    adjusts TTL based on observed frequency. More accesses = longer TTL.
    """
    
    def __init__(self):
        self.access_patterns: Dict[str, list] = {}
        self.learning_window = 3600  # 1 hour rolling window
        
    def record_access(self, key: str, timestamp: float):
        """
        Appends a new access timestamp and prunes entries older than the
        learning window. Keeping the list bounded avoids unbounded memory growth
        for keys that get hit thousands of times.
        """
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(timestamp)
        
        cutoff = timestamp - self.learning_window
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def get_optimal_ttl(self, key: str, base_ttl: int) -> int:
        """
        Returns a TTL scaled to access frequency in the last hour.
        Falls back to base_ttl for keys we haven't seen before.
        The caps (86400, 43200) prevent runaway TTL growth for viral queries.
        """
        if key not in self.access_patterns:
            return base_ttl
        
        accesses = len(self.access_patterns[key])
        
        if accesses > 100:
            return min(base_ttl * 4, 86400)   # cap at 24h
        elif accesses > 50:
            return min(base_ttl * 2, 43200)   # cap at 12h
        elif accesses < 5:
            return base_ttl // 2              # halve TTL for rarely-seen keys
        
        return base_ttl