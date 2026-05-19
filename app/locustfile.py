from locust import HttpUser, task, between, events
import random
import time

# Simple counters for metrics
cache_hits = 0
cache_misses = 0
total_requests = 0

class SemanticCacheUser(HttpUser):
    """Simulates users testing the semantic cache proxy"""
    
    wait_time = between(1, 3)
    
    # Common queries for testing
    queries = [
        "What is Python?",
        "Explain Python programming language",
        "What is machine learning?",
        "Explain ML concepts",
        "What is Docker?",
        "Containerization explained",
        "What is Redis?",
        "Redis cache database",
        "What is FastAPI?",
        "FastAPI framework tutorial"
    ]
    
    def on_start(self):
        """Called when user starts"""
        print(f"User started testing semantic cache")
    
    @task(3)
    def chat_completion(self):
        """Test main chat endpoint"""
        global cache_hits, cache_misses, total_requests
        
        # Pick a random query
        query = random.choice(self.queries)
        
        # Make the request
        with self.client.post(
            "/api/v1/chat/completions",
            json={
                "prompt": query,
                "temperature": 0.7,
                "max_tokens": 500
            },
            catch_response=True,
            name="/chat/completions"
        ) as response:
            
            if response.status_code == 200:
                data = response.json()
                total_requests += 1
                
                if data.get("cached"):
                    cache_hits += 1
                    response.success()
                    # Add custom attribute for response time tracking
                    response.custom_data = {"cache_hit": True, "latency": data.get("latency_ms", 0)}
                else:
                    cache_misses += 1
                    response.success()
                    response.custom_data = {"cache_hit": False, "latency": data.get("latency_ms", 0)}
                
                # Print occasional stats
                if total_requests % 20 == 0:
                    hit_rate = (cache_hits / total_requests) * 100
                    print(f"Stats - Requests: {total_requests}, Cache Hits: {cache_hits}, Hit Rate: {hit_rate:.1f}%")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def batch_completions(self):
        """Test batch endpoint"""
        # Get 3 random queries
        batch_queries = random.sample(self.queries, min(3, len(self.queries)))
        
        with self.client.post(
            "/api/v1/batch/completions",
            json={"prompts": batch_queries},
            catch_response=True,
            name="/batch/completions"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                cached_count = data.get("cached_count", 0)
                response.success()
                print(f"Batch: {cached_count}/{len(batch_queries)} cached")
            else:
                response.failure(f"Batch failed: {response.status_code}")
    
    @task(1)
    def cache_stats(self):
        """Check cache statistics"""
        with self.client.get(
            "/cache/stats",
            catch_response=True,
            name="/cache/stats"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                hit_rate = data.get("metrics", {}).get("hit_rate", 0)
                print(f"Cache Stats - Hit Rate: {hit_rate:.2%}, Size: {data.get('total_entries', 0)}")
                response.success()
            else:
                response.failure(f"Stats failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed")

class LoadTestUser(SemanticCacheUser):
    """Heavy load testing with minimal wait time"""
    wait_time = between(0, 0.5)
    
    @task
    def aggressive_test(self):
        """High throughput testing"""
        query = random.choice(self.queries)
        
        with self.client.post(
            "/api/v1/chat/completions",
            json={"prompt": query},
            catch_response=True,
            name="/aggressive"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure("Failed")

class CachePerformanceUser(HttpUser):
    """Test cache performance with repeated queries"""
    wait_time = between(0.5, 1)
    
    @task
    def repeated_query(self):
        """Send same query multiple times to test cache hit rate"""
        # Use a fixed query that will definitely be cached after first call
        query = "What is semantic caching?"
        
        with self.client.post(
            "/api/v1/chat/completions",
            json={"prompt": query},
            catch_response=True,
            name="/cache_test"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("cached"):
                    response.success()
                else:
                    # First request might be miss, that's ok
                    response.success()
            else:
                response.failure(f"Failed: {response.status_code}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "="*60)
    print("🚀 SEMANTIC CACHE PROXY LOAD TEST STARTING")
    print("="*60)
    print("Testing endpoints:")
    print("  ✓ POST /api/v1/chat/completions")
    print("  ✓ POST /api/v1/batch/completions")
    print("  ✓ GET /cache/stats")
    print("  ✓ GET /health")
    print("="*60 + "\n")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global cache_hits, cache_misses, total_requests
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    if total_requests > 0:
        hit_rate = (cache_hits / total_requests) * 100
        miss_rate = (cache_misses / total_requests) * 100
        
        print(f"Total Requests:        {total_requests}")
        print(f"Cache Hits:            {cache_hits}")
        print(f"Cache Misses:          {cache_misses}")
        print(f"Cache Hit Rate:        {hit_rate:.1f}%")
        print(f"Cache Miss Rate:       {miss_rate:.1f}%")
        print(f"Cost Savings:          {hit_rate:.1f}% reduction in LLM calls")
        
        # Performance assessment
        if hit_rate > 80:
            print("\n✅ EXCELLENT! Cache is working very well")
        elif hit_rate > 60:
            print("\n👍 GOOD! Cache is performing well")
        elif hit_rate > 40:
            print("\n⚠️  MODERATE - Consider adjusting similarity threshold")
        else:
            print("\n❌ POOR - Check Redis connection and similarity threshold")
    else:
        print("No requests completed")
    
    print("="*60 + "\n")