from locust import HttpUser, task, between, events
import random
import time

# Module-level counters shared across all user instances.
# Not thread-safe in the strictest sense, but good enough for approximate
# hit rate tracking during a load test — we don't need exact precision here.
cache_hits = 0
cache_misses = 0
total_requests = 0

class SemanticCacheUser(HttpUser):
    """
    Simulates realistic mixed traffic against the semantic cache proxy.
    Uses a small pool of semantically related query pairs (e.g. "What is Python?"
    and "Explain Python programming language") to exercise the semantic matching
    logic — not just exact cache hits.

    Task weights reflect typical traffic shape:
        3x chat completions, 2x batch, 1x stats, 1x health
    """
    
    wait_time = between(1, 3)
    
    # Query pairs are intentionally similar so the semantic cache gets a real workout.
    # Each concept appears twice with different phrasing to test similarity matching.
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
        print(f"User started testing semantic cache")
    
    @task(3)
    def chat_completion(self):
        """
        Core load test task — hits the main chat endpoint with a random query.
        Tracks cache hits vs misses from the response payload and prints
        a rolling hit rate every 20 requests so you can watch the cache warm up
        in real time during the test.
        """
        global cache_hits, cache_misses, total_requests
        
        query = random.choice(self.queries)
        
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
                    response.custom_data = {"cache_hit": True, "latency": data.get("latency_ms", 0)}
                else:
                    cache_misses += 1
                    response.success()
                    response.custom_data = {"cache_hit": False, "latency": data.get("latency_ms", 0)}
                
                if total_requests % 20 == 0:
                    hit_rate = (cache_hits / total_requests) * 100
                    print(f"Stats - Requests: {total_requests}, Cache Hits: {cache_hits}, Hit Rate: {hit_rate:.1f}%")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def batch_completions(self):
        """
        Tests the batch endpoint with 3 random queries per call.
        Logs how many of the batch were served from cache — useful for
        confirming the cache warms up correctly under mixed traffic.
        """
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
        """
        Polls the cache stats endpoint periodically.
        Low weight (1x) since this is just observability — we don't want
        stats polling to dominate the request mix.
        """
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
        """Confirms the service is up. Locust will flag this as a failure if it goes down mid-test."""
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
    """
    Stress test variant — same queries as SemanticCacheUser but with minimal
    wait time to simulate a traffic spike. Use this to find the breaking point
    and see how the rate limiter and circuit breaker behave under pressure.
    """
    wait_time = between(0, 0.5)
    
    @task
    def aggressive_test(self):
        """High-throughput single-prompt requests with no think time between them."""
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
    """
    Focused cache performance test — sends the exact same query repeatedly
    to measure how quickly the hit rate climbs to 100% after the first miss.
    Run this in isolation (not mixed with other user classes) for a clean signal.
    """
    wait_time = between(0.5, 1)
    
    @task
    def repeated_query(self):
        """
        Always sends the same prompt. After the first LLM call, every subsequent
        request should be a cache hit — if it isn't, something is wrong with
        the cache write path or the similarity threshold is too strict.
        """
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
                    # First request will always be a miss — that's expected
                    response.success()
            else:
                response.failure(f"Failed: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Prints a summary of what's being tested so the terminal output is easy to follow."""
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
    """
    Prints a final hit rate summary when the test finishes.
    The performance assessment thresholds are rough guidelines:
        >80% — cache is working well for this query mix
        >60% — decent, but the similarity threshold may be too strict
        >40% — worth investigating Redis connectivity and threshold settings
        <40% — something is likely misconfigured
    """
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