<div align="center">

<h1 align="center">Semantic Caching Proxy</h1>

<p align="center">An intelligent LLM proxy with semantic caching, circuit breakers, rate limiting, and production monitoring</p>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
  <img alt="Github language count" src="https://img.shields.io/github/languages/count/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
  <img alt="Repository size" src="https://img.shields.io/github/repo-size/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
  <img alt="Github issues" src="https://img.shields.io/github/issues/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
  <img alt="Github forks" src="https://img.shields.io/github/forks/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
  <img alt="Github stars" src="https://img.shields.io/github/stars/shiv1119/Semantic-Caching-Proxy?color=56BEB8">
</p>

<p align="center">
  <a href="#dart-objective">Objective</a> &#xa0;|&#xa0;
  <a href="#rocket-tech-stack">Tech Stack</a> &#xa0;|&#xa0;
  <a href="#building_construction-architecture">Architecture</a> &#xa0;|&#xa0;
  <a href="#zap-how-semantic-caching-works">How It Works</a> &#xa0;|&#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0;|&#xa0;
  <a href="#checkered_flag-getting-started">Getting Started</a> &#xa0;|&#xa0;
  <a href="#bar_chart-monitoring">Monitoring</a> &#xa0;|&#xa0;
  <a href="#gear-configuration">Configuration</a> &#xa0;|&#xa0;
  <a href="#bulb-design-decisions">Design Decisions</a> &#xa0;|&#xa0;
  <a href="https://github.com/shiv1119" target="_blank">Author</a>
</p>

</div>

---

## :dart: Objective

A production-grade **semantic caching proxy** for LLM APIs that dramatically reduces cost and latency by caching semantically similar queries — not just exact matches.

Instead of forwarding every LLM request to Ollama, the proxy checks if a semantically similar question has been asked before. If yes, it returns the cached answer instantly. This achieves a **60% cache hit rate**, cutting repeated LLM query latency and API costs significantly.

Built with real-world reliability patterns: circuit breakers, rate limiting, batch processing, and full observability via Prometheus + Grafana.

---

## :rocket: Tech Stack

| Layer | Technologies |
|---|---|
| Backend Framework | FastAPI (Python) |
| LLM Backend | Ollama (`nomic-embed-text` model) |
| Caching & Vector Search | Redis Stack (HNSW vector index) |
| Embeddings | 768-dimensional vector embeddings |
| Containerization | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Async HTTP | aiohttp, httpx |
| Rate Limiting | limits library |
| Resilience | Circuit Breaker, Tenacity (retry) |

---

## :building_construction: Architecture

```
         Client Request
               │
               ▼
   ┌───────────────────────┐
   │    FastAPI Proxy       │
   │    Port: 8000          │
   │                        │
   │  ┌──────────────────┐  │
   │  │  Rate Limiter    │  │
   │  └────────┬─────────┘  │
   │           │             │
   │  ┌────────▼─────────┐  │
   │  │ Embedding Engine │  │
   │  │  (Ollama         │  │
   │  │  nomic-embed-    │  │
   │  │  text, 768-dim)  │  │
   │  └────────┬─────────┘  │
   │           │             │
   │  ┌────────▼─────────┐  │
   │  │  Semantic Cache  │◄─┼──── Cache HIT → return instantly
   │  │  Lookup          │  │
   │  │  (Redis HNSW     │  │
   │  │  0.75 threshold) │  │
   │  └────────┬─────────┘  │
   │           │ Cache MISS  │
   │  ┌────────▼─────────┐  │
   │  │ Circuit Breaker  │  │
   │  └────────┬─────────┘  │
   └───────────┼────────────┘
               │
               ▼
   ┌───────────────────────┐
   │    Ollama LLM API     │
   │  host.docker.internal │
   │       :11434          │
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │  Store in Redis Cache │
   │  (embedding + answer) │
   └───────────────────────┘
               │
               ▼
         Return Response
```

### Infrastructure Services

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│                   (cache-network)                        │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ Redis Stack  │   │ Semantic     │   │ Prometheus  │ │
│  │ Port: 6379   │   │ Proxy        │   │ Port: 9091  │ │
│  │ UI:  8001    │   │ Port: 8000   │   │ (monitoring │ │
│  │              │   │ Metrics:9090 │   │  profile)   │ │
│  └──────────────┘   └──────────────┘   └─────────────┘ │
│                                                          │
│  ┌──────────────┐   ┌──────────────────────────────┐   │
│  │   Grafana    │   │  Proxy Cluster (2 replicas)  │   │
│  │ Port: 3000   │   │  Ports: 8002–8003             │   │
│  │ (monitoring  │   │  Memory limit: 1GB per node   │   │
│  │  profile)    │   │  CPU limit: 2 cores per node  │   │
│  └──────────────┘   └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## :zap: How Semantic Caching Works

Traditional caches only match **exact strings**. Semantic caching matches **meaning**.

```
Query 1: "What is the capital of France?"
Query 2: "Which city serves as France's capital?"
Query 3: "France capital city?"

→ All three map to the same cached response.
```

### 3-Tier Caching Architecture

```
Tier 1 — Exact Match      : O(1) lookup by query hash
Tier 2 — Semantic Match   : HNSW vector similarity search
                            (768-dim embeddings, threshold 0.75)
Tier 3 — LLM Fallback     : Forward to Ollama if no match found
                            → Store result back in cache
```

### Cache Hit Decision

```
Cosine Similarity ≥ 0.75  →  Cache HIT  → Return cached answer
Cosine Similarity < 0.75  →  Cache MISS → Call Ollama LLM
```

---

## :white_check_mark: Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Docker | Latest |
| Docker Compose | v2+ |
| Ollama | Latest (running locally) |
| Redis Stack | Included via Docker |

> **Important:** Ollama must be running on your host machine at port `11434` before starting the proxy. The proxy connects to it via `host.docker.internal:11434`.

### Install Ollama and pull the embedding model

```bash
# Install Ollama from https://ollama.com
# Then pull the required embedding model
$ ollama pull nomic-embed-text

# Verify Ollama is running
$ ollama list
```

---

## :checkered_flag: Getting Started

### Option 1 — Docker (Recommended)

```bash
# Clone the repository
$ git clone https://github.com/shiv1119/Semantic-Caching-Proxy.git

# Navigate to the project
$ cd Semantic-Caching-Proxy

# Start core services (proxy + Redis)
$ docker-compose up --build

# Proxy available at: http://localhost:8000
# Redis UI available at: http://localhost:8001
# Metrics endpoint at:  http://localhost:9090/metrics
```

### Option 2 — With Monitoring Stack (Prometheus + Grafana)

```bash
# Start all services including monitoring
$ docker-compose --profile monitoring up --build

# Prometheus: http://localhost:9091
# Grafana:    http://localhost:3000  (admin / admin)
```

### Option 3 — Production Cluster (2 replicas)

```bash
# Start the replicated cluster
$ docker-compose up semantic-proxy-cluster redis-stack --build

# Replica 1: http://localhost:8002
# Replica 2: http://localhost:8003
```

### Option 4 — Manual Setup (Without Docker)

```bash
# Clone the repository
$ git clone https://github.com/shiv1119/Semantic-Caching-Proxy.git
$ cd Semantic-Caching-Proxy

# Create and activate virtual environment
$ python -m venv venv
$ source venv/bin/activate        # Linux/macOS
$ venv\Scripts\Activate           # Windows

# Install dependencies
$ pip install -r requirements.txt

# Set environment variables
$ cp .env.example .env
# Edit .env with your Redis URL and Ollama URL

# Run the application
$ uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## :speech_balloon: API Usage

### Query the Proxy (Main Endpoint)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?"
  }'
```

**Response (Cache MISS — first call):**
```json
{
  "response": "The capital of France is Paris.",
  "cache_hit": false,
  "similarity_score": null,
  "latency_ms": 1240
}
```

**Response (Cache HIT — semantically similar query):**
```json
{
  "response": "The capital of France is Paris.",
  "cache_hit": true,
  "similarity_score": 0.94,
  "latency_ms": 12
}
```

### Check Cache Metrics

```bash
curl -X GET http://localhost:8000/metrics
```

```json
{
  "total_requests": 150,
  "cache_hits": 90,
  "cache_misses": 60,
  "hit_rate": "60%",
  "avg_latency_ms": 18.4
}
```

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

### Clear Cache

```bash
curl -X POST http://localhost:8000/cache/clear
```

---

## :bar_chart: Monitoring

The proxy exposes a Prometheus-compatible `/metrics` endpoint on port `9090`.

### Start Monitoring Stack

```bash
$ docker-compose --profile monitoring up
```

| Service | URL | Credentials |
|---|---|---|
| Grafana Dashboard | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9091 | — |
| Redis Insight UI | http://localhost:8001 | — |
| Metrics Endpoint | http://localhost:9090/metrics | — |

### Key Metrics Tracked

- `cache_hit_total` — Total cache hits
- `cache_miss_total` — Total cache misses
- `llm_request_duration_seconds` — LLM response latency
- `circuit_breaker_state` — Current circuit breaker state
- `rate_limit_exceeded_total` — Rate limit violations

---

## :gear: Configuration

All configuration is via environment variables (`.env` file or Docker Compose):

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `WORKERS` | `4` | Number of uvicorn workers |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `SIMILARITY_THRESHOLD` | `0.75` | Minimum cosine similarity for cache hit |
| `EMBEDDING_DIMENSIONS` | `768` | Vector embedding dimensions |
| `REDIS_MAX_MEMORY` | `2gb` | Redis max memory limit |

---

## :bulb: Design Decisions

### A. Why Redis Stack (not plain Redis)?

Redis Stack includes the **RedisSearch** module with native HNSW (Hierarchical Navigable Small World) vector index support. This enables sub-millisecond approximate nearest-neighbor search across 768-dimensional embeddings — something plain Redis cannot do without a custom implementation.

### B. HNSW vs Flat Vector Search

HNSW provides O(log n) search complexity vs O(n) for flat brute-force search. For a growing cache with thousands of embeddings, this keeps lookup times consistently fast under load.

### C. Circuit Breaker Pattern

If Ollama becomes unavailable or slow (timeout), the circuit breaker opens and fast-fails requests rather than queuing them. This prevents cascade failures and keeps the proxy responsive even when the LLM backend degrades.

### D. 0.75 Similarity Threshold

Through testing, 0.75 cosine similarity was found to be the sweet spot — high enough to avoid false positives (returning wrong cached answers) while low enough to catch paraphrases and rewordings of the same question.

### E. Replicated Cluster Mode

The Docker Compose config supports 2-replica deployment with memory and CPU limits (1GB / 2 cores per container). Both replicas share the same Redis cache, so a cache populated by one replica benefits all replicas.

---

## :triangular_ruler: Trade-offs

| Decision | Benefit | Trade-off |
|---|---|---|
| 0.75 similarity threshold | Avoids false cache hits | May miss some valid semantic matches |
| Redis HNSW over pgvector | Faster vector search, native integration | Separate Redis dependency |
| Ollama over OpenAI | Free, local, no API costs | Requires local GPU/CPU for embedding |
| In-memory cache in Redis | Sub-millisecond lookups | Cache lost on Redis restart (mitigated by persistence config) |
| Circuit breaker | Prevents cascade failure | Adds complexity to request path |

---

## :memo: Future Improvements

- [ ] Add support for OpenAI and Anthropic as LLM backends
- [ ] Implement cache TTL (time-to-live) per query type
- [ ] Add authentication middleware (API keys)
- [ ] Grafana dashboard templates for one-click import
- [ ] Kubernetes Helm chart for cloud deployment
- [ ] Support for streaming LLM responses with cache passthrough

---

Made with :heart: by <a href="https://github.com/shiv1119" target="_blank">Shiv Nandan Verma</a>

<a href="#top">Back to top</a>
