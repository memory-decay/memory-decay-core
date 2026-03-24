# memory-decay-core

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg)](https://fastapi.tiangolo.com)

**Human-like memory decay for AI agents.** A Python library that models how memories naturally fade, strengthen through recall, and compete for retrieval — giving agents realistic, bounded memory instead of perfect total recall.

```
Activation
  1.0 ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
      │ ■ high-impact fact (impact=0.9)
  0.8 ┤  ■■■■
      │      ■■■■■                                ← stability slows decay
  0.6 ┤  ●        ■■■■■■■
      │   ●●●           ■■■■■■■■■
  0.4 ┤      ●●●●              ■■■■■■■■■■■■■■■■■ ← floor: high-impact
      │          ●●●●●                              memories never fully
  0.2 ┤  ▴           ●●●●●●●                        vanish
      │   ▴▴▴▴            ●●●●●●●●●●●●●
  0.0 ┤       ▴▴▴▴▴▴▴▴▴▴▴▴▴▴▴                    ← low-impact episodes
      └─────────────────────────────────────────── Time (ticks)
        ■ high-impact fact    ● medium episode    ▴ low-impact episode
```

## Key Ideas

**Memory isn't a database.** Humans don't store-and-retrieve — they encode, decay, interfere, and reconstruct. This library models that process with three measurable pillars:

| Pillar | What it measures | Weight |
|--------|-----------------|--------|
| **Retrieval** | Can the system find the right memory? (recall + MRR) | 40% |
| **Forgetting** | Does it forget what it should? (non-target decay) | 35% |
| **Plausibility** | Does activation predict recallability? (correlation) | 25% |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Your Agent                        │
│                                                      │
│  POST /store   POST /search   POST /auto-tick       │
└────────┬──────────┬──────────────┬──────────────────┘
         │          │              │
┌────────▼──────────▼──────────────▼──────────────────┐
│              FastAPI Server (server.py)               │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ Embedding │  │  Search  │  │ Retrieval          │ │
│  │ Provider  │  │  (vec +  │  │ Consolidation      │ │
│  │ (Gemini/  │  │  BM25    │  │ (testing effect)   │ │
│  │  OpenAI/  │  │  hybrid) │  │                    │ │
│  │  local)   │  │          │  │                    │ │
│  └────┬─────┘  └────┬─────┘  └────────┬───────────┘ │
│       │              │                 │             │
│  ┌────▼──────────────▼─────────────────▼───────────┐ │
│  │           MemoryStore (SQLite + sqlite-vec)      │ │
│  │  memories table │ vec_memories │ embedding_cache  │ │
│  └──────────────────────────────────────────────────┘ │
│                         │                             │
│  ┌──────────────────────▼───────────────────────────┐ │
│  │              DecayEngine                          │ │
│  │  exponential / power_law / custom soft-floor      │ │
│  │  stability-weighted rate scaling                  │ │
│  └───────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Core Components

| Module | Class | Role |
|--------|-------|------|
| `graph.py` | `MemoryGraph` | NetworkX DiGraph with vector embeddings, BM25 hybrid search, spreading activation |
| `decay.py` | `DecayEngine` | Time-step decay with exponential/power-law modes, stability modulation |
| `evaluator.py` | `Evaluator` | 3-pillar scoring: retention AUC, forgetting, plausibility |
| `memory_store.py` | `MemoryStore` | SQLite + sqlite-vec persistence, vector KNN search, embedding cache |
| `server.py` | FastAPI app | HTTP API for store/search/tick/forget operations |
| `embedding_provider.py` | `EmbeddingProvider` | Pluggable embeddings: Gemini, OpenAI, local sentence-transformers |

## Installation

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.10
- NetworkX, NumPy, sentence-transformers
- FastAPI + Uvicorn (server mode)
- sqlite-vec (vector search persistence)
- Optional: `openai`, `google-genai` (for API-based embeddings)

## Quick Start

### As a Library

```python
from memory_decay import MemoryGraph, DecayEngine, Evaluator

# 1. Create a memory graph with a custom embedder
graph = MemoryGraph(embedder=my_embed_fn)

# 2. Add memories
graph.add_memory(
    memory_id="m1",
    mtype="fact",            # "fact" or "episode"
    content="Seoul is the capital of South Korea",
    impact=0.9,              # importance: 0.0-1.0
    created_tick=0,
    associations=[("m2", 0.7)],  # linked memories
)

# 3. Set up decay
engine = DecayEngine(graph, decay_type="exponential")

# 4. Advance time — memories decay each tick
for _ in range(100):
    engine.tick()

# 5. Search with activation-weighted retrieval
results = graph.query_by_similarity(
    "What is the capital?",
    top_k=5,
    activation_weight=0.5,   # blend similarity with activation
    bm25_weight=0.3,         # hybrid semantic + lexical search
)

# 6. Reinforce recalled memories (testing effect)
graph.re_activate("m1", boost_amount=0.1, source="direct", reinforce=True)
```

### As an HTTP Server

```bash
# Start with local embeddings (no API key needed)
python -m memory_decay.server --port 8100

# Start with Gemini embeddings
python -m memory_decay.server \
    --port 8100 \
    --embedding-provider gemini \
    --embedding-api-key $GEMINI_API_KEY \
    --db-path ./data/memories.db

# Start with OpenAI embeddings
python -m memory_decay.server \
    --embedding-provider openai \
    --embedding-api-key $OPENAI_API_KEY \
    --embedding-model text-embedding-3-small
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/store` | Store a memory with text, importance, type, and associations |
| `POST` | `/store-batch` | Store multiple memories in one call |
| `POST` | `/search` | Semantic search with activation weighting + retrieval consolidation |
| `POST` | `/tick` | Advance decay by N ticks |
| `POST` | `/auto-tick` | Apply ticks based on elapsed real time |
| `DELETE` | `/forget/{id}` | Explicitly delete a memory |
| `POST` | `/reset` | Clear all memories |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Memory count and tick state |

#### Example Requests

```bash
# Store a memory
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "importance": 0.8, "mtype": "fact"}'

# Search
curl -X POST http://localhost:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What theme does the user like?", "top_k": 5}'

# Advance time (apply decay)
curl -X POST http://localhost:8100/tick \
  -H "Content-Type: application/json" \
  -d '{"count": 10}'
```

## Core Concepts

### Decay Functions

The engine supports two built-in decay modes plus custom functions:

**Exponential decay** (default):
```
A(t+1) = A(t) * exp(-λ_eff)
λ_eff  = λ / ((1 + α * impact) * (1 + ρ * stability))
```

**Power law decay** (longer tail):
```
A(t+1) = A(t) / (1 + β_eff)
β_eff  = β / ((1 + α * impact) * (1 + ρ * stability))
```

**Soft-floor decay** (custom, used in best config):
```
A(t+1) = floor(impact) + (A(t) - floor(impact)) * exp(-rate)
```
High-impact memories decay toward a non-zero floor rather than vanishing, controlled by a sigmoid gate for smooth consolidation transitions.

### Stability & Consolidation

Memories have a **stability score** that modulates decay rate. Higher stability = slower decay.

- Stability starts at 0 and increases when a memory is successfully recalled
- Each tick, stability itself decays slowly (`stability_decay=0.01`), so reinforcement effects are long-lived but finite
- Stability gain follows a saturation curve: `gain * (1 - current/cap)` — diminishing returns prevent runaway accumulation

### Retrieval Consolidation (Testing Effect)

When a memory is successfully recalled during search, it gets strengthened — modeling the well-established [testing effect](https://en.wikipedia.org/wiki/Testing_effect) from cognitive psychology:

1. Memory is found in top-K search results
2. Retrieval score gets boosted (immediate recall advantage)
3. Storage score gets a fractional boost (long-term strengthening)
4. Stability increases (slower future decay)

Multiple consolidation modes are available:
- `activation_and_stability` — boost both scores + stability
- `retrieval_only` — only boost retrieval score
- `stability_only_direct` — only reinforce stability
- `retrieval_with_storage_fraction` — retrieval gets full boost, storage gets 25%
- `retrieval_rank_scaled_fraction` — boost scales inversely with rank position
- `retrieval_capped_fraction` — boost capped at a ceiling value
- `retrieval_margin_bm25_fraction` — requires both semantic margin and lexical agreement

### Dual-Score Model

Each memory carries two activation scores:

| Score | Role | Analogy |
|-------|------|---------|
| **Storage score** | Can the memory be found at all? | "Is it still in the filing cabinet?" |
| **Retrieval score** | How easily can it be accessed? | "Can I find it quickly?" |

Search results are filtered by storage threshold, then ranked by retrieval score blended with similarity.

### Hybrid Search

Retrieval combines three signals:

1. **Vector similarity** — cosine similarity between query and memory embeddings
2. **Activation weighting** — `similarity * retrieval_score^weight` (faded memories rank lower)
3. **BM25 re-ranking** — lexical matching for exact term overlap (optional, configurable weight)

Spreading activation through graph edges also boosts memories whose neighbors are active.

## Configuration

### Decay Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_fact` | 0.02 | Exponential decay rate for facts |
| `lambda_episode` | 0.035 | Exponential decay rate for episodes |
| `alpha` | 0.5 | Impact scaling factor |
| `stability_weight` | 0.8 | How much stability slows decay |
| `stability_decay` | 0.01 | Per-tick stability erosion |
| `reinforcement_gain_direct` | 0.2 | Stability boost on direct recall |
| `reinforcement_gain_assoc` | 0.05 | Stability boost on associated recall |
| `stability_cap` | 1.0 | Maximum stability value |

### Server Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8100` | Port number |
| `--db-path` | `data/memories.db` | SQLite database path |
| `--tick-interval` | `3600` | Real seconds per tick |
| `--embedding-provider` | `local` | `local`, `gemini`, or `openai` |
| `--embedding-model` | auto | Model name (provider-specific) |
| `--embedding-dim` | auto | Embedding dimension (auto-detected) |
| `--experiment-dir` | `experiments/best` | Custom decay function directory |

### Custom Decay Functions

Place a `decay_fn.py` in your experiment directory with a `compute_decay` function:

```python
# experiments/my_experiment/decay_fn.py
def compute_decay(activation, impact, stability, mtype, params):
    """Custom decay: must return float in [0, 1]."""
    # Your decay math here
    return new_activation
```

The server auto-loads from `experiments/best/` on startup. Override with `--experiment-dir`.

## Benchmarks

Best optimized configuration (soft-floor decay with retrieval consolidation, 200-tick simulation):

| Metric | Score | Description |
|--------|-------|-------------|
| **Retention AUC** | **0.816** | Mean recall across time checkpoints (higher = better retention) |
| **Overall Score** | **0.711** | Weighted composite of all three pillars |
| **Recall Rate** | **0.634** | Fraction of target memories successfully retrieved at final tick |
| **Plausibility** | **0.964** | Correlation between activation and recallability |

The retention AUC of 81.6% means the system retains most important memories through extended time horizons while still allowing low-importance memories to naturally fade — matching the qualitative pattern of human memory.

## OpenClaw Plugin Integration

memory-decay-core is designed to back the **openclaw-memory-decay** TypeScript plugin. The plugin connects to the server's HTTP API and provides AI agents with decaying, searchable memory.

### Setup

1. Start the memory-decay server:
```bash
python -m memory_decay.server --port 8100 --db-path ./data/agent_memories.db
```

2. Configure the plugin to point at `http://localhost:8100`

3. The plugin calls:
   - `POST /store` when the agent forms new memories
   - `POST /search` when the agent needs to recall (triggers retrieval consolidation automatically)
   - `POST /auto-tick` periodically to advance decay based on real elapsed time
   - `DELETE /forget/{id}` for explicit forgetting

### Auto-Tick

The `/auto-tick` endpoint maps real time to simulation ticks:

```
ticks_due = floor(elapsed_seconds / tick_interval)
```

With the default `tick_interval=3600`, one tick equals one hour. This means:
- Recent memories (< 1 hour) are at full activation
- Day-old memories have decayed through ~24 ticks
- Week-old memories have been through ~168 ticks

Adjust `--tick-interval` to control how aggressively memories fade.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
memory-decay-core/
├── src/memory_decay/
│   ├── __init__.py           # Public API: MemoryGraph, DecayEngine, Evaluator, MemoryStore
│   ├── graph.py              # Graph memory store + hybrid search
│   ├── decay.py              # Decay math (exponential, power law, soft-floor)
│   ├── evaluator.py          # 3-pillar evaluation framework
│   ├── memory_store.py       # SQLite + sqlite-vec persistence
│   ├── server.py             # FastAPI HTTP server
│   ├── embedding_provider.py # Pluggable embedding backends
│   └── main.py               # Simulation runner
├── tests/
│   └── test_simulation.py    # Integration tests
├── data/                     # Default SQLite DB location
└── pyproject.toml
```

## References

- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*
- Roediger, H. L., & Butler, A. C. (2011). The critical role of retrieval practice in long-term retention
- Wixted, J. T. (2004). On Common Ground: Jost's (1897) law of forgetting and Ribot's (1881) law of retrograde amnesia
