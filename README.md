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
| `graph.py` | `MemoryGraph` | In-memory NetworkX graph for prototyping (`from memory_decay.graph import MemoryGraph`) |
| `decay.py` | `DecayEngine` | Time-step decay with exponential/power-law modes, stability modulation |
| `memory_store.py` | `MemoryStore` | SQLite + sqlite-vec persistence for production use |
| `server.py` | FastAPI app | HTTP API for store/search/tick/forget operations |
| `embedding_provider.py` | `EmbeddingProvider` | Pluggable embeddings: Gemini, OpenAI, local sentence-transformers |

## Installation

> **Python 3.13.11+ 또는 3.10~3.12 권장.** Python 3.13.8은 torch를 깨뜨리는 CPython 버그가 있고, python.org macOS 인스톨러는 sqlite-vec가 필요로 하는 SQLite extension 로딩을 지원하지 않습니다. 자세한 내용은 아래 [macOS 호환성 참고](#macos-compatibility-notes) 참조.

```bash
# 권장: uv로 Python 버전 고정
uv venv --python 3.13.11   # 또는 3.10, 3.11, 3.12
uv pip install memory-decay

# 로컬 임베딩 (sentence-transformers + torch):
uv pip install "memory-decay[local]"
```

```bash
# pip 사용 시
pip install memory-decay
pip install "memory-decay[local]"   # 로컬 임베딩용
```

### From Source (Development)

```bash
git clone https://github.com/memory-decay/memory-decay-core.git
cd memory-decay-core
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.10 (3.13.11+ 또는 3.10~3.12 권장)
- NetworkX, NumPy
- FastAPI + Uvicorn (server mode)
- sqlite-vec (vector search persistence — SQLite extension 로딩 지원 필요)
- Optional: `openai`, `google-genai` (for API-based embeddings)
- Optional: `sentence-transformers` (for local embeddings, install with `pip install memory-decay[local]`)

### macOS Compatibility Notes

sqlite-vec는 Python이 SQLite loadable extension을 지원해야 하고, local 임베딩은 torch가 정상 import되어야 합니다. macOS에서는 Python 설치 방식에 따라 이 두 가지가 깨질 수 있습니다.

| Python 설치 방식 | sqlite extension 로딩 | torch (local 임베딩) | 비고 |
|---|---|---|---|
| **uv** (python-build-standalone) | O | O | 권장 |
| **homebrew** | O | O* | *3.13.8은 torch 불가 |
| **pyenv** | O | O | 소스 빌드, 플래그 포함 |
| **python.org 인스톨러** | **X** | O | `--enable-loadable-sqlite-extensions` 누락 |

**알려진 이슈:**

- **Python 3.13.8 + torch**: CPython 3.13.8의 `ast.parse()` 리그레션이 torch import를 깨뜨림 ([pytorch/pytorch#178255](https://github.com/pytorch/pytorch/issues/178255)). Python 3.13.11+에서 수정됨.
- **python.org macOS 인스톨러 + sqlite-vec**: 공식 macOS 인스톨러가 SQLite extension 로딩 없이 빌드됨. uv, homebrew, 또는 pyenv를 사용할 것.

## Quick Start

### As a Library

```python
from memory_decay import MemoryStore, DecayEngine
from memory_decay.embedding_provider import create_embedding_provider

# 1. Create a memory store with Gemini embeddings
store = MemoryStore(
    db_path="./data/memories.db",
    embedding_provider=create_embedding_provider("gemini", api_key="your-api-key"),
)

# 2. Add memories
store.add_memory(
    memory_id="m1",
    mtype="fact",            # "fact" or "episode"
    content="Seoul is the capital of South Korea",
    impact=0.9,              # importance: 0.0-1.0
    created_tick=0,
    associations=[("m2", 0.7)],  # linked memories
)

# 3. Set up decay
engine = DecayEngine(store, decay_type="exponential")

# 4. Advance time — memories decay each tick
for _ in range(100):
    engine.tick()

# 5. Search with activation-weighted retrieval
results = store.search(
    query="What is the capital?",
    top_k=5,
    activation_weight=0.5,   # blend similarity with activation
    bm25_weight=0.3,         # hybrid semantic + lexical search
)

# 6. Reinforce recalled memories (testing effect)
store.re_activate("m1", boost_amount=0.1, source="direct", reinforce=True)
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
| `POST` | `/store` | Store a memory with text, importance, type, category, and associations |
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
# Store a memory with category and calibrated importance
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "importance": 0.9, "mtype": "fact", "category": "preference"}'

# Store a decision
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Chose SQLite over Postgres for single-node simplicity", "importance": 0.8, "mtype": "fact", "category": "decision"}'

# Store an episode (low importance — decays faster)
curl -X POST http://localhost:8100/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Finished migrating auth middleware", "importance": 0.5, "mtype": "episode", "category": "episode"}'

# Search
curl -X POST http://localhost:8100/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What theme does the user like?", "top_k": 5}'

# Advance time (apply decay)
curl -X POST http://localhost:8100/tick \
  -H "Content-Type: application/json" \
  -d '{"count": 10}'
```

## Categories vs Types

Memories have two classification fields:

| Field | Values | Purpose |
|-------|--------|---------|
| `mtype` | `fact`, `episode` | Controls **decay rate** — facts decay slower (`lambda_fact=0.02`) than episodes (`lambda_episode=0.035`) |
| `category` | `fact`, `decision`, `preference`, `episode` | **Semantic label** for retrieval and filtering — returned in search results |

If `category` is omitted, it defaults to the `mtype` value. The recommended mapping from plugins:

| Category | `mtype` | Importance | Use case |
|----------|---------|------------|----------|
| `preference` | `fact` | 0.8–1.0 | User's role, style, habits, likes/dislikes |
| `decision` | `fact` | 0.8–0.9 | Why X was chosen, tradeoffs, rejected alternatives |
| `fact` | `fact` | 0.7–0.9 | Technical facts, API behaviors, architecture |
| `episode` | `episode` | 0.3–0.6 | What was worked on, session context |

Preferences and decisions use `mtype: "fact"` because they should decay slowly like facts, but carry a distinct `category` so agents can distinguish them in search results.

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

Evaluated on the full [LongMemBench](https://github.com/jasonphd/LongMemBench) benchmark (500 questions) using GPT-4o as judge, testing the complete pipeline: memory storage → decay → retrieval → answer generation.

| Metric | Score |
|--------|-------|
| **Accuracy** | **81%** |

## OpenClaw Plugin Integration

memory-decay-core is designed to back the **openclaw-memory-decay** TypeScript plugin. The plugin connects to the server's HTTP API and provides AI agents with decaying, searchable memory.

### Setup

1. Install the package:
```bash
pip install memory-decay
```

2. Start the memory-decay server:
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
│   ├── __init__.py           # Public API: MemoryGraph, DecayEngine, MemoryStore
│   ├── graph.py              # Graph memory store + hybrid search
│   ├── decay.py              # Decay math (exponential, power law, soft-floor)
│   ├── bm25.py               # Shared BM25 tokenizer + scorer
│   ├── memory_store.py       # SQLite + sqlite-vec persistence
│   ├── server.py             # FastAPI HTTP server
│   └── embedding_provider.py # Pluggable embedding backends
├── tests/
├── data/                     # Default SQLite DB location
└── pyproject.toml
```

## References

- Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*
- Roediger, H. L., & Butler, A. C. (2011). The critical role of retrieval practice in long-term retention
- Wixted, J. T. (2004). On Common Ground: Jost's (1897) law of forgetting and Ribot's (1881) law of retrograde amnesia
