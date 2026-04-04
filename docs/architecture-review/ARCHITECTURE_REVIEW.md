# Architecture and Design Review — memory-decay-core

**Date:** 2026-04-05
**Reviewer:** arch-reviewer
**Version reviewed:** 0.1.3

---

## 1. Project Overview

**memory-decay-core** is a Python library implementing human-like memory decay for AI agents. It models how memories naturally fade, strengthen through recall (testing effect), and compete for retrieval. Key differentiating feature: memories have an **impact-modulated soft floor** so high-importance memories never fully vanish.

### Core Components

| Module | Class | Role |
|--------|-------|------|
| `decay.py` | `DecayEngine` | Time-step decay with exponential/power-law modes, stability modulation |
| `graph.py` | `MemoryGraph` | In-memory NetworkX graph for prototyping |
| `memory_store.py` | `MemoryStore` | SQLite + sqlite-vec persistence for production |
| `server.py` | FastAPI app | HTTP API for store/search/tick/forget |
| `embedding_provider.py` | `EmbeddingProvider` | Pluggable embeddings: Gemini, OpenAI, local sentence-transformers |
| `bm25.py` | — | Shared BM25 tokenizer and scorer for hybrid retrieval |

---

## 2. Architecture Analysis

### 2.1 Two-Backend Design

The system has two storage backends with a clean drop-in relationship:

```
MemoryGraph (in-memory, NetworkX) ←→ MemoryStore (SQLite + sqlite-vec)
```

**Strengths:**
- `MemoryGraph` is excellent for prototyping, testing, and single-session use cases
- `MemoryStore` provides persistence, multi-user isolation (via `user_id`), and production scalability
- The interface is close enough that switching is seamless

**Concerns:**
- `MemoryGraph._graph` is exposed publicly as `._graph` (line 198, 224, 291), bypassing encapsulation. Any caller can mutate the internal NetworkX graph directly, potentially corrupting invariants (e.g., the precomputed `_emb_retrieval_scores` array going out of sync).
- `DecayEngine._build_tick_arrays()` accesses `self._graph._graph.nodes` directly, tight-coupling the engine to the graph's internal structure.
- `MemoryStore` and `MemoryGraph` are not fully interchangeable: `MemoryGraph` has `re_activate()` and `reinforce_memory()` which `MemoryStore` doesn't expose as methods (they go through SQL). Some operations differ in behavior between backends.

### 2.2 Hybrid Retrieval Pipeline

The search path in both backends is a two-stage pipeline:

```
1. Semantic vector search (cosine similarity via sqlite-vec / numpy dot product)
      ↓ fetch_k = top_k * 3 (or * 5 when BM25 is active)
2. BM25 lexical re-ranking (when bm25_weight > 0)
      ↓ normalized score blend: (1-bm25_weight)*cosine + bm25_weight*bm25
3. Return top_k
```

**Strengths:**
- Separating vector search (recall) from BM25 (precision) is well-grounded in IR literature
- Pre-normalized embeddings mean L2 distance in sqlite-vec approximates cosine similarity correctly
- BM25 global IDF is precomputed alongside the embedding matrix, avoiding repeated IDF computation

**Concerns:**
- `bm25_weight` and `activation_weight` are pulled from `engine.get_params()` at search time (`server.py:393-402`), meaning the retrieval scoring formula changes depending on the decay engine's parameters. This is a non-obvious coupling: changing decay params changes retrieval rankings.
- When `bm25_weight > 0`, the `fetch_k` multiplier (3x or 5x) means database-level KNN returns more results than ultimately displayed. For very large memory stores, this wastes I/O.

### 2.3 Decay Engine Design

**Exponential decay:**
```
A_new = A₀ * exp(-λ_eff)
λ_eff = λ / ((1 + α * impact) * (1 + ρ * stability))
```

**Power law decay:**
```
A_new = A₀ / (1 + β_eff)
β_eff = β / ((1 + α * impact) * (1 + ρ * stability))
```

**Soft floor** (`soft_floor_decay_step` in `decay.py:26-71`):
- Higher impact → higher floor (0.05–0.35 range by default) → memory never fully dies
- Higher stability → slower effective decay rate
- Numerically stable closed form: `a_{t+1} = floor + (a_t - floor) * exp(-rate)` — no iteration needed

**Strengths:**
- Mathematically sound: the soft floor guarantee (activation never increases on pure decay) is properly proven in the docstring
- The saturation function in `reinforce_memory` (`(1 - stability/cap) * gain`) correctly models diminishing returns from repeated reinforcement
- Vectorized tick arrays for bulk decay computation — avoids per-node Python loop overhead

**Concerns:**
- `soft_floor_decay_step` is a standalone function but is **never actually called** from the `DecayEngine` class. The engine uses `_compute_decay` which implements exponential/power_law but NOT the soft floor variant. This is dead code or a planned-but-unfinished feature.
- `DecayEngine._tick_store()` computes new stability as `min(stability * (1 - decay), cap)` but the vectorized path (`_build_tick_arrays`) uses `np.minimum(self._tick_stability * (1.0 - stability_decay), stability_cap)`. Both should be equivalent, but the scalar vs vectorized paths have slightly different semantics (the scalar path uses `min()` without `maximum(..., 0)` so stability could theoretically go negative — though `stability_decay < 1` prevents this in practice).

### 2.4 Embedding Provider Design

Pluggable provider pattern with factory constructor.

**Strengths:**
- Clean ABC with `embed`, `embed_batch`, `aembed`, `aembed_batch` — both sync and async
- Lazy model loading (only loaded on first call, not at construction)
- Helpful error messages with Python version guidance for torch/sentence-transformers failures
- Proper batch API delegation in all three providers

**Concerns:**
- `GeminiEmbeddingProvider.embed()` sets `self._dim = vec.shape[0]` after each embed call — this is a side effect in a getter-like method. If dimension is declared as a known constant per model, this mutation is unnecessary.
- `LocalEmbeddingProvider` hard-codes specific model names in `KNOWN_DIMS`. Adding a new model requires a code change. This is fine for now but limits flexibility.
- `EmbeddingProvider` doesn't define a `model_name` property — callers must reach into private attributes (`_model`, `_model_name`) to get the model identifier for cache keying (`server.py:276-281`). This is fragile.

### 2.5 Server Architecture

FastAPI with async endpoints wrapping sync DB/embedding operations via `asyncio.to_thread`.

**Strengths:**
- Clean async/sync separation: all embedding API calls and DB operations run in thread pool
- Multi-worker support via environment variable configuration
- Lifespan context manager properly saves state on shutdown
- Experiment loading (`_load_best_experiment`) allows swapping decay functions without code changes

**Concerns:**
- `global _state` is not process-safe for multi-worker mode. Each worker process has its own `_state`. The multi-worker path sets env vars and uses `factory=True` (`server.py:711-716`), which is correct for uvicorn — each worker rebuilds state from env vars. However, the `ServerState` includes a `tick_interval_seconds` and `history_interval` that are not persisted — they revert to defaults on worker restart.
- `current_tick` is stored in SQLite metadata on shutdown (`server.py:296`) but `tick_interval_seconds` and `history_interval` are not. If the server restarts, tick interval reverts to 3600s even if it was changed to 60s.
- The `/_test_embedder` parameter (`server.py:240`) is a testing escape hatch that bypasses the provider entirely. While acceptable for tests, it conflates production and test code paths.

### 2.6 Data Model

SQLite schema with WAL mode.

**Tables:**
- `memories` — core memory data with scores
- `vec_memories` — sqlite-vec virtual table for vector search
- `associations` — memory-to-memory edges with weights
- `embedding_cache` — SHA256-based embedding cache keyed by (text_hash, model)
- `activation_history` — time-series of score snapshots
- `metadata` — key-value store for engine state

**Strengths:**
- WAL mode enables concurrent reads during writes
- Schema migrations are handled gracefully (category column, embedding_cache composite PK)
- `activation_history` is properly deduplicated: only records when scores actually changed (`memory_store.py:579`)
- Cascade delete in `delete_memory` correctly removes from vec_memories, activation_history, and associations

**Concerns:**
- `associations` table has no index on `target_id` alone — `get_associated` queries predecessors then successors. If a node has many associations, the successor query (which filters on source_id) can hit the primary key index, but a standalone `target_id` lookup would be a full scan. For graphs with high out-degree or in-degree, this could be slow.
- `embedding_cache` uses SHA256 of raw text as cache key. Different providers should get different cache entries (handled by composite PK on `(text_hash, model)`), but the `model` parameter is user-provided and could be inconsistent across calls.
- `activation_history` can grow unbounded. There is no archival or downsampling strategy documented. For a long-running server with millions of ticks, this table could become very large.

---

## 3. Cross-Cutting Concerns

### 3.1 Score Synchronization

A subtle correctness issue exists in `MemoryGraph`:

1. `_emb_retrieval_scores` is a precomputed numpy array of retrieval scores, used for fast vectorized activation weighting in `query_by_similarity`
2. `_emb_retrieval_scores` is rebuilt by `_ensure_embedding_matrix()` (called at query time)
3. After `re_activate()` updates node attributes, `_sync_tick_arrays_from_graph()` is NOT called before `query_by_similarity` — only before `tick()`
4. This means if you `re_activate()` a node and immediately `query_by_similarity()` with `activation_weight > 0`, the activation-weighted ranking uses stale scores

The issue: `_ensure_embedding_matrix()` rebuilds when node count changes, not when node attributes change. So adding a node triggers rebuild, but updating a score does not.

**Impact:** Low-medium. Only affects retrieval rankings when `activation_weight > 0` and only for in-memory `MemoryGraph`. The SQLite path (`MemoryStore`) is not affected because it reads scores directly from DB at query time.

### 3.2 Concurrency Model

- `MemoryStore` uses `check_same_thread=False` on SQLite connection — correct for FastAPI's threaded execution
- WAL mode handles concurrent readers correctly
- `record_activation_history` does a `SELECT MAX(tick) ... GROUP BY` subquery per memory ID, which could be slow for large histories. The query has no index on `(memory_id, tick)` beyond the PK — the subquery scans `activation_history` grouped by memory_id which is already the PK, so it should be fine.

### 3.3 Error Handling

- Embedding provider errors (API failures) propagate as exceptions — no retry logic, no circuit breaker
- If `store.add_memory` fails mid-way through (e.g., after memories INSERT but before vec_memories INSERT), there is no transaction rollback. The `auto_commit=True` default means partial writes can occur. (SQLite's autocommit is not true autocommit; individual statements are transactional, but `add_memory` does multiple operations in sequence.)
- The `add_memories_batch` path uses `auto_commit=False` and a single `commit()` at the end — correct.

### 3.4 Embedding Cache Invalidation

The `embedding_cache` table uses `(text_hash, model)` as PK. Cache entries are:
- Added on embed operations
- Cleared when embedding dimension changes (`_ensure_vec_table`)
- Never expire based on time

If the same text is embedded with the same model across server restarts, the in-process `_embedding_cache` dict (in `MemoryGraph`) is lost on restart but the SQLite-level `embedding_cache` survives. This is correct but the two caches can be inconsistent (in-memory warm, SQLite cold, or vice versa).

---

## 4. Security Observations

- **No SQL injection**: All DB operations use parameterized queries (`?` placeholders). The `bm25_score_candidates` function builds a counter from tokenized text but never constructs SQL.
- **CORS**: Allowed origins are `http://localhost:\d+` — only localhost in dev mode. Production deployments should restrict this.
- **No authentication**: The server has no auth middleware. All `/admin/*` endpoints are unprotected. This is acceptable if deployed behind a reverse proxy with auth, but should be documented.
- **Embedding API keys**: Passed via constructor or environment variables. In multi-worker mode, API keys are in environment variables which is reasonable but not ideal for containerized deployments (use secrets management instead).

---

## 5. Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `add_memory` | O(1) | Single INSERT + one vec INSERT |
| `search` (MemoryStore) | O(k log k + m) | k=vec search k, m=BM25 docs tokenized |
| `query_by_similarity` (MemoryGraph) | O(n) build + O(k log k) query | n=nodes (build once), k=fetch_k |
| `tick()` vectorized | O(n) per tick | n=active memories |
| `tick()` SQLite | O(n) per tick | Same, but via DB |
| `record_activation_history` | O(n) | n=all memories, reads all scores |
| BM25 scoring | O(d log d) | d=docs tokenized |

**Key bottleneck:** `record_activation_history` is O(n) on every search (when `history_interval > 0`), reading all memory scores regardless of whether they changed. This could become expensive with large memory stores.

---

## 6. Observations & Recommendations

### High Priority

1. **Dead code: `soft_floor_decay_step`** — Defined but never used by `DecayEngine`. Either wire it up as a third decay mode or remove it to avoid confusion.

2. **Score synchronization in `MemoryGraph`** — The precomputed `_emb_retrieval_scores` array can go stale between `re_activate()` and `query_by_similarity()`. Consider calling `_sync_tick_arrays_from_graph()` at the start of `query_by_similarity` when activation weighting is active, or rebuild the matrix when scores change but count doesn't.

3. **Unbounded `activation_history` growth** — No archival or downsampling. Add TTL-based cleanup or downsampling strategy.

### Medium Priority

4. **`associations` table missing index** — Add `CREATE INDEX IF NOT EXISTS idx_associations_target ON associations(target_id)` for faster predecessor queries.

5. **Persist `tick_interval_seconds` and `history_interval`** — These are runtime-configurable but not persisted. Store in `metadata` table and restore on startup.

6. **Partial write risk in `add_memory`** — If `vec_memories` INSERT fails after `memories` INSERT succeeds (e.g., vec extension unavailable), the memory row exists without a vector. Add explicit transaction handling or try/recover logic.

7. **`embedding_provider.py` model name access** — `server.py` reaches into `_provider._model` / `_provider._model_name` via getattr. Add a `model_name` property to `EmbeddingProvider`.

### Low Priority

8. **`LocalEmbeddingProvider` model flexibility** — Consider loading known dimensions from environment or allowing model-specified dimensions rather than hard-coding.

9. **CORS origin restriction** — Document that production deployments must configure a specific allowed origin.

10. **No retry/circuit breaker for embedding APIs** — If the embedding provider is down, all store/search operations fail. Consider adding retry logic with exponential backoff.

---

## 7. Design Strengths

1. **Well-motivated memory model**: The ACT-R-inspired decay with impact-modulated soft floors and testing-effect reinforcement is scientifically grounded and clearly explained.

2. **Clean separation of concerns**: Decay logic, storage, embedding, and HTTP API are in separate modules with minimal cross-cutting dependencies.

3. **Hybrid retrieval is well-implemented**: The two-stage semantic + lexical pipeline with proper IDF precomputation is solid IR practice.

4. **Excellent error messages**: sqlite-vec extension loading failures include specific Python version guidance and workarounds.

5. **Schema migrations handled gracefully**: Existing DBs are migrated automatically without data loss.

6. **Comprehensive test coverage**: Tests exist for BM25, decay, memory store, server, embedding provider, and activation history.

7. **Async-first HTTP layer**: Properly uses `asyncio.to_thread` to avoid blocking the event loop, with batch embedding optimization.

---

## 8. Summary

memory-decay-core is a well-architected library implementing a sophisticated memory model. The dual-backend design (in-memory/prototype vs SQLite/production) is sound, and the hybrid retrieval pipeline combines semantic and lexical search effectively. The primary concerns are: (a) score synchronization edge cases in `MemoryGraph`, (b) unbounded history table growth, (c) dead code in `soft_floor_decay_step`, and (d) missing index on associations. These are all fixable without architectural changes.
