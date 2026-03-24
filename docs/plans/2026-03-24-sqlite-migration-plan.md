# SQLite + sqlite-vec Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace NetworkX in-memory graph with SQLite + sqlite-vec for persistence, multi-user support, and benchmark speedup.

**Architecture:** `MemoryStore` (SQLite) replaces `MemoryGraph` (NetworkX). `DecayEngine` reads/writes via `MemoryStore`. Server becomes thin HTTP wrapper. Benchmarks use direct SQLite (no server).

**Tech Stack:** Python 3.13, SQLite, sqlite-vec 0.1.7, FastAPI, numpy (for embeddings only)

---

### Task 1: Install sqlite-vec and verify

**Files:**
- Modify: `pyproject.toml` (add sqlite-vec dependency)

**Step 1: Install**

```bash
cd /Users/lit/memory-decay
.venv/bin/pip install sqlite-vec
```

**Step 2: Verify it works**

```bash
.venv/bin/python -c "
import sqlite3, sqlite_vec, struct
db = sqlite3.connect(':memory:')
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

db.execute('CREATE VIRTUAL TABLE test_vec USING vec0(embedding float[4])')
emb = struct.pack('4f', 0.1, 0.2, 0.3, 0.4)
db.execute('INSERT INTO test_vec(rowid, embedding) VALUES (1, ?)', [emb])
rows = db.execute('SELECT rowid, distance FROM test_vec WHERE embedding MATCH ? ORDER BY distance LIMIT 1', [emb]).fetchall()
print(f'OK: {rows}')
"
```
Expected: `OK: [(1, 0.0)]`

**Step 3: Add to pyproject.toml**

Add `sqlite-vec >= 0.1.7` to dependencies in `pyproject.toml`.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Add sqlite-vec dependency for vector search migration"
```

---

### Task 2: Create MemoryStore core (schema + add_memory + search)

**Files:**
- Create: `src/memory_decay/memory_store.py`
- Create: `tests/test_memory_store.py`

**Step 1: Write the failing tests**

```python
# tests/test_memory_store.py
"""Tests for SQLite-backed MemoryStore."""
import numpy as np
import pytest
from memory_decay.memory_store import MemoryStore


def _random_embedding(dim=384, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)


class TestMemoryStoreBasic:
    def test_create_in_memory(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        assert store.num_memories == 0
        store.close()

    def test_add_and_count(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "hello world", _random_embedding(384, 1),
                         user_id="u1", mtype="fact", importance=0.8)
        assert store.num_memories == 1
        store.close()

    def test_search_returns_results(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "I love baking cakes", _random_embedding(384, 1),
                         user_id="u1")
        store.add_memory("m2", "The weather is sunny", _random_embedding(384, 2),
                         user_id="u1")
        results = store.search(_random_embedding(384, 1), top_k=2)
        assert len(results) >= 1
        assert results[0]["id"] == "m1"  # closest to itself
        store.close()

    def test_clear(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "test", _random_embedding(384, 1), user_id="u1")
        cleared = store.clear()
        assert cleared == 1
        assert store.num_memories == 0
        store.close()

    def test_clear_by_user(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "a", _random_embedding(384, 1), user_id="u1")
        store.add_memory("m2", "b", _random_embedding(384, 2), user_id="u2")
        store.clear(user_id="u1")
        assert store.num_memories == 1
        store.close()

    def test_get_node(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "hello", _random_embedding(384, 1),
                         user_id="u1", mtype="fact", importance=0.9,
                         speaker="Alice", created_tick=5)
        node = store.get_node("m1")
        assert node is not None
        assert node["content"] == "hello"
        assert node["mtype"] == "fact"
        assert node["importance"] == pytest.approx(0.9)
        assert node["speaker"] == "Alice"
        assert node["created_tick"] == 5
        store.close()

    def test_search_with_activation_weight(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, 1)
        store.add_memory("m1", "high", emb, user_id="u1")
        store.add_memory("m2", "low", emb, user_id="u1")  # same embedding
        # Set m1 high retrieval, m2 low
        store.set_retrieval_score("m1", 0.9)
        store.set_retrieval_score("m2", 0.1)
        results = store.search(emb, top_k=2, activation_weight=1.0)
        assert results[0]["id"] == "m1"  # higher activation wins
        store.close()
```

**Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_memory_store.py -v
```
Expected: FAIL (module not found)

**Step 3: Implement MemoryStore**

```python
# src/memory_decay/memory_store.py
"""SQLite + sqlite-vec backed memory store."""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from typing import Optional

import numpy as np
import sqlite_vec


def _serialize_f32(vec: np.ndarray) -> bytes:
    """Serialize numpy float32 array to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32))


class MemoryStore:
    """SQLite-backed memory store with vector search via sqlite-vec.

    Drop-in replacement for MemoryGraph. Data lives on disk (or :memory:),
    no full-load needed, supports multi-user via user_id column.
    """

    def __init__(self, db_path: str, embedding_dim: int = 3072):
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id                   TEXT PRIMARY KEY,
                user_id              TEXT NOT NULL DEFAULT '',
                content              TEXT NOT NULL,
                mtype                TEXT DEFAULT 'episode',
                importance           REAL DEFAULT 0.7,
                speaker              TEXT DEFAULT '',
                created_tick         INTEGER DEFAULT 0,
                storage_score        REAL DEFAULT 1.0,
                retrieval_score      REAL DEFAULT 1.0,
                stability_score      REAL DEFAULT 0.0,
                last_activated_tick  INTEGER DEFAULT 0,
                last_reinforced_tick INTEGER DEFAULT 0,
                retrieval_count      INTEGER DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                embedding float[{self._embedding_dim}]
            );

            CREATE TABLE IF NOT EXISTS associations (
                source_id    TEXT,
                target_id    TEXT,
                weight       REAL DEFAULT 0.5,
                created_tick INTEGER DEFAULT 0,
                PRIMARY KEY (source_id, target_id)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.db.commit()

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: np.ndarray,
        *,
        user_id: str = "",
        mtype: str = "episode",
        importance: float = 0.7,
        speaker: str = "",
        created_tick: int = 0,
        associations: list[tuple[str, float]] | None = None,
    ) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO memories
               (id, user_id, content, mtype, importance, speaker, created_tick,
                last_activated_tick, last_reinforced_tick)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, user_id, content, mtype, importance, speaker,
             created_tick, created_tick, created_tick),
        )
        # sqlite-vec uses rowid internally; we map memory_id to rowid
        # Store the rowid mapping
        rowid = self.db.execute(
            "SELECT rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        self.db.execute(
            "INSERT OR REPLACE INTO vec_memories(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize_f32(embedding)),
        )
        if associations:
            for target_id, weight in associations:
                self.db.execute(
                    "INSERT OR IGNORE INTO associations VALUES (?, ?, ?, ?)",
                    (memory_id, target_id, weight, created_tick),
                )
                self.db.execute(
                    "INSERT OR IGNORE INTO associations VALUES (?, ?, ?, ?)",
                    (target_id, memory_id, weight, created_tick),
                )
        self.db.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        current_tick: int | None = None,
        activation_weight: float = 0.0,
        user_id: str | None = None,
    ) -> list[dict]:
        query_bytes = _serialize_f32(query_embedding)

        # KNN search via sqlite-vec
        rows = self.db.execute(
            """SELECT v.rowid, v.distance, m.*
               FROM vec_memories v
               JOIN memories m ON m.rowid = v.rowid
               WHERE v.embedding MATCH ?
               ORDER BY v.distance
               LIMIT ?""",
            (query_bytes, top_k * 3),  # fetch extra for filtering/reranking
        ).fetchall()

        results = []
        for row in rows:
            if current_tick is not None and row["created_tick"] > current_tick:
                continue
            if user_id is not None and row["user_id"] != user_id:
                continue

            # Convert distance to similarity (sqlite-vec returns L2 distance)
            # For cosine: similarity = 1 - distance (if using cosine distance)
            # sqlite-vec default is L2; we normalize to 0-1 similarity
            distance = row["distance"]
            similarity = max(1.0 - distance, 0.0)

            # Apply activation weight
            if activation_weight > 0:
                retrieval_score = max(float(row["retrieval_score"]), 0.0)
                similarity *= retrieval_score ** activation_weight

            results.append({
                "id": row["id"],
                "text": row["content"],
                "score": round(similarity, 4),
                "storage_score": round(float(row["storage_score"]), 4),
                "retrieval_score": round(float(row["retrieval_score"]), 4),
                "category": row["mtype"],
                "created_tick": row["created_tick"],
                "speaker": row["speaker"] or "",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_node(self, memory_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_for_decay(self, current_tick: int) -> list[dict]:
        rows = self.db.execute(
            """SELECT id, retrieval_score, storage_score, stability_score,
                      importance, mtype
               FROM memories WHERE created_tick <= ?""",
            (current_tick,),
        ).fetchall()
        return [dict(r) for r in rows]

    def batch_update_scores(self, updates: list[tuple]) -> None:
        """Update scores in bulk. Each tuple: (retrieval, storage, stability, id)."""
        self.db.executemany(
            """UPDATE memories
               SET retrieval_score = ?, storage_score = ?, stability_score = ?
               WHERE id = ?""",
            updates,
        )
        self.db.commit()

    def set_retrieval_score(self, memory_id: str, score: float) -> None:
        self.db.execute(
            "UPDATE memories SET retrieval_score = ? WHERE id = ?",
            (score, memory_id),
        )
        self.db.commit()

    def set_storage_score(self, memory_id: str, score: float) -> None:
        self.db.execute(
            "UPDATE memories SET storage_score = ? WHERE id = ?",
            (score, memory_id),
        )
        self.db.commit()

    def set_activation(self, memory_id: str, score: float) -> None:
        self.db.execute(
            "UPDATE memories SET retrieval_score = ? WHERE id = ?",
            (score, memory_id),
        )
        self.db.commit()

    def clear(self, user_id: str | None = None) -> int:
        if user_id:
            # Get rowids to delete from vec_memories
            rowids = [r[0] for r in self.db.execute(
                "SELECT rowid FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()]
            count = self.db.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            ).rowcount
            for rid in rowids:
                self.db.execute("DELETE FROM vec_memories WHERE rowid = ?", (rid,))
        else:
            count = self.db.execute("DELETE FROM memories").rowcount
            self.db.execute("DELETE FROM vec_memories")
            self.db.execute("DELETE FROM associations")
        self.db.commit()
        return count

    @property
    def num_memories(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def get_metadata(self, key: str, default: str = "") -> str:
        row = self.db.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def set_metadata(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_memory_store.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/memory_decay/memory_store.py tests/test_memory_store.py
git commit -m "Add MemoryStore: SQLite + sqlite-vec backed memory storage"
```

---

### Task 3: Update DecayEngine to support MemoryStore

**Files:**
- Modify: `src/memory_decay/decay.py`
- Create: `tests/test_decay_sqlite.py`

**Step 1: Write failing test**

```python
# tests/test_decay_sqlite.py
"""Test DecayEngine with SQLite-backed MemoryStore."""
import numpy as np
import pytest
from memory_decay.memory_store import MemoryStore
from memory_decay.decay import DecayEngine


def _emb(seed=42, dim=384):
    return np.random.RandomState(seed).randn(dim).astype(np.float32)


def test_decay_with_store():
    store = MemoryStore(":memory:", embedding_dim=384)
    store.add_memory("m1", "hello", _emb(1), importance=0.7, mtype="episode", created_tick=0)
    store.add_memory("m2", "world", _emb(2), importance=0.3, mtype="fact", created_tick=0)

    engine = DecayEngine(store=store, params={"lambda_fact": 0.05, "lambda_episode": 0.08,
        "alpha": 0.5, "stability_weight": 0.8, "stability_decay": 0.01,
        "stability_cap": 1.0})

    # Tick 10 times
    for _ in range(10):
        engine.tick()

    node = store.get_node("m1")
    assert node["retrieval_score"] < 1.0  # decayed
    assert node["retrieval_score"] > 0.0  # not zero
    store.close()
```

**Step 2: Modify DecayEngine to accept MemoryStore**

Add a `store` parameter to `DecayEngine.__init__`. When `store` is provided, use it instead of `self._graph`. The `tick()` method should:
1. Call `store.get_all_for_decay(current_tick)`
2. Compute decay for each row
3. Call `store.batch_update_scores(updates)`

Keep backward compatibility: if `graph` is passed, use existing behavior.

Key changes in `decay.py`:
- `__init__`: accept `store: MemoryStore | None = None` alongside `graph`
- `tick()`: add SQLite path using `store.get_all_for_decay()` + `store.batch_update_scores()`
- `_compute_decay()`: no change (pure math)
- `custom_decay_fn`: works as-is (called per-row)

**Step 3: Run tests**

```bash
.venv/bin/pytest tests/test_decay_sqlite.py tests/test_core.py -v
```
Expected: All PASS (both old and new tests)

**Step 4: Commit**

```bash
git add src/memory_decay/decay.py tests/test_decay_sqlite.py
git commit -m "Support MemoryStore in DecayEngine alongside MemoryGraph"
```

---

### Task 4: Embedding cache in SQLite

**Files:**
- Modify: `src/memory_decay/memory_store.py` (add embedding cache methods)
- Create: `tests/test_embedding_cache_sqlite.py`

**Step 1: Write failing test**

```python
# tests/test_embedding_cache_sqlite.py
import numpy as np
from memory_decay.memory_store import MemoryStore


def test_embedding_cache_roundtrip():
    store = MemoryStore(":memory:", embedding_dim=384)
    emb = np.random.randn(384).astype(np.float32)
    store.cache_embedding("hello world", emb, model="test-model")
    cached = store.get_cached_embedding("hello world")
    assert cached is not None
    np.testing.assert_array_almost_equal(cached, emb)
    store.close()

def test_embedding_cache_miss():
    store = MemoryStore(":memory:", embedding_dim=384)
    assert store.get_cached_embedding("not cached") is None
    store.close()
```

**Step 2: Add to MemoryStore**

Add `embedding_cache` table to schema, plus `cache_embedding()` and `get_cached_embedding()` methods. Use `hashlib.sha256` for text hashing, `struct.pack` for embedding serialization.

**Step 3: Run tests and commit**

```bash
.venv/bin/pytest tests/test_embedding_cache_sqlite.py -v
git add -A && git commit -m "Add embedding cache to MemoryStore"
```

---

### Task 5: Update server.py to use MemoryStore

**Files:**
- Modify: `src/memory_decay/server.py`
- Modify: `tests/test_server.py`

**Step 1: Read existing test_server.py for test patterns**

Check `tests/test_server.py` for existing test structure to maintain compatibility.

**Step 2: Modify server.py**

Replace `_state.graph` (MemoryGraph) with `_state.store` (MemoryStore):
- `lifespan`: create MemoryStore with db_path from args
- `/store`: call `store.add_memory()` with embedding from provider
- `/search`: call `store.search()` with query embedding from provider
- `/tick`: call `engine.tick()` (engine now wraps store)
- `/reset`: call `store.clear()`
- `/health`, `/stats`: read from store
- Remove: `persistence.py` dependency (SQLite IS the persistence)

Keep embedding provider as-is — it computes embeddings, MemoryStore stores them.

**Step 3: Run server tests**

```bash
.venv/bin/pytest tests/test_server.py -v
```

**Step 4: Manual smoke test**

```bash
.venv/bin/python -m memory_decay.server --port 8100 --db-path /tmp/test.db \
  --embedding-provider openai --embedding-api-key "$OPENAI_API_KEY" \
  --embedding-model text-embedding-3-large &
sleep 3
curl -s localhost:8100/health
curl -s localhost:8100/store -H "Content-Type: application/json" -d '{"text":"hello"}'
curl -s localhost:8100/search -H "Content-Type: application/json" -d '{"query":"hello","top_k":5}'
pkill -f "memory_decay.server"
```

**Step 5: Commit**

```bash
git add src/memory_decay/server.py tests/test_server.py
git commit -m "Switch server to SQLite-backed MemoryStore"
```

---

### Task 6: Verify existing benchmarks still work

**Step 1: Run MemoryBench with server mode (existing flow)**

```bash
# Start server with SQLite backend
.venv/bin/python -m memory_decay.server --port 8100 --db-path /tmp/bench.db \
  --embedding-provider openai --embedding-api-key "$OPENAI_API_KEY" \
  --embedding-model text-embedding-3-large \
  --experiment-dir experiments/exp_bench_0001 &

# Run quick benchmark
cd ~/memorybench
OPENAI_API_KEY="$OPENAI_API_KEY" MEMORY_DECAY_AGENT_MODE=1 \
  bun run src/index.ts run -p memory-decay -b convomem \
  -j gpt-4o-mini -m sonnet -s 2 --sample-type random \
  -r sqlite-smoke-test --force
```

**Step 2: Compare accuracy with pre-migration results**

Verify accuracy is similar (±5%) to previous runs. SQLite search (L2/cosine) may give slightly different rankings than numpy dot product.

**Step 3: Commit any fixes**

---

### Task 7: Cleanup — remove NetworkX dependency

**Files:**
- Delete: `src/memory_decay/persistence.py`
- Modify: `src/memory_decay/__init__.py` (update exports)
- Modify: `pyproject.toml` (remove networkx)
- Keep: `src/memory_decay/graph.py` (mark deprecated, keep for reference)

**Step 1: Update __init__.py**

Export `MemoryStore` alongside `MemoryGraph` (deprecated).

**Step 2: Update imports in files that use MemoryGraph**

Files to check: `evaluator.py`, `main.py`, `cache_builder.py`, `runner.py`, `cross_validator.py`, `multi_runner.py`, `calibration_pipeline.py`

**Step 3: Run full test suite**

```bash
.venv/bin/pytest tests/ -v --ignore=tests/test_self_evolution.py
```

**Step 4: Commit**

```bash
git commit -m "Deprecate MemoryGraph, export MemoryStore as primary store"
```

---

## Execution Notes

- **Phase 2 (MemoryBench direct mode)** is a separate plan — do after Phase 1 is stable
- Keep `graph.py` around initially for backward compat; delete after Phase 2
- sqlite-vec uses **L2 distance** by default; pre-normalize embeddings for cosine behavior
- WAL mode (`PRAGMA journal_mode=WAL`) enables concurrent reads during tick/search
