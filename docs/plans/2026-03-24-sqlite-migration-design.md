# SQLite + sqlite-vec Migration Design

## Problem

Current architecture uses NetworkX in-memory graph + serialized cache:
- **No persistence**: server restart = data loss
- **Full memory load**: every memory must be loaded into RAM
- **Single graph**: no multi-user support, no per-question isolation
- **Benchmark bottleneck**: MemoryBench resets + re-stores + re-ticks for every question (~8 hours for 500 questions)

## Solution

Replace NetworkX + in-memory cache with SQLite + sqlite-vec. Two deployment modes:

### Mode 1: Production (Server + SQLite backend)
```
Client → FastAPI server → SQLite DB (user_memories.db)
```
- Server wraps SQLite with HTTP API (same endpoints as today)
- DB file = persistence, instant restart
- Multi-user via `user_id` column

### Mode 2: Benchmark (Direct SQLite, no server)
```
MemoryBench provider → SQLite DB per conversation
```
- Each conversation gets its own `.db` file
- store + tick once per conversation, search N times for N questions
- Fully parallelizable — no shared state
- Temp DBs in `/tmp`, deleted after evaluation

## DB Schema

```sql
CREATE TABLE memories (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL,
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
    retrieval_count      INTEGER DEFAULT 0,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE vec_memories USING vec0(
    id        TEXT PRIMARY KEY,
    embedding float[3072]
);

CREATE TABLE associations (
    source_id    TEXT,
    target_id    TEXT,
    weight       REAL DEFAULT 0.5,
    created_tick INTEGER DEFAULT 0,
    PRIMARY KEY (source_id, target_id)
);

CREATE TABLE embedding_cache (
    text_hash  TEXT PRIMARY KEY,
    embedding  BLOB,
    model      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- metadata stores: current_tick, decay_params, etc.
```

## Component Changes

### 1. `memory_store.py` (replaces `graph.py`)

New `MemoryStore` class wrapping SQLite:

```python
class MemoryStore:
    def __init__(self, db_path: str, embedding_dim: int = 3072):
        self.db = sqlite3.connect(db_path)
        sqlite_vec.load(self.db)
        self._init_schema()

    def add_memory(self, memory_id, content, embedding, **kwargs):
        # INSERT INTO memories + INSERT INTO vec_memories

    def search(self, query_embedding, top_k=20, activation_weight=0.0):
        # SELECT from vec_memories (KNN) JOIN memories (scores)

    def clear(self, user_id=None):
        # DELETE WHERE user_id = ? (or all)

    def get_all_for_decay(self):
        # SELECT id, scores, importance, mtype WHERE created_tick <= current_tick

    def batch_update_scores(self, updates: list[tuple]):
        # executemany UPDATE memories SET scores WHERE id = ?

    def close(self):
        self.db.close()
```

### 2. `decay.py` (minimal change)

`DecayEngine` keeps the same decay math, but reads/writes via `MemoryStore`:

```python
def tick(self):
    rows = self.store.get_all_for_decay()
    updates = []
    for row in rows:
        new_r = self._compute_decay(row.retrieval_score, row.importance, row.stability_score, row.mtype)
        new_s = self._compute_decay(row.storage_score, row.importance, row.stability_score, row.mtype)
        new_stab = row.stability_score * (1 - self.params['stability_decay'])
        updates.append((new_r, new_s, new_stab, row.id))
    self.store.batch_update_scores(updates)
    self.current_tick += 1
```

### 3. `server.py` (simplified)

Server becomes a thin HTTP wrapper around `MemoryStore` + `DecayEngine`:
- No more `_state` global with in-memory graph
- `lifespan`: open DB file, create MemoryStore
- `shutdown`: just close DB connection (data already persisted)
- Embedding provider passes embeddings to MemoryStore

### 4. `embedding_provider.py` (cache in SQLite)

Replace serialized cache with `embedding_cache` table:
```python
def get_or_compute(self, text: str) -> np.ndarray:
    hash = hashlib.sha256(text.encode()).hexdigest()
    row = db.execute("SELECT embedding FROM embedding_cache WHERE text_hash=?", [hash])
    if row:
        return deserialize(row[0])
    emb = self._call_api(text)
    db.execute("INSERT INTO embedding_cache ...", [hash, serialize(emb)])
    return emb
```

### 5. MemoryBench provider (direct SQLite mode)

```python
class MemoryDecayProvider:
    async def search(self, query, options):
        cached = self.cache.get(options.containerTag)

        # Create per-conversation DB
        db_path = f"/tmp/memorybench/{options.containerTag}.db"
        store = MemoryStore(db_path)

        if not os.path.exists(db_path):
            # First question for this conversation: store + tick
            for msg in cached.messages:
                store.add_memory(msg.text, embedding, ...)
            engine = DecayEngine(store, params=self.params)
            for _ in range(simulate_ticks):
                engine.tick()

        # Search (fast — DB already has decay state)
        results = store.search(query_embedding, top_k=30)
        store.close()
        return results
```

## Migration Path

### Phase 1: Core SQLite store
- Create `memory_store.py` with MemoryStore class
- Migrate embedding cache to SQLite table
- Update `decay.py` to use MemoryStore
- Update `server.py` to use MemoryStore
- All existing API endpoints preserved

### Phase 2: MemoryBench direct mode
- Update memorybench provider to use MemoryStore directly
- Per-conversation DB files
- Parallel search support
- Remove server dependency for benchmarks

### Phase 3: Cleanup
- Remove NetworkX dependency
- Remove in-memory cache persistence code
- Update tests

## Performance Expectations

| Operation | Current | After SQLite |
|-----------|---------|-------------|
| Server start (10K memories) | ~30s (deserialize) | <1s (open file) |
| Store 1 memory | ~1ms (in-memory) | ~2ms (INSERT + WAL) |
| Search top-20 | ~5ms (numpy dot) | ~10ms (sqlite-vec KNN) |
| Tick (500 memories) | ~50ms (numpy) | ~100ms (batch UPDATE) |
| MemoryBench 500q LME | ~8 hours | ~2 hours (parallel store+tick) |
| MemoryBench 200q LoCoMo | ~3 hours | ~10 min (store+tick once per conv) |

## Dependencies

```
sqlite-vec >= 0.1.7  (pip install sqlite-vec)
```
No other new dependencies. `networkx` and `numpy` can be removed after migration.
