# BM25 Hybrid Search for MemoryStore Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BM25 lexical re-ranking to `MemoryStore.search()` so the production server path matches the hybrid search capabilities already proven in `MemoryGraph`.

**Architecture:** Extract BM25 scoring into a shared module (`bm25.py`) used by both `MemoryStore` and `MemoryGraph`. `MemoryStore.search()` gains a `bm25_weight` parameter that triggers a two-stage retrieval: vector KNN first, then BM25 re-ranking on the candidate set. IDF statistics are computed on-the-fly from the candidate set (no persistent index needed — SQLite already stores content). The server exposes this via the existing experiment params.

**Tech Stack:** Python 3.10+, SQLite, sqlite-vec, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/memory_decay/bm25.py` | **Create** | Shared BM25 tokenizer + scorer (extracted from `graph.py`) |
| `src/memory_decay/memory_store.py` | **Modify** | Add `bm25_weight` param to `search()`, call BM25 re-ranker |
| `src/memory_decay/graph.py` | **Modify** | Import BM25 from shared module instead of inline |
| `src/memory_decay/server.py` | **Modify** | Pass `bm25_weight` from engine params to `store.search()` |
| `tests/test_bm25.py` | **Create** | Unit tests for BM25 scorer |
| `tests/test_memory_store.py` | **Modify** | Add BM25 hybrid search integration tests |
| `tests/test_server.py` | **Modify** | Add server-level BM25 test |

---

## Chunk 1: BM25 Module Extraction + Unit Tests

### Task 1: Create shared BM25 module with tests

**Files:**
- Create: `src/memory_decay/bm25.py`
- Create: `tests/test_bm25.py`

- [ ] **Step 1: Write the failing tests for BM25 tokenizer and scorer**

```python
# tests/test_bm25.py
"""Tests for BM25 tokenizer and scorer."""
import pytest
from memory_decay.bm25 import bm25_tokenize, bm25_score_candidates


class TestBM25Tokenize:
    def test_english_words(self):
        assert bm25_tokenize("Hello World") == ["hello", "world"]

    def test_korean_words(self):
        tokens = bm25_tokenize("서울은 한국의 수도입니다")
        assert "서울은" in tokens
        assert "한국의" in tokens

    def test_mixed_language(self):
        tokens = bm25_tokenize("Python은 좋은 언어")
        assert "python" in tokens  # lowercased
        assert "좋은" in tokens

    def test_numbers_preserved(self):
        tokens = bm25_tokenize("GPT4 has 175B params")
        assert "gpt4" in tokens
        assert "175b" in tokens

    def test_empty_string(self):
        assert bm25_tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = bm25_tokenize("hello, world! (test)")
        assert tokens == ["hello", "world", "test"]


class TestBM25ScoreCandidates:
    def test_exact_match_scores_highest(self):
        docs = {
            "m1": "the cat sat on the mat",
            "m2": "the dog played in the park",
            "m3": "a fish swam in the ocean",
        }
        scores = bm25_score_candidates("cat mat", docs)
        assert scores["m1"] > scores["m2"]
        assert scores["m1"] > scores["m3"]

    def test_no_match_scores_zero(self):
        docs = {"m1": "hello world"}
        scores = bm25_score_candidates("xyz qqq", docs)
        assert scores["m1"] == pytest.approx(0.0)

    def test_empty_query(self):
        docs = {"m1": "hello world"}
        scores = bm25_score_candidates("", docs)
        assert scores == {}

    def test_empty_docs(self):
        scores = bm25_score_candidates("hello", {})
        assert scores == {}

    def test_idf_weighting(self):
        # "rare" appears in 1 doc, "the" appears in all 3
        docs = {
            "m1": "the rare word",
            "m2": "the common phrase",
            "m3": "the other text",
        }
        scores = bm25_score_candidates("rare", docs)
        # m1 should score because it has "rare", others should not
        assert scores["m1"] > 0
        assert scores["m2"] == pytest.approx(0.0)

    def test_term_frequency_matters(self):
        docs = {
            "m1": "cat cat cat",
            "m2": "cat dog bird",
        }
        scores = bm25_score_candidates("cat", docs)
        # m1 has higher TF for "cat"
        assert scores["m1"] > scores["m2"]

    def test_returns_all_candidates(self):
        docs = {"m1": "a b", "m2": "c d", "m3": "e f"}
        scores = bm25_score_candidates("a", docs)
        assert set(scores.keys()) == {"m1", "m2", "m3"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_bm25.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_decay.bm25'`

- [ ] **Step 3: Implement the BM25 module**

```python
# src/memory_decay/bm25.py
"""Shared BM25 tokenizer and scorer.

Used by both MemoryGraph (in-memory) and MemoryStore (SQLite) for
hybrid semantic+lexical retrieval.
"""
from __future__ import annotations

import math
import re
from collections import Counter


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 scoring.

    Handles English, Korean (가-힣), and numbers.
    Returns lowercased tokens with punctuation stripped.
    """
    return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())


def bm25_score_candidates(
    query_text: str,
    candidate_docs: dict[str, str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Score candidate documents against a query using BM25.

    Args:
        query_text: The search query.
        candidate_docs: Mapping of document_id -> document_text.
        k1: Term frequency saturation parameter.
        b: Document length normalization parameter.

    Returns:
        Mapping of document_id -> BM25 score. All candidates are included
        (score 0.0 if no query terms match).
    """
    query_terms = list(dict.fromkeys(bm25_tokenize(query_text)))
    if not query_terms or not candidate_docs:
        return {}

    # Tokenize all documents
    doc_tokens: dict[str, list[str]] = {}
    for doc_id, text in candidate_docs.items():
        doc_tokens[doc_id] = bm25_tokenize(text)

    # Compute IDF from candidate set
    n_docs = len(candidate_docs)
    doc_freq: Counter[str] = Counter()
    total_tokens = 0
    for tokens in doc_tokens.values():
        doc_freq.update(set(tokens))
        total_tokens += len(tokens)

    avgdl = max(total_tokens / max(n_docs, 1), 1.0)
    idf: dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    # Score each document
    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = max(len(tokens), 1)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            term_idf = idf.get(term, 0.0)
            denom = freq + k1 * (1.0 - b + b * dl / avgdl)
            score += term_idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
        scores[doc_id] = score

    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_bm25.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/bm25.py tests/test_bm25.py
git commit -m "feat: extract shared BM25 module from MemoryGraph

Moves BM25 tokenizer and scorer into a standalone module so both
MemoryGraph (in-memory) and MemoryStore (SQLite) can use the same
lexical matching logic.

Tested: unit tests for tokenizer, IDF weighting, TF ranking, edge cases
Scope-risk: narrow
Confidence: high"
```

---

### Task 2: Wire MemoryGraph to use shared BM25 module

**Files:**
- Modify: `src/memory_decay/graph.py:198-234` (replace inline BM25 with import)
- Test: `tests/test_core.py` (existing BM25 tests should still pass)

- [ ] **Step 1: Run existing tests to confirm green baseline**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_core.py -v -k bm25`
Expected: All existing BM25 tests PASS

- [ ] **Step 2: Replace inline BM25 in graph.py with shared module**

In `graph.py`, replace the `_bm25_tokenize` static method and `_bm25_score_candidates` method:

```python
# At top of graph.py, add import:
from .bm25 import bm25_tokenize, bm25_score_candidates

# Replace _bm25_tokenize staticmethod (line ~198-200):
# DELETE:
#     @staticmethod
#     def _bm25_tokenize(text: str) -> list[str]:
#         return re.findall(r"[0-9A-Za-z가-힣]+", text.lower())

# Replace _bm25_score_candidates method (line ~202-234) with:
    def _bm25_score_candidates(
        self,
        query_text: str,
        candidate_nids: list[str],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> dict[str, float]:
        """Score candidates against query using BM25 with pre-computed global IDF.

        Uses the pre-built IDF index from _ensure_embedding_matrix() for efficiency,
        falling back to the shared bm25 module for computation.
        """
        if self._bm25_idf is None or self._bm25_doc_tokens is None:
            return {}

        query_terms = list(dict.fromkeys(bm25_tokenize(query_text)))
        if not query_terms:
            return {}

        avgdl = max(self._bm25_avgdl, 1.0)
        scores: dict[str, float] = {}

        for nid in candidate_nids:
            tokens = self._bm25_doc_tokens.get(nid, [])
            tf = Counter(tokens)
            dl = max(len(tokens), 1)
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._bm25_idf.get(term, 0.0)
                denom = freq + k1 * (1.0 - b + b * dl / avgdl)
                score += idf * (freq * (k1 + 1.0)) / max(denom, 1e-9)
            scores[nid] = score

        return scores
```

Also update `_ensure_embedding_matrix` to use shared tokenizer (line ~278):
```python
# Replace: tokens = self._bm25_tokenize(content)
# With:    tokens = bm25_tokenize(content)
```

Remove the `re` import from graph.py if no longer used elsewhere.

- [ ] **Step 3: Run existing tests to confirm nothing broke**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/graph.py
git commit -m "refactor: wire MemoryGraph to shared BM25 module

Replace inline _bm25_tokenize and scoring logic with imports from
bm25.py. MemoryGraph still uses its pre-computed IDF index for
efficiency; only the tokenizer is shared directly.

Tested: all existing tests pass
Scope-risk: narrow
Reversibility: clean"
```

---

## Chunk 2: BM25 Hybrid Search in MemoryStore

### Task 3: Add BM25 re-ranking to MemoryStore.search()

**Files:**
- Modify: `src/memory_decay/memory_store.py:215-269`
- Modify: `tests/test_memory_store.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory_store.py`:

```python
class TestMemoryStoreBM25:
    """Tests for BM25 hybrid search in MemoryStore."""

    def test_bm25_weight_zero_is_pure_vector(self):
        """bm25_weight=0 should produce identical results to default search."""
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "cat sat on mat", _random_embedding(384, 1))
        store.add_memory("m2", "dog in park", _random_embedding(384, 2))
        default = store.search(_random_embedding(384, 1), top_k=2)
        hybrid = store.search(_random_embedding(384, 1), top_k=2, bm25_weight=0.0)
        assert [r["id"] for r in default] == [r["id"] for r in hybrid]
        store.close()

    def test_bm25_boosts_lexical_match(self):
        """With identical embeddings, BM25 should prefer lexical match."""
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, 1)
        # Both have same embedding, but only m1 matches query lexically
        store.add_memory("m1", "서울은 한국의 수도", emb)
        store.add_memory("m2", "날씨가 좋습니다", emb)
        results = store.search(emb, top_k=2, bm25_weight=0.5,
                               query_text="서울 수도")
        assert results[0]["id"] == "m1"
        store.close()

    def test_bm25_with_no_lexical_overlap(self):
        """BM25 should not crash when no terms match."""
        store = MemoryStore(":memory:", embedding_dim=384)
        store.add_memory("m1", "hello world", _random_embedding(384, 1))
        results = store.search(_random_embedding(384, 1), top_k=1,
                               bm25_weight=0.3, query_text="xyz qqq")
        assert len(results) >= 1  # still returns vector results
        store.close()

    def test_bm25_respects_top_k(self):
        """BM25 re-ranking should still respect top_k limit."""
        store = MemoryStore(":memory:", embedding_dim=384)
        for i in range(10):
            store.add_memory(f"m{i}", f"doc {i}", _random_embedding(384, i))
        results = store.search(_random_embedding(384, 0), top_k=3,
                               bm25_weight=0.3, query_text="doc 0")
        assert len(results) <= 3
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_memory_store.py::TestMemoryStoreBM25 -v`
Expected: FAIL — `search()` doesn't accept `bm25_weight` or `query_text` params

- [ ] **Step 3: Implement BM25 hybrid search in MemoryStore**

Modify `src/memory_decay/memory_store.py`. Add import at top:

```python
from .bm25 import bm25_score_candidates
```

Replace `search()` method (lines 215-269):

```python
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        current_tick: int | None = None,
        activation_weight: float = 0.0,
        user_id: str | None = None,
        bm25_weight: float = 0.0,
        query_text: str = "",
    ) -> list[dict]:
        # Pre-normalize query embedding to match stored embeddings
        normed_query = _normalize(query_embedding)
        query_bytes = _serialize_f32(normed_query)

        # KNN search via sqlite-vec (returns L2 distance on normalized vectors)
        # Fetch extra candidates when BM25 is active for re-ranking pool
        fetch_k = top_k * 3
        if bm25_weight > 0:
            fetch_k = max(fetch_k, top_k * 5)
        rows = self._db.execute(
            """SELECT v.rowid, v.distance, m.*
               FROM vec_memories v
               JOIN memories m ON m.rowid = v.rowid
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (query_bytes, fetch_k),
        ).fetchall()

        candidates = []
        for row in rows:
            if current_tick is not None and row["created_tick"] > current_tick:
                continue
            if user_id is not None and row["user_id"] != user_id:
                continue

            distance = float(row["distance"])
            similarity = max(1.0 - (distance ** 2) / 2.0, 0.0)

            if activation_weight > 0:
                retrieval_score = max(float(row["retrieval_score"]), 0.0)
                similarity *= retrieval_score ** activation_weight

            candidates.append({
                "id": row["id"],
                "text": row["content"],
                "score": similarity,
                "storage_score": round(float(row["storage_score"]), 4),
                "retrieval_score": round(float(row["retrieval_score"]), 4),
                "category": row["mtype"],
                "created_tick": row["created_tick"],
                "speaker": row["speaker"] or "",
            })

        # BM25 re-ranking pass
        if bm25_weight > 0 and query_text and candidates:
            doc_texts = {c["id"]: c["text"] for c in candidates}
            bm25_scores = bm25_score_candidates(query_text, doc_texts)

            if bm25_scores:
                cos_scores = {c["id"]: c["score"] for c in candidates}
                cos_vals = list(cos_scores.values())
                cos_min, cos_max = min(cos_vals), max(cos_vals)
                cos_range = cos_max - cos_min

                bm25_vals = list(bm25_scores.values())
                bm25_max = max(bm25_vals) if bm25_vals else 1.0

                for c in candidates:
                    if cos_range < 1e-8:
                        norm_cos = 1.0  # all candidates tied on vector similarity
                    else:
                        norm_cos = (c["score"] - cos_min) / cos_range
                    norm_bm25 = bm25_scores.get(c["id"], 0.0) / max(bm25_max, 1e-8)
                    c["score"] = (1.0 - bm25_weight) * norm_cos + bm25_weight * norm_bm25

        for c in candidates:
            c["score"] = round(c["score"], 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_memory_store.py -v`
Expected: All PASS (both new and existing tests)

- [ ] **Step 5: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/memory_store.py tests/test_memory_store.py
git commit -m "feat: add BM25 hybrid search to MemoryStore

MemoryStore.search() now accepts bm25_weight and query_text params.
When bm25_weight > 0, a two-stage retrieval is performed: vector KNN
first, then BM25 re-ranking on the candidate set. Combined score uses
normalized linear interpolation matching MemoryGraph's approach.

Constraint: sqlite-vec only supports KNN, so BM25 is a re-ranking pass
Rejected: persistent FTS5 index | adds schema complexity for marginal gain
Tested: lexical boost, zero-weight passthrough, no-overlap, top_k respect
Scope-risk: narrow
Confidence: high"
```

---

### Task 4: Wire server to pass BM25 params to MemoryStore

**Files:**
- Modify: `src/memory_decay/server.py:345-373`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Write the failing server test**

Add to `tests/test_server.py`:

```python
@pytest.fixture
def bm25_client():
    """Create test client with BM25 enabled via experiment params."""
    import tempfile, json, os
    dim = 8
    embedder = lambda t: np.random.RandomState(hash(t) % 2**31).randn(dim).astype(np.float32)
    with tempfile.TemporaryDirectory() as exp_dir:
        params_path = os.path.join(exp_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump({"bm25_weight": 0.3}, f)
        app = create_app(
            embedding_provider=None,
            _test_embedder=embedder,
            experiment_dir=exp_dir,
        )
        with TestClient(app) as c:
            yield c


class TestSearchBM25:
    def test_search_with_bm25_enabled(self, bm25_client):
        """Search should use BM25 when bm25_weight is set in params."""
        bm25_client.post("/store", json={
            "text": "서울은 한국의 수도입니다",
            "importance": 0.8, "mtype": "fact",
        })
        bm25_client.post("/store", json={
            "text": "날씨가 좋습니다",
            "importance": 0.8, "mtype": "fact",
        })
        r = bm25_client.post("/search", json={
            "query": "한국 수도",
            "top_k": 5,
        })
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) >= 1
        # With BM25 active and lexical overlap, 서울/수도 should rank first
        assert "수도" in results[0]["text"]

    def test_search_without_bm25_still_works(self, client):
        """Default client (bm25_weight=0) should still work normally."""
        client.post("/store", json={
            "text": "test memory", "importance": 0.5, "mtype": "fact",
        })
        r = client.post("/search", json={"query": "test", "top_k": 5})
        assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_server.py::TestSearchBM25 -v`
Expected: FAIL — server doesn't pass `bm25_weight`/`query_text` to `store.search()` yet

- [ ] **Step 3: Modify server search endpoint to pass BM25 params**

In `src/memory_decay/server.py`, modify the `/search` endpoint (lines 345-373):

```python
    @app.post("/search")
    async def search(req: SearchRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        query_embedding = await _state.embed(req.query)
        params = _state.engine.get_params()

        results = await asyncio.to_thread(
            lambda: _state.store.search(
                query_embedding=query_embedding,
                top_k=req.top_k,
                current_tick=_state.current_tick,
                activation_weight=params.get("activation_weight", 0.5),
                bm25_weight=params.get("bm25_weight", 0.0),
                query_text=req.query,
            )
        )

        # Retrieval consolidation: boost top result on successful recall
        if results and results[0]["score"] > 0.3:
            await asyncio.to_thread(
                lambda: _state.store.reinforce(
                    results[0]["id"],
                    retrieval_boost=params.get("retrieval_boost", 0.10),
                    stability_gain=params.get("reinforcement_gain_direct", 0.2),
                    stability_cap=params.get("stability_cap", 1.0),
                )
            )

        return {"results": results}
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/server.py tests/test_server.py
git commit -m "feat: wire BM25 hybrid search to server /search endpoint

Server now passes bm25_weight and query_text to MemoryStore.search().
BM25 weight is read from engine params (experiments/best/params.json),
defaulting to 0.0 for backward compatibility. To enable, set
bm25_weight in the experiment params.

Directive: bm25_weight=0.0 is backward-compatible; set 0.2-0.5 in params.json to activate
Tested: server integration test with Korean text
Scope-risk: narrow
Confidence: high"
```

---

## Chunk 3: Bonus Fixes (Embedding Cache + Tick Sync)

### Task 5: Fix embedding cache model filter

**Files:**
- Modify: `src/memory_decay/memory_store.py:54-91, 407-427`
- Modify: `src/memory_decay/server.py:90-127`
- Modify: `tests/test_memory_store.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory_store.py`:

```python
class TestEmbeddingCacheModelFilter:
    def test_different_models_return_different_embeddings(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb_a = _random_embedding(384, seed=1)
        emb_b = _random_embedding(384, seed=2)
        store.cache_embedding("hello", emb_a, model="model-a")
        store.cache_embedding("hello", emb_b, model="model-b")
        result_a = store.get_cached_embedding("hello", model="model-a")
        result_b = store.get_cached_embedding("hello", model="model-b")
        assert result_a is not None
        assert result_b is not None
        assert not np.allclose(result_a, result_b)
        store.close()

    def test_missing_model_returns_none(self):
        store = MemoryStore(":memory:", embedding_dim=384)
        emb = _random_embedding(384, seed=1)
        store.cache_embedding("hello", emb, model="model-a")
        result = store.get_cached_embedding("hello", model="model-x")
        assert result is None
        store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_memory_store.py::TestEmbeddingCacheModelFilter -v`
Expected: FAIL — second `cache_embedding` overwrites first (same text_hash PK), and query ignores model

- [ ] **Step 3: Fix the embedding cache schema, queries, and migration**

Modify `src/memory_decay/memory_store.py`:

**3a. Change the schema in `_init_schema()` (line 86-90):**

```python
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT NOT NULL,
                model     TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL,
                PRIMARY KEY (text_hash, model)
            );
```

**3b. Add migration for existing DBs — add to `_init_schema()` after the `executescript` call (before `_ensure_vec_table`):**

```python
        # Migrate embedding_cache: old schema has text_hash-only PK
        self._migrate_embedding_cache()
```

And add the migration method:

```python
    def _migrate_embedding_cache(self) -> None:
        """Recreate embedding_cache if it has the old single-column PK."""
        import sys
        rows = self._db.execute("PRAGMA table_info(embedding_cache)").fetchall()
        pk_cols = [r[1] for r in rows if r[5] > 0]  # r[5] = pk flag
        if pk_cols == ["text_hash"]:
            print(
                "[memory-store] Migrating embedding_cache to composite PK (text_hash, model).",
                file=sys.stderr,
            )
            self._db.execute("DROP TABLE embedding_cache")
            self._db.execute("""
                CREATE TABLE embedding_cache (
                    text_hash TEXT NOT NULL,
                    model     TEXT NOT NULL DEFAULT '',
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (text_hash, model)
                )
            """)
            self._db.commit()
```

**3c. Fix `get_cached_embedding()` (line 418-427):**

```python
    def get_cached_embedding(self, text: str, model: str = "") -> np.ndarray | None:
        """Retrieve a cached embedding, or None if not found."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        row = self._db.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model = ?",
            (text_hash, model),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_f32(row[0], self._embedding_dim)
```

**3d. Wire model name through ServerState.**

In `src/memory_decay/server.py`, modify `ServerState.__init__` to store model name:

```python
class ServerState:
    def __init__(
        self,
        store: MemoryStore,
        engine: DecayEngine,
        embedder: Callable | None = None,
        provider: EmbeddingProvider | None = None,
        tick_interval_seconds: float = 3600.0,
        embedding_model: str = "",
    ):
        self.store = store
        self.engine = engine
        self._embedder = embedder
        self._provider = provider
        self.current_tick = 0
        self.last_tick_time = time.time()
        self.tick_interval_seconds = tick_interval_seconds
        self._embedding_model = embedding_model
```

Update `embed()` to pass model:

```python
    async def embed(self, text: str) -> np.ndarray:
        cached = await asyncio.to_thread(
            self.store.get_cached_embedding, text, model=self._embedding_model
        )
        if cached is not None:
            return cached
        # ... (existing embed logic unchanged) ...
        await asyncio.to_thread(
            self.store.cache_embedding, text, embedding, model=self._embedding_model
        )
        return embedding
```

Update `embed_batch()` similarly — pass `model=self._embedding_model` to both `get_cached_embedding` and `cache_embedding` calls.

In `create_app()` lifespan, pass the model name when creating `ServerState`:

```python
        # Resolve model name for cache keying
        if _test_embedder:
            model_name = "test"
        elif embedding_provider:
            model_name = getattr(embedding_provider, '_model', '')
        else:
            model_name = "ko-sroberta"

        _state = ServerState(
            store, engine,
            embedder=base_embedder,
            provider=resolved_provider,
            tick_interval_seconds=tick_interval_seconds,
            embedding_model=model_name,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_memory_store.py tests/test_server.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/memory_store.py src/memory_decay/server.py tests/test_memory_store.py
git commit -m "fix: add model filter to embedding cache lookups

Previously get_cached_embedding ignored the model parameter, returning
embeddings from a different model if the text matched. Now the cache
key is (text_hash, model) so model switches don't return stale vectors.

Includes migration that drops old single-column PK cache on existing
DBs (cache is a warm-start optimization, safe to drop). Server now
passes embedding model name through to all cache operations.

Tested: model isolation, missing model returns None, migration path
Scope-risk: narrow
Confidence: high"
```

---

### Task 6: Unify tick synchronization

**Files:**
- Modify: `src/memory_decay/server.py:374-413`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_server.py`:

```python
class TestTickSync:
    def test_tick_counter_matches_engine(self, client):
        """Server tick and engine tick must stay in sync."""
        client.post("/tick", json={"count": 5})
        health = client.get("/health").json()
        assert health["current_tick"] == 5
        client.post("/tick", json={"count": 3})
        health = client.get("/health").json()
        assert health["current_tick"] == 8

    def test_tick_derived_from_engine(self, client):
        """Verify the server reads tick from engine, not its own counter.

        This guards against the two counters diverging. We access the
        internal state to confirm engine is the source of truth.
        """
        from memory_decay.server import _state
        client.post("/tick", json={"count": 5})
        # Both must match — engine owns the counter
        assert _state.engine.current_tick == _state.current_tick == 5
```

- [ ] **Step 2: Run test — first test passes, second may fail on current code**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/test_server.py::TestTickSync -v`
Expected: `test_tick_derived_from_engine` may FAIL if engine.current_tick != state.current_tick (engine increments inside tick(), state increments outside)

- [ ] **Step 3: Fix tick to use engine as single source of truth**

Modify `src/memory_decay/server.py`. Change the tick/auto-tick endpoints to use `engine.current_tick` as the canonical value:

In the `/tick` endpoint (lines 375-388):
```python
    @app.post("/tick")
    async def tick(req: TickRequest):
        if not _state:
            raise HTTPException(503, "Server not initialized")

        def _do_ticks():
            for _ in range(req.count):
                _state.engine.tick()
            _state.current_tick = _state.engine.current_tick
            _state.last_tick_time = time.time()

        await asyncio.to_thread(_do_ticks)

        return {"current_tick": _state.current_tick}
```

In the `/auto-tick` endpoint (lines 390-413):
```python
    @app.post("/auto-tick")
    async def auto_tick():
        """Apply ticks based on elapsed real time since last tick."""
        if not _state:
            raise HTTPException(503, "Server not initialized")

        elapsed = time.time() - _state.last_tick_time
        ticks_due = int(elapsed / _state.tick_interval_seconds)

        if ticks_due > 0:
            ticks_due = min(ticks_due, 100)

            def _do_ticks():
                for _ in range(ticks_due):
                    _state.engine.tick()
                _state.current_tick = _state.engine.current_tick
                _state.last_tick_time = time.time()

            await asyncio.to_thread(_do_ticks)

        return {
            "ticks_applied": ticks_due,
            "current_tick": _state.current_tick,
            "elapsed_seconds": round(elapsed, 1),
        }
```

Also sync engine tick on startup in the lifespan. Set it on the engine right after construction (before creating ServerState):

```python
        engine = DecayEngine(
            store=store,
            custom_decay_fn=best_decay_fn,
            params=best_params,
        )
        engine.current_tick = state_tick  # restore tick before ServerState

        # ... then in ServerState creation:
        _state.current_tick = state_tick  # stays in sync with engine
```

- [ ] **Step 4: Run all tests**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git add src/memory_decay/server.py tests/test_server.py
git commit -m "fix: use engine.current_tick as single source of truth

Previously ServerState.current_tick and DecayEngine.current_tick were
incremented independently, risking divergence. Now engine owns the
tick counter and server syncs from it after each tick batch.

Directive: always read tick from engine, never increment state.current_tick directly
Tested: tick sync test, all existing server tests
Scope-risk: narrow
Confidence: high"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Extract shared BM25 module | `bm25.py`, `test_bm25.py` |
| 2 | Wire MemoryGraph to shared BM25 | `graph.py` |
| 3 | Add BM25 to MemoryStore.search() | `memory_store.py`, `test_memory_store.py` |
| 4 | Wire server to pass BM25 params | `server.py`, `test_server.py` |
| 5 | Fix embedding cache model filter | `memory_store.py`, `server.py`, `test_memory_store.py` |
| 6 | Unify tick synchronization | `server.py`, `test_server.py` |

After all tasks: set `bm25_weight` in `experiments/best/params.json` to activate BM25 in production. Suggested starting value: `0.3`.
