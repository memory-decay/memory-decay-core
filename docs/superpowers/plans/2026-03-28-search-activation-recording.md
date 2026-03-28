# Search-Triggered Activation Recording Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record activation history automatically when memories are searched (retrieved), not just on decay ticks.

**Architecture:** When a search returns results, call `record_activation_history(current_tick)` to snapshot all memory scores at that tick. The existing dedup logic (`INSERT OR REPLACE`) ensures no duplicate rows at the same tick — subsequent calls at the same tick overwrite, preserving the post-reinforcement scores.

**Tech Stack:** Python/fastapi, SQLite, existing `MemoryStore.record_activation_history` method.

---

## Chunk 1: Add test for search-triggered activation recording

**Files:**
- Modify: `tests/test_activation_history.py`

- [ ] **Step 1: Add integration test to `TestActivationHistoryIntegration`**

Read the existing test file first (`tests/test_activation_history.py`), then add this test after the existing integration tests:

```python
def test_search_records_activation_history(self):
    """Search should automatically record activation history for returned memories."""
    store = MemoryStore(":memory:", embedding_dim=dim)
    engine = DecayEngine()
    app = create_app(store, engine)
    client = TestClient(app)

    # Store a memory
    r = client.post("/store", json={"text": "important meeting notes", "importance": 0.9, "mtype": "fact"})
    assert r.status_code == 200
    memory_id = r.json()["id"]

    # Search for it — this should trigger activation history recording
    r = client.post("/search", json={"query": "important meeting notes", "top_k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == memory_id

    # Verify activation history was recorded at the current tick
    history = store.get_activation_history(memory_id)
    assert len(history) == 1
    assert history[0]["tick"] == 0  # current_tick starts at 0
    assert history[0]["retrieval_score"] == 1.0  # initial score

    # Search again — should record another snapshot (scores may change due to consolidation)
    r = client.post("/search", json={"query": "important meeting", "top_k": 1})
    # Tick may or may not have advanced depending on auto-tick interval
    history = store.get_activation_history(memory_id)
    # At minimum, first snapshot should exist
    assert len(history) >= 1

    store.close()
```

Run: `pytest tests/test_activation_history.py::TestActivationHistoryIntegration::test_search_records_activation_history -v`
Expected: **FAIL** — `search` handler doesn't call `record_activation_history` yet.

- [ ] **Step 2: Commit**

```bash
git add tests/test_activation_history.py
git commit -m "test: add search-triggered activation history test (TDD)"
```

---

## Chunk 2: Implement search-triggered recording in server

**Files:**
- Modify: `src/memory_decay/server.py:406-417` (search handler)

- [ ] **Step 1: Modify `search` handler to call `record_activation_history`**

Read `src/memory_decay/server.py` lines 387–417, then add the call:

In the `search` handler, after the reinforcement block (line 415) and before the return (line 417), add:

```python
        # Record activation history for all returned memories
        if _state.history_interval > 0:
            await asyncio.to_thread(
                lambda: _state.store.record_activation_history(_state.current_tick)
            )
```

The `history_interval > 0` guard ensures this is a no-op when history recording is disabled.

Run: `pytest tests/test_activation_history.py::TestActivationHistoryIntegration::test_search_records_activation_history -v`
Expected: **PASS**

- [ ] **Step 2: Run full activation history test suite**

Run: `pytest tests/test_activation_history.py -v`
Expected: All PASS

- [ ] **Step 3: Run server tests to ensure no regression**

Run: `pytest tests/test_server.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/memory_decay/server.py
git commit -m "feat: record activation history on every search

Call record_activation_history(current_tick) in the search handler,
with history_interval guard to allow disabling. Existing INSERT OR
REPLACE dedup handles repeated searches at the same tick — scores
reflect post-reinforcement state.

Constraint: history_interval must be > 0 (default) to enable"
```

---

## Chunk 3: Run E2E verification

- [ ] **Step 1: Start temp server with history recording enabled**

```bash
.venv/bin/python -m memory_decay.server --port 8199 --db-path :memory: &
sleep 3
```

- [ ] **Step 2: Store and search, verify history is recorded**

```bash
# Store
ID=$(curl -s -X POST http://127.0.0.1:8199/store \
  -H "Content-Type: application/json" \
  -d '{"text": "test memory for activation", "importance": 0.8}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Search twice
curl -s -X POST http://127.0.0.1:8199/search -H "Content-Type: application/json" -d '{"query": "test memory activation", "top_k": 1}' > /dev/null
curl -s -X POST http://127.0.0.1:8199/search -H "Content-Type: application/json" -d '{"query": "test memory", "top_k": 1}' > /dev/null

# Check history via admin endpoint
curl -s "http://127.0.0.1:8199/admin/activation-history?memory_id=$ID"
```

Expected: Returns history array with at least 1 entry. Scores reflect consolidation effect.

- [ ] **Step 3: Verify history_interval=0 disables recording**

```bash
# Kill and restart with history_interval=0
kill %1
.venv/bin/python -m memory_decay.server --port 8199 --db-path :memory: --history-interval 0 &
sleep 3

ID2=$(curl -s -X POST http://127.0.0.1:8199/store \
  -H "Content-Type: application/json" \
  -d '{"text": "no history", "importance": 0.5}' | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
curl -s -X POST http://127.0.0.1:8199/search -H "Content-Type: application/json" -d '{"query": "no history", "top_k": 1}' > /dev/null
curl -s "http://127.0.0.1:8199/admin/activation-history?memory_id=$ID2"
```

Expected: `{"history":[]}` — recording disabled.

- [ ] **Step 4: Cleanup**

```bash
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(e2e): verify search-triggered activation recording and history_interval guard"
```

---

## Files Summary

| File | Change |
|------|--------|
| `tests/test_activation_history.py` | Add `test_search_records_activation_history` integration test |
| `src/memory_decay/server.py:415` | Add `record_activation_history` call in search handler |
| `docs/superpowers/plans/2026-03-28-search-activation-recording.md` | This plan |
