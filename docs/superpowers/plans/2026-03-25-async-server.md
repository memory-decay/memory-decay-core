# Async Server Conversion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert FastAPI server from sync to async, add batch embedding, minimize SQLite locking, set workers=2.

**Architecture:** Async endpoints with `asyncio.to_thread()` for CPU-bound/blocking ops (SQLite, local embeddings). Native async for OpenAI API. Batch embedding in `/store-batch`. Reduced commit frequency in MemoryStore.

**Tech Stack:** FastAPI (async), asyncio, openai.AsyncOpenAI, httpx (test), uvicorn

---

## Task 1: Add async methods to EmbeddingProvider

**Files:**
- Modify: `src/memory_decay/embedding_provider.py`
- Modify: `tests/test_embedding_provider.py`

- [ ] **Step 1: Add `aembed` and `aembed_batch` to base class**

Default impl wraps sync via `asyncio.to_thread()`. OpenAI overrides with native async client.

- [ ] **Step 2: Add async to OpenAIEmbeddingProvider**

Use `openai.AsyncOpenAI` for native async.

- [ ] **Step 3: Add tests for async embedding**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embedding_provider.py -v`

- [ ] **Step 5: Commit**

---

## Task 2: Reduce MemoryStore commit frequency

**Files:**
- Modify: `src/memory_decay/memory_store.py`
- Modify: `tests/test_memory_store.py`

- [ ] **Step 1: Add `auto_commit` param to write methods**

Methods that currently call `self._db.commit()` get `auto_commit=True` param. When False, skip commit. Add public `commit()` method.

- [ ] **Step 2: Add `add_memories_batch()` method**

Single-transaction batch insert for multiple memories.

- [ ] **Step 3: Add tests for batch insert and manual commit**

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_memory_store.py -v`

- [ ] **Step 5: Commit**

---

## Task 3: Convert server to async

**Files:**
- Modify: `src/memory_decay/server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Make ServerState.embed async**

Use provider's `aembed` / `aembed_batch`, with cache check via `to_thread`.

- [ ] **Step 2: Convert all endpoints to `async def`**

Wrap SQLite calls with `asyncio.to_thread()`.

- [ ] **Step 3: Optimize `/store-batch` with batch embedding**

Separate cache hits/misses, batch-embed misses in one call.

- [ ] **Step 4: Set workers=2 in CLI**

- [ ] **Step 5: Update tests to use httpx.AsyncClient**

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 7: Commit**
