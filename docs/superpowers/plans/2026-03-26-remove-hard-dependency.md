# Remove Hard Dependency on sentence-transformers

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `sentence-transformers` from required dependencies to avoid forcing ~2GB PyTorch installation on users who only need Gemini/OpenAI embeddings. Also remove `MemoryGraph` from default exports since it's not used in production server paths.

**Architecture:** 
1. Move `sentence-transformers` to `[project.optional-dependencies]` under `local` extra group
2. Remove `MemoryGraph` from `__init__.py` default exports (keep it importable from `memory_decay.graph` for direct use)
3. Update README Quick Start to use `MemoryStore` instead of `MemoryGraph`

**Tech Stack:** Python packaging (pyproject.toml), setuptools optional dependencies

---

## Task 1: Move sentence-transformers to optional dependencies

**Files:**
- Modify: `pyproject.toml:26-35`

**Context:**
Currently `sentence-transformers>=2.2` is in `[project.dependencies]`. It pulls in PyTorch (~2GB) which is unnecessary for users using Gemini/OpenAI embeddings. `LocalEmbeddingProvider` already uses lazy import (imports inside `_ensure_model()`), so no code changes needed there.

- [ ] **Step 1: Remove sentence-transformers from dependencies**

Edit `pyproject.toml` dependencies list to remove `sentence-transformers>=2.2`.

Expected result: `dependencies` should contain only:
- networkx>=3.0
- numpy>=1.24
- openai>=1.0
- google-genai>=1.0
- fastapi>=0.115.0
- uvicorn>=0.34.0
- sqlite-vec>=0.1.7

- [ ] **Step 2: Add local extra group with sentence-transformers**

Add under `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
local = ["sentence-transformers>=2.2"]
dev = [
    "pytest>=7.0",
    "httpx>=0.28.0",
]
```

- [ ] **Step 3: Test that package installs without sentence-transformers**

Run:
```bash
pip install -e . 2>&1 | grep -i "torch\|sentence-transformers" || echo "No torch/st installation"
```

Expected: No torch or sentence-transformers in output (they shouldn't be installed).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: move sentence-transformers to optional local extra

Avoid forcing ~2GB PyTorch install on users who only use
Gemini/OpenAI embeddings. LocalEmbeddingProvider already
uses lazy import, so no code changes needed.

Install with local embeddings:
  pip install -e '.[local]'"
```

---

## Task 2: Remove MemoryGraph from default exports

**Files:**
- Modify: `src/memory_decay/__init__.py:1-18`
- Modify: `README.md:95-132` (Quick Start section)

**Context:**
`MemoryGraph` is exported in `__init__.py` but the production server uses `MemoryStore`. Keeping `MemoryGraph` in default exports means users importing `from memory_decay import MemoryGraph` will trigger import side effects even if they don't use it. Since `graph.py` has lazy import for sentence-transformers, the main issue is API clarity - we want to guide users toward `MemoryStore` for production use.

- [ ] **Step 1: Remove MemoryGraph from __init__.py**

Edit `src/memory_decay/__init__.py`:

```python
"""Core memory decay library.

Graph-based memory model with activation, stability,
reinforcement-aware reactivation, and decay-weighted retrieval.
"""

from .decay import DecayEngine
from .memory_store import MemoryStore
from .embedding_provider import EmbeddingProvider, create_embedding_provider

__all__ = [
    "DecayEngine",
    "MemoryStore",
    "EmbeddingProvider",
    "create_embedding_provider",
]
```

- [ ] **Step 2: Update README Quick Start to use MemoryStore**

Replace the "As a Library" Quick Start section (lines ~95-132) with `MemoryStore` based example:

```python
from memory_decay import MemoryStore, DecayEngine

# 1. Create a memory store with Gemini embeddings
store = MemoryStore(
    db_path="./data/memories.db",
    embedding_provider="gemini",
    api_key="your-gemini-api-key",
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

Also update the "Core Components" table to indicate `MemoryGraph` is available via direct import:

```markdown
| Module | Class | Role | Import |
|--------|-------|------|--------|
| `graph.py` | `MemoryGraph` | In-memory NetworkX graph for prototyping | `from memory_decay.graph import MemoryGraph` |
| `decay.py` | `DecayEngine` | Time-step decay with exponential/power-law modes | `from memory_decay import DecayEngine` |
| `memory_store.py` | `MemoryStore` | SQLite + sqlite-vec persistence (production) | `from memory_decay import MemoryStore` |
```

- [ ] **Step 3: Verify MemoryGraph can still be imported directly**

Run:
```bash
cd /home/roach/.openclaw/workspace/memory-decay-core && python -c "from memory_decay.graph import MemoryGraph; print('OK: MemoryGraph import works')"
```

Expected: `OK: MemoryGraph import works`

- [ ] **Step 4: Verify default imports work without sentence-transformers**

Run:
```bash
cd /home/roach/.openclaw/workspace/memory-decay-core && python -c "from memory_decay import MemoryStore, DecayEngine, create_embedding_provider; print('OK: Default imports work')"
```

Expected: `OK: Default imports work` (no ImportError for sentence_transformers)

- [ ] **Step 5: Commit**

```bash
git add src/memory_decay/__init__.py README.md
git commit -m "refactor: remove MemoryGraph from default exports

MemoryGraph is no longer exported by default from memory_decay.
It can still be imported directly from memory_decay.graph for
prototyping use cases. The default exports now focus on
MemoryStore which is used by the production server.

Updated README Quick Start to use MemoryStore instead."
```

---

## Task 3: Update tests if needed

**Files:**
- Check: `tests/test_simulation.py`

**Context:**
The test file may import `MemoryGraph` from the package root. Need to update imports if so.

- [ ] **Step 1: Check test imports**

Run:
```bash
grep -n "from memory_decay import\|import memory_decay" tests/*.py
```

- [ ] **Step 2: Update test imports if MemoryGraph is used**

If tests use `from memory_decay import MemoryGraph`, change to:
```python
from memory_decay.graph import MemoryGraph
```

- [ ] **Step 3: Run tests to verify**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core && pip install -e ".[dev,local]" && pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit if changes made**

```bash
git add tests/
git commit -m "test: update imports for MemoryGraph relocation" || echo "No changes to commit"
```

---

## Task 4: Final verification and cleanup

- [ ] **Step 1: Verify clean install without local extra**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
pip uninstall -y sentence-transformers torch torchvision torchaudio 2>/dev/null || true
pip install -e .
python -c "
from memory_decay import MemoryStore, DecayEngine, create_embedding_provider
print('✓ Default imports work')

# Verify MemoryGraph can be imported from submodule
from memory_decay.graph import MemoryGraph
print('✓ MemoryGraph direct import works')

# Verify sentence-transformers is NOT imported by default
try:
    import sentence_transformers
    print('✗ sentence_transformers was imported (should not be)')
except ImportError:
    print('✓ sentence_transformers not auto-imported')
"
```

- [ ] **Step 2: Verify install with local extra works**

```bash
pip install -e ".[local]"
python -c "
import sentence_transformers
print('✓ sentence_transformers available with [local] extra')

from memory_decay import LocalEmbeddingProvider
print('✓ LocalEmbeddingProvider works')
"
```

- [ ] **Step 3: Delete FIX.md or mark as completed**

Since the Critical items from FIX.md are now complete, either delete the file or mark the items as done.

```bash
rm /home/roach/.openclaw/workspace/memory-decay-core/FIX.md
git add FIX.md
git commit -m "chore: remove FIX.md after completing critical fixes"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `pyproject.toml` | Move `sentence-transformers` from `dependencies` to `[project.optional-dependencies]` under `local` |
| `src/memory_decay/__init__.py` | Remove `MemoryGraph` from imports and `__all__` |
| `README.md` | Update Quick Start to use `MemoryStore`, update Core Components table |
| `tests/test_simulation.py` | Update imports if `MemoryGraph` imported from package root |
| `FIX.md` | Delete after completion |

## User Impact

**Before:**
```bash
pip install memory-decay  # Installs ~2GB PyTorch even if using Gemini/OpenAI
from memory_decay import MemoryGraph  # Works
```

**After:**
```bash
pip install memory-decay  # Lightweight, no PyTorch
pip install "memory-decay[local]"  # With local embeddings (~2GB)

from memory_decay import MemoryStore  # Recommended
from memory_decay.graph import MemoryGraph  # For prototyping
```
