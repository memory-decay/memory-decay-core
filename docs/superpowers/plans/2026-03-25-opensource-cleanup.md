# Open-Source Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip internal research/simulation code from the core library and add missing open-source files (LICENSE, pyproject.toml metadata) for public release.

**Architecture:** Remove `main.py` (simulation runner), `evaluator.py` (research metrics), `persistence.py` (pickle graph saver) and their tests. Keep only the production path: `graph.py`, `decay.py`, `bm25.py`, `memory_store.py`, `embedding_provider.py`, `server.py`. Update `__init__.py`, README, and pyproject.toml to reflect the trimmed surface area.

**Tech Stack:** Python 3.10+, git

---

## File Structure

| File | Action | Why |
|------|--------|-----|
| `src/memory_decay/main.py` | **Delete** | Simulation runner — internal research only |
| `src/memory_decay/evaluator.py` | **Delete** | 3-pillar scoring — research metrics, not used by server |
| `src/memory_decay/persistence.py` | **Delete** | Pickle-based graph save/load — not used by server |
| `tests/test_simulation.py` | **Delete** | Tests for main.py |
| `tests/test_persistence.py` | **Delete** | Tests for persistence.py |
| `tests/test_core.py` | **Modify** | Remove `TestEvaluator` class, keep `TestMemoryGraph`, `TestDecayEngine`, `TestCustomDecayFn` |
| `src/memory_decay/__init__.py` | **Modify** | Remove `Evaluator` export |
| `README.md` | **Modify** | Remove evaluator/simulation references, update project structure |
| `pyproject.toml` | **Modify** | Add authors, license, repository, classifiers |
| `LICENSE` | **Create** | MIT license file |

---

## Chunk 1: Remove Research Code

### Task 1: Delete research modules and their tests

**Files:**
- Delete: `src/memory_decay/main.py`
- Delete: `src/memory_decay/evaluator.py`
- Delete: `src/memory_decay/persistence.py`
- Delete: `tests/test_simulation.py`
- Delete: `tests/test_persistence.py`
- Modify: `tests/test_core.py:1-8, 412-517` (remove Evaluator import and TestEvaluator class)
- Modify: `src/memory_decay/__init__.py`

- [ ] **Step 1: Delete research source files**

```bash
cd /home/roach/.openclaw/workspace/memory-decay-core
git rm src/memory_decay/main.py src/memory_decay/evaluator.py src/memory_decay/persistence.py
```

- [ ] **Step 2: Delete research test files**

```bash
git rm tests/test_simulation.py tests/test_persistence.py
```

- [ ] **Step 3: Clean up test_core.py — remove TestEvaluator class and Evaluator import**

In `tests/test_core.py`:

1. Remove `Evaluator` from the import line (line 7):
```python
# Before: from memory_decay import MemoryGraph, DecayEngine, Evaluator
# After:  from memory_decay import MemoryGraph, DecayEngine
```

2. Delete the entire `TestEvaluator` class (lines 412-517, from `# --- Evaluator Tests ---` through the end of the class, right before `# --- Custom Decay Function Tests ---`).

- [ ] **Step 4: Clean up __init__.py — remove Evaluator export**

Replace `src/memory_decay/__init__.py` with:

```python
"""Core memory decay library.

Graph-based memory model with activation, stability,
reinforcement-aware reactivation, and decay-weighted retrieval.
"""

from .graph import MemoryGraph
from .decay import DecayEngine
from .memory_store import MemoryStore
from .embedding_provider import EmbeddingProvider, create_embedding_provider

__all__ = [
    "MemoryGraph",
    "DecayEngine",
    "MemoryStore",
    "EmbeddingProvider",
    "create_embedding_provider",
]
```

- [ ] **Step 5: Run tests to verify nothing broke**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && .venv/bin/python -m pytest tests/ -v`
Expected: All remaining tests PASS (test_core.py, test_bm25.py, test_memory_store.py, test_server.py, test_embedding_provider.py, test_decay_sqlite.py, test_graph_temporal.py)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "remove internal research code from core library

Strip simulation runner (main.py), evaluation framework (evaluator.py),
and pickle persistence (persistence.py) along with their tests. These
are internal research tools not needed by the production server path.

Rejected: move to src/memory_decay/_internal/ | still ships in package, confuses users
Confidence: high
Scope-risk: narrow
Directive: evaluator.py lives in the experiment repo if needed for benchmarking
Tested: all remaining production tests pass"
```

---

## Chunk 2: Update README and Add Open-Source Files

### Task 2: Update README to reflect trimmed library

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Make these edits to `README.md`:

1. **Remove evaluator from Core Components table** (around line 74):
   Delete the row: `| evaluator.py | Evaluator | 3-pillar scoring: retention AUC, forgetting, plausibility |`

2. **Remove Evaluator from Quick Start code** (around line 101):
```python
# Before: from memory_decay import MemoryGraph, DecayEngine, Evaluator
# After:  from memory_decay import MemoryGraph, DecayEngine
```

3. **Update Project Structure** (around line 352-367):
Replace with:
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

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "update README to reflect trimmed public API

Remove references to evaluator.py, main.py, persistence.py.
Update project structure and quick start examples.

Scope-risk: narrow"
```

---

### Task 3: Add LICENSE and complete pyproject.toml

**Files:**
- Create: `LICENSE`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create MIT LICENSE file**

Create `LICENSE` with standard MIT text. Use `2025-2026 memory-decay contributors` as copyright holder (confirm with user if different).

```
MIT License

Copyright (c) 2025-2026 memory-decay contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Complete pyproject.toml metadata**

Replace `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "memory-decay"
version = "0.1.0"
description = "Human-like memory decay for AI agents"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "memory-decay contributors"},
]
keywords = ["memory", "decay", "ai", "agent", "forgetting", "retrieval"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "networkx>=3.0",
    "sentence-transformers>=2.2",
    "numpy>=1.24",
    "openai>=1.0",
    "google-genai>=1.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "sqlite-vec>=0.1.7",
]

[project.urls]
Homepage = "https://github.com/memory-decay/memory-decay-core"
Repository = "https://github.com/memory-decay/memory-decay-core"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "httpx>=0.28.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Note:** Confirm the GitHub org/repo URL with user. Using `memory-decay/memory-decay-core` based on git remote.

- [ ] **Step 3: Run tests one final time**

Run: `cd /home/roach/.openclaw/workspace/memory-decay-core && .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add LICENSE pyproject.toml
git commit -m "add MIT license and complete package metadata

Required for open-source release. Adds LICENSE file referenced
by README badge, and completes pyproject.toml with authors,
classifiers, URLs, and keywords.

Scope-risk: narrow
Confidence: high"
```

---

## Summary

| Task | What | Lines removed |
|------|------|---------------|
| 1 | Delete main.py, evaluator.py, persistence.py + tests | ~2,070 |
| 2 | Update README | ~20 lines changed |
| 3 | Add LICENSE, complete pyproject.toml | ~50 lines added |

**Net result:** ~2,000 lines of research slop removed. Clean, focused library: 6 source modules, 6 test files, MIT licensed, proper metadata.
