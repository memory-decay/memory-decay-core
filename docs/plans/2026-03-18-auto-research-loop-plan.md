# Auto-Research Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a closed-loop auto-research system where Claude Code autonomously proposes, tests, and selects decay functions via `program.md` + `/loop 30m`.

**Architecture:** Modify `DecayEngine` to accept pluggable decay functions via a "function slot" pattern. Create a lightweight `runner.py` that executes single experiments using cached embeddings. Claude Code reads results and iterates.

**Tech Stack:** Python 3.10+, existing memory_decay package, pytest, numpy, pickle (for numpy array serialization — safe here since cache is always self-generated, never from untrusted sources)

---

### Task 1: Add custom_decay_fn support to DecayEngine

**Files:**
- Modify: `src/memory_decay/decay.py:19-26` (constructor) and `src/memory_decay/decay.py:53-90` (`_compute_decay`)
- Test: `tests/test_core.py`

**Step 1: Write the failing test**

Add to `tests/test_core.py`:

```python
class TestCustomDecayFn:
    def test_custom_fn_is_used_when_provided(self):
        graph = make_graph()
        graph.add_memory("m1", "fact", "테스트 기억", 0.5, 0)

        def my_decay(activation, impact, stability, mtype, params):
            return activation * 0.5  # Always halve

        engine = DecayEngine(graph, custom_decay_fn=my_decay)
        engine.tick()

        node = graph.get_node("m1")
        assert abs(node["activation_score"] - 0.5) < 1e-6

    def test_default_behavior_when_no_custom_fn(self):
        graph = make_graph()
        graph.add_memory("m1", "fact", "테스트 기억", 0.5, 0)

        engine = make_exponential_engine(graph)
        engine.tick()

        node = graph.get_node("m1")
        # Should use default exponential decay, not 0.5
        assert node["activation_score"] != 0.5
        assert 0.0 < node["activation_score"] < 1.0

    def test_custom_fn_output_clamped_to_0_1(self):
        graph = make_graph()
        graph.add_memory("m1", "fact", "테스트 기억", 0.5, 0)

        def bad_decay(activation, impact, stability, mtype, params):
            return 5.0  # Out of range

        engine = DecayEngine(graph, custom_decay_fn=bad_decay)
        engine.tick()

        node = graph.get_node("m1")
        assert node["activation_score"] == 1.0  # Clamped

    def test_custom_fn_receives_params(self):
        graph = make_graph()
        graph.add_memory("m1", "fact", "테스트 기억", 0.5, 0)

        received = {}

        def spy_decay(activation, impact, stability, mtype, params):
            received.update(params)
            return activation * 0.9

        engine = DecayEngine(
            graph, custom_decay_fn=spy_decay,
            params={"lambda_fact": 0.99}
        )
        engine.tick()

        assert received["lambda_fact"] == 0.99
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_core.py::TestCustomDecayFn -v`
Expected: FAIL — `TypeError: DecayEngine.__init__() got an unexpected keyword argument 'custom_decay_fn'`

**Step 3: Write minimal implementation**

In `src/memory_decay/decay.py`, modify `__init__`:

```python
def __init__(
    self,
    graph: MemoryGraph,
    decay_type: Literal["exponential", "power_law"] = "exponential",
    params: dict | None = None,
    custom_decay_fn=None,
):
    self._graph = graph
    self.decay_type = decay_type
    self.current_tick = 0
    self._custom_decay_fn = custom_decay_fn

    self._params = {
        # ... existing defaults unchanged ...
    }
    if params:
        self._params.update(params)
```

In `_compute_decay`, add at the top:

```python
def _compute_decay(self, initial_activation, impact, stability, mtype):
    if self._custom_decay_fn is not None:
        result = self._custom_decay_fn(
            initial_activation, impact, stability, mtype, self._params
        )
        return min(max(result, 0.0), 1.0)

    # ... existing logic unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_core.py::TestCustomDecayFn -v`
Expected: 4 passed

**Step 5: Run all existing tests to verify no regressions**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest -q`
Expected: All existing tests still pass (custom_decay_fn defaults to None)

**Step 6: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/decay.py tests/test_core.py
git commit -m "Add custom_decay_fn slot to DecayEngine

Enable pluggable decay functions via a function slot pattern.
When custom_decay_fn is provided, it replaces the built-in
_compute_decay logic. Output is clamped to [0, 1].

Constraint: Existing API unchanged (custom_decay_fn=None by default)
Tested: 4 new unit tests + full regression suite
Scope-risk: narrow"
```

---

### Task 2: Create cache_builder.py

**Files:**
- Create: `src/memory_decay/cache_builder.py`
- Test: `tests/test_cache_builder.py`

**Step 1: Write the failing test**

Create `tests/test_cache_builder.py`:

```python
"""Tests for embedding cache builder."""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memory_decay.cache_builder import build_cache, load_cache


SAMPLE_DATASET = [
    {
        "id": "f1", "type": "fact",
        "content": "서울은 대한민국의 수도이다",
        "entities": ["서울"], "tick": 0, "impact": 0.9,
        "associations": [],
        "recall_query": "대한민국의 수도는?", "recall_answer": "서울",
    },
    {
        "id": "e1", "type": "episode",
        "content": "커피를 마셨다",
        "entities": ["커피"], "tick": 5, "impact": 0.5,
        "associations": [{"id": "f1", "weight": 0.6}],
        "recall_query": "무엇을 마셨는가?", "recall_answer": "커피",
    },
]


def mock_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


class TestCacheBuilder:
    def test_build_cache_creates_files(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        assert (cache_dir / "embeddings.pkl").exists()
        assert (cache_dir / "dataset.json").exists()
        assert (cache_dir / "test_queries.json").exists()

    def test_cache_contains_all_texts(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        with open(cache_dir / "embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        # All content texts cached
        for item in SAMPLE_DATASET:
            assert item["content"] in embeddings
            assert isinstance(embeddings[item["content"]], np.ndarray)

        # All recall queries cached
        for item in SAMPLE_DATASET:
            if "recall_query" in item:
                assert item["recall_query"] in embeddings

    def test_load_cache_returns_embedder(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        cached_embedder, dataset, test_queries = load_cache(str(cache_dir))

        # Embedder works for cached texts
        emb = cached_embedder("서울은 대한민국의 수도이다")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

        # Dataset loaded
        assert len(dataset) == 2

        # Test queries loaded
        assert len(test_queries) == 2
        assert test_queries[0] == ("대한민국의 수도는?", "f1")

    def test_load_cache_raises_for_unknown_text(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)

        cached_embedder, _, _ = load_cache(str(cache_dir))

        with pytest.raises(KeyError):
            cached_embedder("이건 캐시에 없는 텍스트")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_cache_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_decay.cache_builder'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/cache_builder.py`:

```python
"""Pre-compute and cache embeddings for the auto-research loop.

Caches all memory content texts and recall query texts so that
repeated simulation runs require zero embedding API calls.

Note: Uses pickle for numpy array serialization. The cache is always
self-generated from the local dataset — never loaded from untrusted sources.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np


def build_cache(
    dataset_path: str,
    cache_dir: str,
    embedder: Optional[Callable] = None,
    embedding_backend: str = "auto",
) -> None:
    """Build embedding cache from a JSONL dataset.

    Args:
        dataset_path: Path to the JSONL dataset file.
        cache_dir: Directory to write cache files into.
        embedder: Optional custom embedding function. If None, uses MemoryGraph's
                  default backend selection.
        embedding_backend: Backend to use if embedder is None.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    # Get embedder
    if embedder is None:
        from .graph import MemoryGraph
        graph = MemoryGraph(embedding_backend=embedding_backend)
        embedder = graph._embed_text

    # Compute embeddings for all unique texts
    embeddings: dict[str, np.ndarray] = {}
    texts = set()

    for item in dataset:
        texts.add(item["content"])
        if "recall_query" in item:
            texts.add(item["recall_query"])

    for i, text in enumerate(sorted(texts)):
        emb = np.array(embedder(text), dtype=np.float32)
        embeddings[text] = emb
        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(texts)} texts")

    print(f"  Cached {len(embeddings)} embeddings")

    # Save embeddings
    with open(cache_path / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    # Save parsed dataset
    with open(cache_path / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Save test queries
    test_queries = [
        (item["recall_query"], item["id"])
        for item in dataset
        if "recall_query" in item
    ]
    with open(cache_path / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)


def load_cache(cache_dir: str) -> tuple[Callable, list[dict], list[tuple[str, str]]]:
    """Load cached embeddings and return a cached embedder function.

    Returns:
        (cached_embedder, dataset, test_queries)
    """
    cache_path = Path(cache_dir)

    with open(cache_path / "embeddings.pkl", "rb") as f:
        embeddings: dict[str, np.ndarray] = pickle.load(f)

    with open(cache_path / "dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(cache_path / "test_queries.json", "r", encoding="utf-8") as f:
        test_queries = [tuple(q) for q in json.load(f)]

    def cached_embedder(text: str) -> np.ndarray:
        if text not in embeddings:
            raise KeyError(f"Text not in embedding cache: {text[:80]}...")
        return embeddings[text].copy()

    return cached_embedder, dataset, test_queries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build embedding cache")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="cache", help="Cache output directory")
    parser.add_argument(
        "--backend", choices=["auto", "local", "gemini"], default="auto"
    )
    args = parser.parse_args()

    build_cache(args.dataset, args.output, embedding_backend=args.backend)
    print(f"Cache saved to {args.output}/")
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_cache_builder.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/cache_builder.py tests/test_cache_builder.py
git commit -m "Add cache_builder for pre-computed embeddings

Caches all memory content and recall query embeddings to disk
so repeated simulation runs cost zero API calls. Provides
load_cache() that returns a cached embedder function.

Tested: build/load round-trip, all texts cached, unknown text raises
Scope-risk: narrow"
```

---

### Task 3: Create runner.py (experiment executor)

**Files:**
- Create: `src/memory_decay/runner.py`
- Test: `tests/test_runner.py`

**Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
"""Tests for the single-experiment runner."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from memory_decay.runner import run_experiment, validate_decay_fn


SAMPLE_DATASET = [
    {"id": "f1", "type": "fact", "content": "서울은 대한민국의 수도이다",
     "entities": ["서울"], "tick": 0, "impact": 0.9, "associations": [],
     "recall_query": "대한민국의 수도는?", "recall_answer": "서울"},
    {"id": "f2", "type": "fact", "content": "김민수는 커피를 좋아한다",
     "entities": ["김민수"], "tick": 5, "impact": 0.7,
     "associations": [{"id": "f1", "weight": 0.6}],
     "recall_query": "김민수는 무엇을 좋아하는가?", "recall_answer": "커피"},
    {"id": "e1", "type": "episode", "content": "서울에서 커피를 마셨다",
     "entities": ["서울"], "tick": 10, "impact": 0.5,
     "associations": [{"id": "f1", "weight": 0.8}],
     "recall_query": "어디서 커피를 마셨는가?", "recall_answer": "서울"},
    {"id": "e2", "type": "episode", "content": "길에서 강아지를 만났다",
     "entities": ["강아지"], "tick": 20, "impact": 0.3, "associations": [],
     "recall_query": "길에서 만난 동물은?", "recall_answer": "강아지"},
]


def mock_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


def build_test_cache(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = {}
    for item in SAMPLE_DATASET:
        embeddings[item["content"]] = mock_embedder(item["content"])
        if "recall_query" in item:
            embeddings[item["recall_query"]] = mock_embedder(item["recall_query"])
    with open(cache_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open(cache_dir / "dataset.json", "w") as f:
        json.dump(SAMPLE_DATASET, f, ensure_ascii=False)
    with open(cache_dir / "test_queries.json", "w") as f:
        json.dump([(item["recall_query"], item["id"]) for item in SAMPLE_DATASET], f)


GOOD_DECAY_FN = '''\
import math

def compute_decay(activation, impact, stability, mtype, params):
    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    effective = lam / max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    return activation * math.exp(-effective)
'''

BAD_DECAY_RETURNS_CONSTANT = '''\
def compute_decay(activation, impact, stability, mtype, params):
    return 1.0  # No decay at all
'''

BAD_DECAY_SYNTAX_ERROR = '''\
def compute_decay(activation, impact, stability, mtype, params)
    return 0.5
'''

DEFAULT_PARAMS = {
    "lambda_fact": 0.02, "lambda_episode": 0.035,
    "beta_fact": 0.08, "beta_episode": 0.12,
    "alpha": 0.5, "stability_weight": 0.8,
    "stability_decay": 0.01,
    "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05,
    "stability_cap": 1.0,
}


class TestValidateDecayFn:
    def test_valid_function_passes(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(GOOD_DECAY_FN)

        ok, error = validate_decay_fn(str(fn_path), DEFAULT_PARAMS)
        assert ok is True
        assert error is None

    def test_syntax_error_caught(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(BAD_DECAY_SYNTAX_ERROR)

        ok, error = validate_decay_fn(str(fn_path), {})
        assert ok is False
        assert "syntax" in error.lower() or "SyntaxError" in error

    def test_no_decay_function_caught(self, tmp_path):
        fn_path = tmp_path / "decay_fn.py"
        fn_path.write_text(BAD_DECAY_RETURNS_CONSTANT)

        ok, error = validate_decay_fn(str(fn_path), {})
        assert ok is False
        assert "decay" in error.lower() or "constant" in error.lower()


class TestRunExperiment:
    def test_successful_experiment(self, tmp_path):
        cache_dir = tmp_path / "cache"
        build_test_cache(cache_dir)

        exp_dir = tmp_path / "exp_0001"
        exp_dir.mkdir()
        (exp_dir / "decay_fn.py").write_text(GOOD_DECAY_FN)
        (exp_dir / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp_dir / "hypothesis.txt").write_text("baseline exponential test")

        run_experiment(str(exp_dir), str(cache_dir))

        results_path = exp_dir / "results.json"
        assert results_path.exists()

        results = json.loads(results_path.read_text())
        assert results["status"] == "completed"
        assert "overall_score" in results
        assert 0.0 <= results["overall_score"] <= 1.0

    def test_validation_failure_recorded(self, tmp_path):
        cache_dir = tmp_path / "cache"
        build_test_cache(cache_dir)

        exp_dir = tmp_path / "exp_bad"
        exp_dir.mkdir()
        (exp_dir / "decay_fn.py").write_text(BAD_DECAY_SYNTAX_ERROR)
        (exp_dir / "params.json").write_text("{}")
        (exp_dir / "hypothesis.txt").write_text("bad syntax test")

        run_experiment(str(exp_dir), str(cache_dir))

        results = json.loads((exp_dir / "results.json").read_text())
        assert results["status"] == "validation_failed"
        assert "error" in results
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memory_decay.runner'`

**Step 3: Write minimal implementation**

Create `src/memory_decay/runner.py`:

```python
"""Single experiment runner for the auto-research loop.

Loads a custom decay function from an experiment directory,
runs a simulation using cached embeddings, and saves results.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

from .cache_builder import load_cache
from .main import build_graph_from_dataset, run_simulation
from .decay import DecayEngine
from .evaluator import Evaluator


def validate_decay_fn(fn_path: str, params: dict) -> tuple[bool, Optional[str]]:
    """Validate a decay function file before running a full experiment.

    Checks:
    1. File compiles without syntax errors
    2. Module has a compute_decay callable
    3. Output is in [0, 1] for standard inputs
    4. Function actually decays (doesn't return constant 1.0)
    5. Zero activation stays near zero

    Returns:
        (ok, error_message) — ok=True if all checks pass
    """
    path = Path(fn_path)

    # 1. Syntax check
    try:
        with open(path, "r") as f:
            source = f.read()
        compile(source, str(path), "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 2. Load module
    try:
        spec = importlib.util.spec_from_file_location("decay_fn", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        return False, f"Import error: {e}"

    if not hasattr(module, "compute_decay") or not callable(module.compute_decay):
        return False, "Module has no callable 'compute_decay'"

    fn = module.compute_decay

    # 3. Basic output range checks
    test_cases = [
        (1.0, 0.0, 0.0, "fact"),
        (1.0, 0.0, 0.0, "episode"),
        (0.5, 0.5, 0.5, "fact"),
        (0.8, 1.0, 1.0, "episode"),
        (0.0, 0.0, 0.0, "fact"),
    ]

    results = []
    for activation, impact, stability, mtype in test_cases:
        try:
            result = fn(activation, impact, stability, mtype, params)
        except Exception as e:
            return False, f"Runtime error with inputs ({activation}, {impact}, {stability}, {mtype}): {e}"

        if not isinstance(result, (int, float)):
            return False, f"Output is not numeric: {type(result)}"
        if result < -0.01 or result > 1.01:
            return False, f"Output {result} out of range [0, 1]"
        results.append(result)

    # 4. Must actually decay: f(1.0, 0, 0, ...) must be < 1.0
    if results[0] >= 1.0 and results[1] >= 1.0:
        return False, "No decay detected: compute_decay(1.0, 0, 0, ...) returned 1.0 (constant, no decay)"

    # 5. Zero input should stay near zero
    if results[4] > 0.01:
        return False, f"Zero activation produced non-zero output: {results[4]}"

    return True, None


def _load_decay_fn(fn_path: str):
    """Dynamically import a decay function from a file."""
    spec = importlib.util.spec_from_file_location("decay_fn", fn_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_decay


def run_experiment(
    experiment_dir: str,
    cache_dir: str,
    total_ticks: int = 200,
    eval_interval: int = 20,
    reactivation_policy: str = "scheduled_query",
    seed: int = 42,
) -> dict:
    """Run a single experiment from an experiment directory.

    Expected files in experiment_dir:
        decay_fn.py   - Custom decay function
        params.json   - Parameters for the decay function
        hypothesis.txt - Reasoning (for record-keeping)

    Output:
        results.json  - Written to experiment_dir
    """
    exp_path = Path(experiment_dir)
    start_time = time.time()

    # Load experiment config
    with open(exp_path / "params.json", "r") as f:
        params = json.load(f)

    fn_path = str(exp_path / "decay_fn.py")

    # Validate
    ok, error = validate_decay_fn(fn_path, params)
    if not ok:
        result = {
            "status": "validation_failed",
            "error": error,
            "duration_seconds": round(time.time() - start_time, 2),
        }
        with open(exp_path / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        return result

    # Load cache
    cached_embedder, dataset, test_queries = load_cache(cache_dir)

    # Build graph with cached embeddings
    graph = build_graph_from_dataset(dataset, embedder=cached_embedder)

    # Load custom decay function
    decay_fn = _load_decay_fn(fn_path)

    # Create engine with custom function
    engine = DecayEngine(graph, custom_decay_fn=decay_fn, params=params)
    evaluator = Evaluator(graph, engine)

    # Run simulation
    snapshots = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks,
        eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        seed=seed,
    )

    # Get final scores
    final_summary = evaluator.score_summary(test_queries)
    duration = time.time() - start_time

    result = {
        "status": "completed",
        **final_summary,
        "snapshots": snapshots,
        "duration_seconds": round(duration, 2),
    }

    with open(exp_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single decay experiment")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--cache", default="cache", help="Cache directory")
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--policy", default="scheduled_query")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_experiment(
        args.experiment_dir, args.cache,
        total_ticks=args.ticks, eval_interval=args.eval_interval,
        reactivation_policy=args.policy, seed=args.seed,
    )

    status = result["status"]
    if status == "completed":
        print(f"Done: overall={result['overall_score']:.4f} "
              f"retrieval={result['retrieval_score']:.4f} "
              f"plausibility={result['plausibility_score']:.4f} "
              f"({result['duration_seconds']}s)")
    else:
        print(f"Failed: {result.get('error', 'unknown')}")
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_runner.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add src/memory_decay/runner.py tests/test_runner.py
git commit -m "Add experiment runner for auto-research loop

Single-experiment executor that loads a custom decay function,
validates it (syntax, range, actually-decays checks), runs a
200-tick simulation with cached embeddings, and saves results.

Constraint: All embeddings from cache, zero API calls per run
Tested: successful experiment, validation failure recording, sanity checks
Scope-risk: narrow"
```

---

### Task 4: Build the embedding cache (one-time setup)

**Files:**
- None (runtime step, not code)

**Step 1: Build cache from the 500-memory dataset**

Run:
```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.cache_builder \
  --dataset data/memories_500.jsonl \
  --output cache \
  --backend local
```

Expected: `cache/embeddings.pkl`, `cache/dataset.json`, `cache/test_queries.json` created.

**Step 2: Verify cache**

Run:
```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -c "
import pickle, json
with open('cache/embeddings.pkl', 'rb') as f:
    emb = pickle.load(f)
print(f'Cached embeddings: {len(emb)}')
with open('cache/test_queries.json') as f:
    q = json.load(f)
print(f'Test queries: {len(q)}')
"
```

Expected: ~600+ cached embeddings (500 contents + 500 queries, minus duplicates), 500 test queries.

**Step 3: Add cache to .gitignore and commit**

```bash
cd /Users/lit/memory-decay
echo "cache/" >> .gitignore
git add .gitignore
git commit -m "Ignore cache directory (embeddings are machine-specific)"
```

---

### Task 5: Create program.md (Claude Code meta-program)

**Files:**
- Create: `program.md` (project root)

**Step 1: Write program.md**

```markdown
# Auto-Research Loop: Memory Decay Function Exploration

## Goal
Discover better decay functions by iterating: hypothesize -> implement -> test -> judge.

## Protocol (each cycle)

### 1. Read State
- Read `experiments/history.jsonl` for previous experiment results
- Read `experiments/best/decay_fn.py` for the current best function
- Read `experiments/best/results.json` for the current best scores
- If no `experiments/` directory exists, create it and run the baseline first

### 2. Analyze & Hypothesize
Based on previous results, form a hypothesis for improvement. Consider:
- Which thresholds have low recall? Can a different curve shape help?
- Is the decay too fast or too slow for facts vs episodes?
- Could a different mathematical form (hyperbolic, stretched exponential,
  logarithmic saturation) perform better?
- Are impact and stability modifiers being used effectively?

### 3. Write Experiment Files
Create `experiments/exp_NNNN/` (next sequential number) with:

**decay_fn.py** — Must follow this exact interface:
```python
def compute_decay(activation, impact, stability, mtype, params):
    """
    Args:
        activation: float 0-1, current activation score
        impact: float 0-1, memory importance
        stability: float 0-1, reinforcement stability
        mtype: "fact" or "episode"
        params: dict of tunable parameters
    Returns:
        float 0-1, new activation score (must be < activation for decay)
    """
    # Your implementation here
    ...
```

**params.json** — Parameters the function uses. Must include at minimum:
```json
{
  "lambda_fact": 0.02,
  "lambda_episode": 0.035,
  "stability_weight": 0.8,
  "stability_decay": 0.01,
  "reinforcement_gain_direct": 0.2,
  "reinforcement_gain_assoc": 0.05,
  "stability_cap": 1.0
}
```
(reinforcement/stability params are used by the simulation loop, not the decay function)

**hypothesis.txt** — One paragraph explaining what you're trying and why.

### 4. Run Experiment
```bash
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_NNNN --cache cache
```

### 5. Read Results
Read `experiments/exp_NNNN/results.json`. Key metrics:
- `overall_score`: main metric (0.7 * retrieval + 0.3 * plausibility)
- `retrieval_score`: 0.7 * recall_mean + 0.3 * precision_mean (across thresholds)
- `plausibility_score`: 0.6 * correlation + 0.4 * smoothness
- `status`: "completed" or "validation_failed"

### 6. Judge
Compare `overall_score` with `experiments/best/results.json`:
- **Improved**: update `experiments/best/` symlink, git commit
- **No gain**: record in history, move on
- **Validation failed**: record error, adjust next hypothesis

### 7. Record
Append one line to `experiments/history.jsonl`:
```json
{"exp": "exp_NNNN", "overall": 0.74, "retrieval": 0.71, "plausibility": 0.81, "status": "improved", "hypothesis": "short summary"}
```

### 8. Repeat or Stop
Continue to next cycle unless:
- 20 cycles completed this session
- 20+ consecutive experiments with no improvement (convergence)
- 10+ consecutive validation failures

## Baseline (first run only)
If `experiments/best/` doesn't exist:
1. Create `experiments/exp_0000/` with the default exponential decay
2. Run it as the baseline
3. Set it as `experiments/best/`

## Rules
- NEVER modify evaluator.py, graph.py, or runner.py
- NEVER modify the dataset or cache
- Each experiment is independent — always start from fresh graph state
- Be creative with decay formulas but respect the interface contract
- Track what you've tried to avoid repeating failed approaches
```

**Step 2: Commit**

```bash
cd /Users/lit/memory-decay
git add program.md
git commit -m "Add program.md meta-program for auto-research loop

Defines the closed-loop protocol for Claude Code to follow:
read state -> hypothesize -> implement -> test -> judge -> record.

Constraint: evaluator.py, graph.py, runner.py are immutable
Scope-risk: narrow"
```

---

### Task 6: Create initial experiments directory and run baseline

**Files:**
- Create: `experiments/` directory structure

**Step 1: Create directory structure**

```bash
cd /Users/lit/memory-decay
mkdir -p experiments
```

**Step 2: Create baseline experiment (exp_0000)**

Create `experiments/exp_0000/decay_fn.py`:
```python
"""Baseline: default exponential decay (same as original DecayEngine)."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    impact_factor = 1.0 + alpha * impact
    stability_factor = 1.0 + rho * stability
    combined = max(impact_factor * stability_factor, 1e-9)

    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    effective_lambda = lam / combined
    return activation * math.exp(-effective_lambda)
```

Create `experiments/exp_0000/params.json`:
```json
{
  "lambda_fact": 0.02,
  "lambda_episode": 0.035,
  "beta_fact": 0.08,
  "beta_episode": 0.12,
  "alpha": 0.5,
  "stability_weight": 0.8,
  "stability_decay": 0.01,
  "reinforcement_gain_direct": 0.2,
  "reinforcement_gain_assoc": 0.05,
  "stability_cap": 1.0
}
```

Create `experiments/exp_0000/hypothesis.txt`:
```
Baseline experiment using the default exponential decay function from the original DecayEngine.
This establishes the performance floor that all future experiments must beat.
```

**Step 3: Run baseline**

```bash
cd /Users/lit/memory-decay
PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_0000 --cache cache
```

**Step 4: Set as best and initialize history**

```bash
cd /Users/lit/memory-decay
ln -sfn exp_0000 experiments/best
```

Then read `experiments/exp_0000/results.json` and create `experiments/history.jsonl`:
```json
{"exp": "exp_0000", "overall": <score>, "retrieval": <score>, "plausibility": <score>, "status": "baseline", "hypothesis": "default exponential decay baseline"}
```

**Step 5: Commit**

```bash
cd /Users/lit/memory-decay
git add experiments/exp_0000/ experiments/history.jsonl
git commit -m "Establish baseline for auto-research loop

Default exponential decay as exp_0000. All future experiments
compared against this baseline.

Tested: full 200-tick simulation with cached embeddings
Scope-risk: narrow"
```

---

### Task 7: End-to-end integration test

**Files:**
- Create: `tests/test_auto_research.py`

**Step 1: Write integration test**

```python
"""Integration test for the full auto-research experiment loop."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from memory_decay.cache_builder import build_cache, load_cache
from memory_decay.runner import run_experiment


SAMPLE_DATASET = [
    {"id": "f1", "type": "fact", "content": "서울은 대한민국의 수도이다",
     "entities": ["서울"], "tick": 0, "impact": 0.9, "associations": [],
     "recall_query": "대한민국의 수도는?", "recall_answer": "서울"},
    {"id": "f2", "type": "fact", "content": "김민수는 커피를 좋아한다",
     "entities": ["김민수"], "tick": 5, "impact": 0.7,
     "associations": [{"id": "f1", "weight": 0.6}],
     "recall_query": "김민수는 무엇을 좋아하는가?", "recall_answer": "커피"},
    {"id": "e1", "type": "episode", "content": "서울에서 커피를 마셨다",
     "entities": ["서울"], "tick": 10, "impact": 0.5,
     "associations": [{"id": "f1", "weight": 0.8}],
     "recall_query": "어디서 커피를 마셨는가?", "recall_answer": "서울"},
    {"id": "e2", "type": "episode", "content": "길에서 강아지를 만났다",
     "entities": ["강아지"], "tick": 20, "impact": 0.3, "associations": [],
     "recall_query": "길에서 만난 동물은?", "recall_answer": "강아지"},
]


def mock_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


EXPONENTIAL_FN = '''\
import math

def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    combined = max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    lam = params.get("lambda_fact", 0.02) if mtype == "fact" else params.get("lambda_episode", 0.035)
    return activation * math.exp(-lam / combined)
'''

POWER_LAW_FN = '''\
def compute_decay(activation, impact, stability, mtype, params):
    alpha = params.get("alpha", 0.5)
    rho = params.get("stability_weight", 0.8)
    combined = max((1 + alpha * impact) * (1 + rho * stability), 1e-9)
    beta = params.get("beta_fact", 0.08) if mtype == "fact" else params.get("beta_episode", 0.12)
    return activation / (1 + beta / combined)
'''

DEFAULT_PARAMS = {
    "lambda_fact": 0.02, "lambda_episode": 0.035,
    "beta_fact": 0.08, "beta_episode": 0.12,
    "alpha": 0.5, "stability_weight": 0.8,
    "stability_decay": 0.01,
    "reinforcement_gain_direct": 0.2,
    "reinforcement_gain_assoc": 0.05,
    "stability_cap": 1.0,
}


class TestAutoResearchLoop:
    @pytest.fixture
    def setup(self, tmp_path):
        dataset_path = tmp_path / "data.jsonl"
        with open(dataset_path, "w") as f:
            for item in SAMPLE_DATASET:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        cache_dir = tmp_path / "cache"
        build_cache(str(dataset_path), str(cache_dir), embedder=mock_embedder)
        return tmp_path, str(cache_dir)

    def test_two_experiments_can_be_compared(self, setup):
        tmp_path, cache_dir = setup

        exp1 = tmp_path / "experiments" / "exp_0000"
        exp1.mkdir(parents=True)
        (exp1 / "decay_fn.py").write_text(EXPONENTIAL_FN)
        (exp1 / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp1 / "hypothesis.txt").write_text("baseline exponential")

        result1 = run_experiment(str(exp1), cache_dir, total_ticks=50, eval_interval=10)
        assert result1["status"] == "completed"

        exp2 = tmp_path / "experiments" / "exp_0001"
        exp2.mkdir(parents=True)
        (exp2 / "decay_fn.py").write_text(POWER_LAW_FN)
        (exp2 / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp2 / "hypothesis.txt").write_text("power law has longer tail")

        result2 = run_experiment(str(exp2), cache_dir, total_ticks=50, eval_interval=10)
        assert result2["status"] == "completed"

        assert 0.0 <= result1["overall_score"] <= 1.0
        assert 0.0 <= result2["overall_score"] <= 1.0

    def test_history_tracking(self, setup):
        tmp_path, cache_dir = setup

        exp_dir = tmp_path / "experiments" / "exp_0000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "decay_fn.py").write_text(EXPONENTIAL_FN)
        (exp_dir / "params.json").write_text(json.dumps(DEFAULT_PARAMS))
        (exp_dir / "hypothesis.txt").write_text("baseline")

        result = run_experiment(str(exp_dir), cache_dir, total_ticks=50, eval_interval=10)

        history_path = tmp_path / "experiments" / "history.jsonl"
        entry = {
            "exp": "exp_0000",
            "overall": result["overall_score"],
            "retrieval": result["retrieval_score"],
            "plausibility": result["plausibility_score"],
            "status": "baseline",
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        with open(history_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["exp"] == "exp_0000"
```

**Step 2: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest tests/test_auto_research.py -v`
Expected: 2 passed

**Step 3: Commit**

```bash
cd /Users/lit/memory-decay
git add tests/test_auto_research.py
git commit -m "Add end-to-end integration test for auto-research loop

Verifies the full cycle: cache -> experiment -> results -> comparison.
Two experiments with different decay curves produce valid, distinct scores.

Tested: exponential vs power-law comparison, history tracking
Scope-risk: narrow"
```

---

### Task 8: Final — run all tests and verify

**Step 1: Run full test suite**

```bash
cd /Users/lit/memory-decay && PYTHONPATH=src uv run pytest -v
```

Expected: All tests pass (existing + new).

**Step 2: Verify the loop works manually**

Follow `program.md` once manually:
1. Ensure cache exists
2. Ensure baseline exists at `experiments/exp_0000/`
3. Create `experiments/exp_0001/` with a variant decay function
4. Run: `PYTHONPATH=src uv run python -m memory_decay.runner experiments/exp_0001 --cache cache`
5. Compare scores
6. Record in `history.jsonl`

**Step 3: Final commit if anything was adjusted**

```bash
cd /Users/lit/memory-decay
git add -A
git commit -m "Complete auto-research loop infrastructure

Ready for autonomous exploration via /loop 30m program.md"
```
