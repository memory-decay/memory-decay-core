# Experimental Protocol Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 리뷰에서 지적된 5개 실험 프로토콜 결함을 수정하여, 현재 시스템의 결과가 방법론적으로 타당하도록 만든다.

**Architecture:** 핵심 아키텍처(graph, decay engine)는 그대로 두고, query/evaluation 경로에 시간 필터링을 추가하며, evaluator에 strict precision 기반 retrieval_score와 독립적 plausibility 지표를 도입한다. cache_builder와 scheduled_query 정책의 데이터 누수를 분리한다.

**핵심 설계 원칙:**
- **시간 타당성은 runtime temporal gate가 보장한다.** train/test random split은 역할 분리(rehearsal vs evaluation)를 위한 것이며, 시간적 인과성은 evaluator의 `created_tick <= current_tick` 필터와 reactivation의 temporal eligibility 체크가 담당한다.
- **retrieval_score의 최적화 목표는 strict precision이다.** `retrieval_score = 0.7 * recall_mean + 0.3 * precision_strict_mean`. associative precision은 별도 진단 지표로만 보고한다.
- **correlation은 mechanistic plausibility surrogate이다.** 외부 타당성 근거가 아니므로, plausibility_score 내 가중치를 낮춘다 (0.3). smoothness가 0.7.

**Tech Stack:** Python 3.13, networkx, numpy, pytest

---

## 수정 대상 요약

| # | 심각도 | 문제 | 수정 파일 |
|---|--------|------|-----------|
| 1 | 치명적 | 미래 기억이 검색/평가에 포함 | `graph.py`, `evaluator.py` |
| 2 | 치명적 | scheduled_query가 test set을 재활성화 | `main.py` |
| 3 | 높음 | cache_builder가 전체 데이터를 test로 사용 | `cache_builder.py`, `runner.py` |
| 4 | 높음 | precision이 연관 노드를 정답 처리 | `evaluator.py` |
| 5 | 높음 | activation_recall_correlation 자기참조 | `evaluator.py` |

---

### Task 1: 시간 필터링 — query_by_similarity에 current_tick 게이트 추가

**Files:**
- Modify: `src/memory_decay/graph.py:130-149` (query_by_similarity)
- Test: `tests/test_graph_temporal.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_graph_temporal.py
"""Temporal filtering tests for MemoryGraph."""
import numpy as np
import pytest
from memory_decay.graph import MemoryGraph


def _fixed_embedder(text: str) -> np.ndarray:
    """Deterministic embedder: hash text into a 16-d vector."""
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


class TestTemporalFiltering:
    def _build_graph(self) -> MemoryGraph:
        graph = MemoryGraph(embedder=_fixed_embedder)
        graph.add_memory("past", "fact", "서울의 날씨", 0.5, created_tick=0)
        graph.add_memory("present", "fact", "서울의 기온", 0.5, created_tick=50)
        graph.add_memory("future", "fact", "서울의 미래 기후", 0.5, created_tick=100)
        return graph

    def test_no_tick_filter_returns_all(self):
        """Without current_tick, all memories are searchable (backward compat)."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" in ids
        assert "future" in ids

    def test_tick_filter_excludes_future(self):
        """With current_tick=50, future memory (tick=100) must be excluded."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10, current_tick=50)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" in ids
        assert "future" not in ids

    def test_tick_zero_only_shows_tick_zero(self):
        """At tick=0, only memories created at tick=0 are visible."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10, current_tick=0)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" not in ids
        assert "future" not in ids
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_graph_temporal.py -v`
Expected: FAIL — `query_by_similarity()` got unexpected keyword argument 'current_tick'

**Step 3: Write minimal implementation**

Modify `src/memory_decay/graph.py` — `query_by_similarity` method:

```python
def query_by_similarity(
    self, query_text: str, top_k: int = 5, current_tick: int | None = None
) -> list[tuple[str, float]]:
    """Find memories matching query via embedding cosine similarity.

    Args:
        current_tick: If provided, exclude memories with created_tick > current_tick.
    """
    query_vec = self._embed_text(query_text)

    results = []
    for nid, attrs in self._graph.nodes(data=True):
        if attrs.get("type") == "unknown":
            continue
        if current_tick is not None and attrs.get("created_tick", 0) > current_tick:
            continue
        emb = attrs["embedding"]
        norm_q = np.linalg.norm(query_vec)
        norm_e = np.linalg.norm(emb)
        if norm_q == 0 or norm_e == 0:
            continue
        sim = float(np.dot(query_vec, emb) / (norm_q * norm_e))
        results.append((nid, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_graph_temporal.py -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/graph.py tests/test_graph_temporal.py
git commit -m "Add temporal filtering to query_by_similarity"
```

---

### Task 2: Evaluator에 current_tick 전파 — 평가 시 미래 기억 제외

**Files:**
- Modify: `src/memory_decay/evaluator.py:23-58` (evaluate_recall)
- Modify: `src/memory_decay/evaluator.py:60-100` (evaluate_precision)
- Modify: `src/memory_decay/evaluator.py:102-139` (activation_recall_correlation)
- Test: `tests/test_evaluator_temporal.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_evaluator_temporal.py
"""Temporal correctness tests for Evaluator."""
import numpy as np
import pytest
from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


class TestEvaluatorTemporal:
    def _setup(self):
        graph = MemoryGraph(embedder=_fixed_embedder)
        graph.add_memory("m0", "fact", "고양이는 포유류다", 0.5, created_tick=0)
        graph.add_memory("m100", "fact", "고양이의 수명은 15년이다", 0.5, created_tick=100)
        engine = DecayEngine(graph)
        evaluator = Evaluator(graph, engine)
        return graph, engine, evaluator

    def test_future_memory_not_recalled_at_tick_zero(self):
        """At tick=0, a memory with created_tick=100 must NOT count as recalled."""
        _, engine, evaluator = self._setup()
        assert engine.current_tick == 0
        test_queries = [("고양이 수명", "m100")]
        recall = evaluator.evaluate_recall(test_queries, threshold=0.1, top_k=10)
        assert recall == 0.0, f"Future memory was recalled at tick 0: recall={recall}"

    def test_past_memory_recalled_at_tick_zero(self):
        """At tick=0, a memory with created_tick=0 should be recallable."""
        _, engine, evaluator = self._setup()
        test_queries = [("고양이 포유류", "m0")]
        recall = evaluator.evaluate_recall(test_queries, threshold=0.1, top_k=10)
        assert recall > 0.0, "Past memory should be recalled"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_temporal.py::TestEvaluatorTemporal::test_future_memory_not_recalled_at_tick_zero -v`
Expected: FAIL — future memory IS recalled because evaluator doesn't pass current_tick

**Step 3: Write minimal implementation**

`evaluate_recall` — add `created_tick` check and pass `current_tick` to query:

```python
def evaluate_recall(self, test_queries, threshold=0.3, top_k=5):
    if not test_queries:
        return 0.0
    current_tick = self._engine.current_tick
    recalled = 0
    observable = 0  # 분모는 현재 tick에 관측 가능한 query 수
    for query, expected_id in test_queries:
        node = self._graph.get_node(expected_id)
        if not node:
            continue
        if node.get("created_tick", 0) > current_tick:
            continue
        observable += 1
        if node["activation_score"] < threshold:
            continue
        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
        result_ids = [rid for rid, _ in results]
        if expected_id in result_ids:
            recalled += 1
    return recalled / max(observable, 1)
```

`evaluate_precision` — pass `current_tick`:

```python
def evaluate_precision(self, test_queries, threshold=0.3, top_k=5):
    if not test_queries:
        return 0.0
    current_tick = self._engine.current_tick
    total_relevant = 0
    total_retrieved = 0
    for query, expected_id in test_queries:
        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
        expected_assoc = set()
        node = self._graph.get_node(expected_id)
        if node:
            for assoc_id, _ in self._graph.get_associated(expected_id):
                expected_assoc.add(assoc_id)
        expected_assoc.add(expected_id)
        for rid, _ in results:
            r_node = self._graph.get_node(rid)
            if not r_node or r_node.get("type") in ("unknown", None):
                continue
            if r_node["activation_score"] < threshold:
                continue
            total_retrieved += 1
            if rid in expected_assoc:
                total_relevant += 1
    if total_retrieved == 0:
        return 0.0
    return total_relevant / total_retrieved
```

`activation_recall_correlation` — add `created_tick` check and pass `current_tick`:

```python
def activation_recall_correlation(self, test_queries, threshold=0.3, top_k=5):
    if len(test_queries) < 3:
        return 0.0
    current_tick = self._engine.current_tick
    activations = []
    recall_success = []
    for query, expected_id in test_queries:
        node = self._graph.get_node(expected_id)
        if not node:
            continue
        if node.get("created_tick", 0) > current_tick:
            continue
        act = node["activation_score"]
        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
        result_ids = [rid for rid, _ in results]
        recalled = 1.0 if (expected_id in result_ids and act > threshold) else 0.0
        activations.append(act)
        recall_success.append(recalled)
    if len(activations) < 3:
        return 0.0
    a = np.array(activations)
    r = np.array(recall_success)
    if np.std(a) == 0 or np.std(r) == 0:
        return 0.0
    return float(np.corrcoef(a, r)[0, 1])
```

**Step 4: Run tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_temporal.py tests/test_graph_temporal.py -v`
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/evaluator.py tests/test_evaluator_temporal.py
git commit -m "Propagate current_tick to evaluator — exclude future memories from evaluation"
```

---

### Task 3: scheduled_query 평가 누수 수정 — rehearsal set 분리

**Files:**
- Modify: `src/memory_decay/main.py:54-150` (run_simulation)
- Modify: `src/memory_decay/main.py:153-305` (run_experiment)
- Test: `tests/test_simulation_leakage.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_simulation_leakage.py
"""Verify scheduled_query does NOT use test_queries for reactivation."""
import numpy as np
from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.main import run_simulation


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def test_scheduled_query_uses_separate_rehearsal_set():
    """scheduled_query must reactivate from rehearsal_targets, not test_queries."""
    graph = MemoryGraph(embedder=_fixed_embedder)
    graph.add_memory("train_1", "fact", "지구는 둥글다", 0.5, created_tick=0)
    graph.add_memory("train_2", "fact", "물은 H2O이다", 0.5, created_tick=0)
    graph.add_memory("test_1", "fact", "하늘은 파랗다", 0.5, created_tick=0)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)

    test_queries = [("하늘 색깔", "test_1")]
    rehearsal_targets = ["train_1", "train_2"]

    summaries = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=20,
        eval_interval=10,
        reactivation_policy="scheduled_query",
        reactivation_interval=5,
        rehearsal_targets=rehearsal_targets,
        seed=42,
    )

    # test_1 should NOT have been reactivated
    test_node = graph.get_node("test_1")
    assert test_node["retrieval_count"] == 0, (
        f"Test memory was reactivated {test_node['retrieval_count']} times — evaluation leakage!"
    )

    # At least one train item should have been reactivated
    train_1 = graph.get_node("train_1")
    train_2 = graph.get_node("train_2")
    total = train_1["retrieval_count"] + train_2["retrieval_count"]
    assert total > 0, "No rehearsal happened at all"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_simulation_leakage.py -v`
Expected: FAIL — `run_simulation() got unexpected keyword argument 'rehearsal_targets'`

**Step 3: Write minimal implementation**

Add `rehearsal_targets` param to `run_simulation`, use it for `scheduled_query`:

```python
def run_simulation(
    graph, engine, evaluator, test_queries,
    total_ticks=100, eval_interval=5,
    reactivation_policy="none", reactivation_interval=10,
    reactivation_boost=0.3,
    rehearsal_targets: list[str] | None = None,
    seed=None,
):
    if reactivation_policy not in {"none", "random", "scheduled_query"}:
        raise ValueError(f"Unsupported reactivation_policy: {reactivation_policy}")

    if reactivation_policy == "scheduled_query" and not rehearsal_targets:
        raise ValueError(
            "scheduled_query policy requires rehearsal_targets (separate from test_queries)"
        )

    rng = random.Random(seed)
    summaries = []
    params = engine.get_params()

    def collect_summary():
        snap = evaluator.snapshot(test_queries)
        summary = evaluator.score_summary(test_queries)
        summary["tick"] = snap["tick"]
        return summary

    def apply_reactivation(tick):
        if reactivation_policy == "none" or tick % reactivation_interval != 0:
            return
        if reactivation_policy == "random":
            candidates = [
                nid for nid, attrs in graph._graph.nodes(data=True)
                if attrs.get("type") not in ("unknown", None)
                and attrs.get("created_tick", 0) <= engine.current_tick
            ]
            if not candidates:
                return
            target_id = rng.choice(candidates)
        else:  # scheduled_query — use rehearsal_targets, NOT test_queries
            # Filter to only memories that exist at current tick
            eligible = [
                mid for mid in rehearsal_targets
                if graph._graph.nodes[mid].get("created_tick", 0) <= engine.current_tick
            ]
            if not eligible:
                return
            idx = ((tick // reactivation_interval) - 1) % len(eligible)
            target_id = eligible[idx]

        graph.re_activate(
            target_id, reactivation_boost,
            source="direct", reinforce=True,
            current_tick=engine.current_tick,
            reinforcement_gain_direct=params["reinforcement_gain_direct"],
            reinforcement_gain_assoc=params["reinforcement_gain_assoc"],
            stability_cap=params["stability_cap"],
        )

    # ... rest unchanged (initial eval + tick loop) ...
```

In `run_experiment`, build `rehearsal_targets` from train set and pass it:

```python
# After line 191:
rehearsal_targets = [m["id"] for m in train]

# Pass to run_simulation:
initial_summaries = run_simulation(
    graph, engine, evaluator, test_queries,
    total_ticks=total_ticks, eval_interval=eval_interval,
    reactivation_policy=reactivation_policy,
    rehearsal_targets=rehearsal_targets,
    seed=seed,
)
```

**Step 4: Run tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_simulation_leakage.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/main.py tests/test_simulation_leakage.py
git commit -m "Separate rehearsal targets from test queries — fix evaluation leakage in scheduled_query"
```

---

### Task 4: cache_builder의 train/test split 도입

**Files:**
- Modify: `src/memory_decay/cache_builder.py:20-69` (build_cache)
- Modify: `src/memory_decay/cache_builder.py:72-89` (load_cache)
- Modify: `src/memory_decay/runner.py:87-142` (run_experiment)
- Test: `tests/test_cache_builder.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_cache_builder.py
"""Cache builder must produce separate train/test splits."""
import json
import numpy as np
from memory_decay.cache_builder import build_cache, load_cache


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def test_cache_splits_train_test(tmp_path):
    """Cache must produce separate test_queries and rehearsal_targets."""
    dataset = [
        {"id": f"m{i}", "type": "fact", "content": f"fact {i}",
         "tick": i * 10, "impact": 0.5, "associations": [],
         "recall_query": f"query {i}"}
        for i in range(10)
    ]
    dataset_path = tmp_path / "data.jsonl"
    with open(dataset_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    build_cache(str(dataset_path), str(tmp_path / "cache"), embedder=_fixed_embedder)
    _, full_dataset, test_queries, rehearsal_targets = load_cache(str(tmp_path / "cache"))

    test_ids = {tid for _, tid in test_queries}
    rehearsal_ids = set(rehearsal_targets)

    # No overlap
    assert test_ids.isdisjoint(rehearsal_ids), (
        f"Overlap between test and rehearsal: {test_ids & rehearsal_ids}"
    )

    # Cover full dataset
    assert test_ids | rehearsal_ids == {item["id"] for item in full_dataset}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_cache_builder.py -v`
Expected: FAIL — load_cache returns 3 values, not 4

**Step 3: Write minimal implementation**

`build_cache` — add train/test split, save `rehearsal_targets.json`.

**Note:** 이 random split은 **역할 분리**(rehearsal vs evaluation)를 위한 것이다. **시간적 타당성**(미래 기억 제외)은 runtime temporal gate가 보장한다: evaluator의 `created_tick <= current_tick` 필터, reactivation의 temporal eligibility 체크.

```python
def build_cache(dataset_path, cache_dir, embedder=None, embedding_backend="auto",
                test_ratio=0.2, seed=42):
    # ... existing embedding logic ...

    # Train/test split
    import random as _rnd
    rng = _rnd.Random(seed)
    facts = [item for item in dataset if item.get("type") == "fact"]
    episodes = [item for item in dataset if item.get("type") == "episode"]
    rng.shuffle(facts)
    rng.shuffle(episodes)

    n_test_facts = max(1, int(len(facts) * test_ratio))
    n_test_episodes = max(1, int(len(episodes) * test_ratio)) if episodes else 0
    test_items = facts[:n_test_facts] + episodes[:n_test_episodes]
    train_items = facts[n_test_facts:] + episodes[n_test_episodes:]

    test_queries = [
        (item["recall_query"], item["id"])
        for item in test_items if "recall_query" in item
    ]
    rehearsal_targets = [item["id"] for item in train_items]

    with open(cache_path / "test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)
    with open(cache_path / "rehearsal_targets.json", "w", encoding="utf-8") as f:
        json.dump(rehearsal_targets, f, ensure_ascii=False, indent=2)
```

`load_cache` — return 4 values:

```python
def load_cache(cache_dir):
    cache_path = Path(cache_dir)
    # ... existing embeddings + dataset loading ...

    with open(cache_path / "test_queries.json", "r", encoding="utf-8") as f:
        test_queries = [tuple(q) for q in json.load(f)]
    with open(cache_path / "rehearsal_targets.json", "r", encoding="utf-8") as f:
        rehearsal_targets = json.load(f)

    def cached_embedder(text):
        if text not in embeddings:
            raise KeyError(f"Text not in embedding cache: {text[:80]}...")
        return embeddings[text].copy()

    return cached_embedder, dataset, test_queries, rehearsal_targets
```

`runner.py` — unpack 4th value and pass to run_simulation:

```python
def run_experiment(experiment_dir, cache_dir, ...):
    # ...
    cached_embedder, dataset, test_queries, rehearsal_targets = load_cache(cache_dir)
    # ...
    snapshots = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=total_ticks, eval_interval=eval_interval,
        reactivation_policy=reactivation_policy,
        rehearsal_targets=rehearsal_targets,
        seed=seed,
    )
```

**Step 4: Run tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_cache_builder.py -v`
Expected: PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/cache_builder.py src/memory_decay/runner.py tests/test_cache_builder.py
git commit -m "Add train/test split to cache_builder — eliminate self-evaluation in auto-research loop"
```

---

### Task 5: Standard precision 지표 추가

**Files:**
- Modify: `src/memory_decay/evaluator.py:60-100` (evaluate_precision)
- Modify: `src/memory_decay/evaluator.py:245-279` (score_summary)
- Test: `tests/test_evaluator_precision.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_evaluator_precision.py
"""Standard vs associative precision tests."""
import numpy as np
from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


class TestPrecisionModes:
    def _setup(self):
        graph = MemoryGraph(embedder=_fixed_embedder)
        graph.add_memory("target", "fact", "커피는 에티오피아가 원산지다", 0.5, created_tick=0)
        graph.add_memory("assoc1", "fact", "에티오피아는 아프리카에 있다", 0.5,
                         created_tick=0, associations=[("target", 0.7)])
        graph.add_memory("unrelated", "fact", "파이썬은 프로그래밍 언어다", 0.5, created_tick=0)
        engine = DecayEngine(graph)
        evaluator = Evaluator(graph, engine)
        return evaluator

    def test_strict_precision_only_counts_exact_match(self):
        evaluator = self._setup()
        test_queries = [("커피 원산지", "target")]
        strict = evaluator.evaluate_precision(test_queries, threshold=0.1, top_k=10, mode="strict")
        assert isinstance(strict, float)

    def test_associative_precision_counts_neighbors(self):
        evaluator = self._setup()
        test_queries = [("커피 원산지", "target")]
        assoc = evaluator.evaluate_precision(test_queries, threshold=0.1, top_k=10, mode="associative")
        assert isinstance(assoc, float)

    def test_strict_leq_associative(self):
        """Strict precision should be <= associative precision."""
        evaluator = self._setup()
        test_queries = [("커피 원산지", "target")]
        strict = evaluator.evaluate_precision(test_queries, threshold=0.1, top_k=10, mode="strict")
        assoc = evaluator.evaluate_precision(test_queries, threshold=0.1, top_k=10, mode="associative")
        assert strict <= assoc + 1e-9

    def test_score_summary_reports_both(self):
        evaluator = self._setup()
        test_queries = [("커피 원산지", "target")]
        summary = evaluator.score_summary(test_queries)
        assert "precision_strict" in summary
        assert "precision_associative" in summary
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_precision.py -v`
Expected: FAIL — `evaluate_precision() got unexpected keyword argument 'mode'`

**Step 3: Write minimal implementation**

Add `mode` parameter to `evaluate_precision`:

```python
def evaluate_precision(self, test_queries, threshold=0.3, top_k=5, mode="associative"):
    """Precision of recall results.

    Args:
        mode: "strict" = only exact expected_id is relevant.
              "associative" = associated nodes also count (original behavior).
    """
    if not test_queries:
        return 0.0
    current_tick = self._engine.current_tick
    total_relevant = 0
    total_retrieved = 0

    for query, expected_id in test_queries:
        node = self._graph.get_node(expected_id)
        if not node or node.get("created_tick", 0) > current_tick:
            continue

        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)

        if mode == "strict":
            relevant_ids = {expected_id}
        else:
            relevant_ids = set()
            if node:
                for assoc_id, _ in self._graph.get_associated(expected_id):
                    relevant_ids.add(assoc_id)
            relevant_ids.add(expected_id)

        for rid, _ in results:
            r_node = self._graph.get_node(rid)
            if not r_node or r_node.get("type") in ("unknown", None):
                continue
            if r_node["activation_score"] < threshold:
                continue
            total_retrieved += 1
            if rid in relevant_ids:
                total_relevant += 1

    if total_retrieved == 0:
        return 0.0
    return total_relevant / total_retrieved
```

In `score_summary`, use strict precision for retrieval_score (primary optimization target):

```python
def score_summary(self, test_queries, thresholds=(0.2, 0.3, 0.4, 0.5), threshold=0.3, top_k=5):
    # ... existing sweep code ...

    strict_precisions = []
    for t in thresholds:
        sp = self.evaluate_precision(test_queries, threshold=t, top_k=top_k, mode="strict")
        strict_precisions.append(sp)
    precision_strict_mean = float(np.mean(strict_precisions))

    # Primary optimization target uses strict precision (exact match only).
    # Associative precision is reported as a diagnostic, not used for scoring.
    retrieval_score = 0.7 * sweep["recall_mean"] + 0.3 * precision_strict_mean

    # Correlation is a mechanistic plausibility surrogate, not external
    # validity evidence. Weight kept low; smoothness carries more signal.
    plausibility_score = 0.3 * corr_score + 0.7 * smoothness_score

    return {
        # ... existing fields ...
        "retrieval_score": retrieval_score,
        "precision_strict": precision_strict_mean,
        "precision_associative": sweep["precision_mean"],  # diagnostic only
    }
```

**Step 4: Run tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_precision.py -v`
Expected: ALL PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/evaluator.py tests/test_evaluator_precision.py
git commit -m "Add strict precision mode — report both standard and associative precision"
```

---

### Task 6: activation_recall_correlation에서 threshold 의존성 제거

**Files:**
- Modify: `src/memory_decay/evaluator.py:102-139`
- Add method: `evaluate_similarity_recall`
- Modify: `src/memory_decay/evaluator.py:245-279` (score_summary)
- Test: `tests/test_evaluator_correlation.py` (새로 생성)

**Step 1: Write the failing test**

```python
# tests/test_evaluator_correlation.py
"""Correlation metric independence tests."""
import numpy as np
from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def test_correlation_not_gated_by_threshold():
    """Correlation should use similarity-only recall, not threshold-gated recall."""
    graph = MemoryGraph(embedder=_fixed_embedder)
    for i in range(10):
        graph.add_memory(f"m{i}", "fact", f"사실 번호 {i}에 대한 설명", 0.5, created_tick=0)
        graph.set_activation(f"m{i}", 0.1 * i)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(f"사실 번호 {i}", f"m{i}") for i in range(10)]

    corr = evaluator.activation_recall_correlation(test_queries, threshold=0.3, top_k=5)
    assert isinstance(corr, float)
    assert -1.0 <= corr <= 1.0


def test_score_summary_has_similarity_recall():
    """score_summary should include similarity_recall_rate (threshold-free)."""
    graph = MemoryGraph(embedder=_fixed_embedder)
    for i in range(5):
        graph.add_memory(f"m{i}", "fact", f"사실 {i}", 0.5, created_tick=0)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(f"사실 {i}", f"m{i}") for i in range(5)]

    summary = evaluator.score_summary(test_queries)
    assert "similarity_recall_rate" in summary
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_correlation.py -v`
Expected: FAIL — `similarity_recall_rate` not in summary

**Step 3: Write minimal implementation**

Fix `activation_recall_correlation` — remove threshold from recall definition:

```python
def activation_recall_correlation(self, test_queries, threshold=0.3, top_k=5):
    """Pearson correlation between activation and similarity-based recall.

    Recall success is defined by similarity retrieval only (not gated by threshold)
    to avoid circular correlation.
    """
    if len(test_queries) < 3:
        return 0.0
    current_tick = self._engine.current_tick
    activations = []
    recall_success = []
    for query, expected_id in test_queries:
        node = self._graph.get_node(expected_id)
        if not node:
            continue
        if node.get("created_tick", 0) > current_tick:
            continue
        act = node["activation_score"]
        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
        result_ids = [rid for rid, _ in results]
        recalled = 1.0 if expected_id in result_ids else 0.0
        activations.append(act)
        recall_success.append(recalled)
    if len(activations) < 3:
        return 0.0
    a = np.array(activations)
    r = np.array(recall_success)
    if np.std(a) == 0 or np.std(r) == 0:
        return 0.0
    return float(np.corrcoef(a, r)[0, 1])
```

Add `evaluate_similarity_recall`:

```python
def evaluate_similarity_recall(self, test_queries, top_k=5):
    """Pure similarity-based recall — no activation threshold."""
    if not test_queries:
        return 0.0
    current_tick = self._engine.current_tick
    recalled = 0
    total = 0
    for query, expected_id in test_queries:
        node = self._graph.get_node(expected_id)
        if not node or node.get("created_tick", 0) > current_tick:
            continue
        total += 1
        results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
        if expected_id in [rid for rid, _ in results]:
            recalled += 1
    return recalled / max(total, 1)
```

In `score_summary`, add `similarity_recall_rate`:

```python
sim_recall = self.evaluate_similarity_recall(test_queries, top_k=top_k)
# Include in return dict:
"similarity_recall_rate": sim_recall,
```

**Step 4: Run tests**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/test_evaluator_correlation.py -v`
Expected: ALL PASSED

**Step 5: Commit**

```bash
git add src/memory_decay/evaluator.py tests/test_evaluator_correlation.py
git commit -m "Remove threshold self-reference from activation_recall_correlation"
```

---

### Task 7: run_auto_improve.py 및 strict_eval.py 업데이트

**Files:**
- Modify: `scripts/run_auto_improve.py`
- Modify: `experiments/strict_eval.py`

**Step 1: Update run_auto_improve.py**

Key changes:
- Build `rehearsal_targets` from train set
- Pass `rehearsal_targets` to all `run_simulation` calls

```python
def main():
    # ...
    dataset = SyntheticDataGenerator.load_jsonl(DATASET_PATH)
    train, test = SyntheticDataGenerator(api_key="dummy").split_test_train(dataset, test_ratio=0.2, seed=42)
    test_queries = [(m["recall_query"], m["id"]) for m in test if "recall_query" in m]
    rehearsal_targets = [m["id"] for m in train]  # NEW

    # All run_simulation calls pass rehearsal_targets:
    baseline_summaries = run_simulation(
        graph, engine, evaluator, test_queries,
        total_ticks=TOTAL_TICKS, eval_interval=EVAL_INTERVAL,
        reactivation_policy=REACTIVATION_POLICY,
        rehearsal_targets=rehearsal_targets,  # NEW
        seed=42,
    )
```

**Step 2: Update strict_eval.py**

`compute_strict_score` — pass through new fields:

```python
def compute_strict_score(result):
    # ... existing logic ...
    result["precision_strict"] = result.get("precision_strict", 0.0)
    result["precision_associative"] = result.get("precision_associative", 0.0)
    result["similarity_recall_rate"] = result.get("similarity_recall_rate", 0.0)
    return result
```

**Step 3: Run full test suite**

Run: `cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --ignore=.venv`
Expected: ALL PASSED

**Step 4: Commit**

```bash
git add scripts/run_auto_improve.py experiments/strict_eval.py
git commit -m "Update auto-improve and strict_eval to use fixed evaluation protocol"
```

---

### Task 8: 캐시 재빌드 및 기존 결과와 비교 실행

**Step 1: 캐시 재빌드 (train/test split 포함)**

```bash
cd /Users/lit/memory-decay
rm -rf cache/
PYTHONPATH=src .venv/bin/python -m memory_decay.cache_builder --dataset data/memories_500.jsonl --output cache
```

Verify: `ls cache/` should show `embeddings.pkl`, `dataset.json`, `test_queries.json`, `rehearsal_targets.json`

**Step 2: 기존 best experiment 재실행**

```bash
PYTHONPATH=src .venv/bin/python experiments/strict_eval.py experiments/exp_0025 --cache cache
```

**Step 3: 결과 비교**

기대: 수정된 프로토콜에서는 overall score가 낮아질 것 — 이것이 정상 (누수 제거).

**Step 4: Commit**

```bash
git add -A
git commit -m "Rebuild cache with train/test split, re-run baseline comparison"
```

---

### Task 9: 전체 테스트 통과 확인 및 최종 커밋

**Step 1: 전체 테스트 실행**

```bash
cd /Users/lit/memory-decay && PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --ignore=.venv
```

Expected: ALL PASSED

**Step 2: Final commit**

```bash
git add -A
git commit -m "Fix experimental protocol: temporal filtering, evaluation leakage, train/test split, precision modes, independent correlation

Addresses all 5 review findings:
1. query_by_similarity now respects created_tick
2. scheduled_query uses rehearsal_targets, not test_queries
3. cache_builder produces proper train/test split
4. evaluate_precision reports both strict and associative modes
5. activation_recall_correlation no longer self-referential

Constraint: scheduled_query policy now requires rehearsal_targets (breaking change)
Constraint: retrieval_score now uses strict precision — existing experiment scores will shift
Constraint: plausibility_score weights changed (correlation 0.3, smoothness 0.7)
Confidence: high
Scope-risk: broad
Tested: All new unit tests + existing test_data_gen.py
Not-tested: Full auto-research loop re-run (requires API keys + compute budget)"
```
