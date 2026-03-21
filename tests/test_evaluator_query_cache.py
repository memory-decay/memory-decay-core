"""Evaluator query-cache regression tests."""

import numpy as np

from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.graph import MemoryGraph


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def _setup():
    graph = MemoryGraph(embedder=_fixed_embedder)
    graph.add_memory("target", "fact", "커피는 에티오피아가 원산지다", 0.8, created_tick=0)
    graph.add_memory(
        "assoc",
        "fact",
        "에티오피아는 아프리카에 있다",
        0.3,
        created_tick=0,
        associations=[("target", 0.7)],
    )
    graph.add_memory("other", "fact", "파이썬은 프로그래밍 언어다", 0.3, created_tick=0)
    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    queries = [("커피 원산지", "target")]
    return graph, engine, evaluator, queries


def test_score_summary_reuses_similarity_results_within_tick(monkeypatch):
    graph, _, evaluator, queries = _setup()
    real_query = graph.query_by_similarity
    call_count = 0

    def counting_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(graph, "query_by_similarity", counting_query)

    evaluator.score_summary(queries, threshold=0.1, top_k=10)

    assert call_count == 1


def test_query_cache_is_tick_scoped(monkeypatch):
    graph, engine, evaluator, queries = _setup()
    real_query = graph.query_by_similarity
    call_count = 0

    def counting_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(graph, "query_by_similarity", counting_query)

    evaluator.score_summary(queries, threshold=0.1, top_k=10)
    engine.tick()
    evaluator.score_summary(queries, threshold=0.1, top_k=10)

    assert call_count == 2


def test_score_summary_can_record_snapshot_history():
    _, _, evaluator, queries = _setup()

    assert len(evaluator.history) == 0

    summary = evaluator.score_summary(queries, threshold=0.1, top_k=10, record_snapshot=True)

    assert len(evaluator.history) == 1
    assert summary["tick"] == 0
