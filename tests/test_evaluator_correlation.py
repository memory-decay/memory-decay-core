"""Correlation metric independence tests."""
import numpy as np
import pytest
from memory_decay.graph import MemoryGraph
from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def test_correlation_not_gated_by_threshold():
    graph = MemoryGraph(embedder=_fixed_embedder)
    for i in range(10):
        graph.add_memory(f"m{i}", "fact", f"사실 번호 {i}에 대한 설명", 0.5, created_tick=0)
        graph.set_activation(f"m{i}", 0.1 * i)
        graph.set_storage_score(f"m{i}", 0.1 * i)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(f"사실 번호 {i}", f"m{i}") for i in range(10)]

    corr = evaluator.activation_recall_correlation(test_queries, threshold=0.3, top_k=5)
    assert isinstance(corr, float)
    assert -1.0 <= corr <= 1.0


def test_score_summary_has_similarity_recall():
    graph = MemoryGraph(embedder=_fixed_embedder)
    for i in range(5):
        graph.add_memory(f"m{i}", "fact", f"사실 {i}", 0.5, created_tick=0)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(f"사실 {i}", f"m{i}") for i in range(5)]

    summary = evaluator.score_summary(test_queries)
    assert "similarity_recall_rate" in summary


def test_correlation_respects_threshold(monkeypatch):
    activations_by_id = {
        "m0": 0.1,
        "m1": 0.3,
        "m2": 0.6,
        "m3": 0.8,
        "m4": 0.9,
    }
    recalled_by_id = {
        "m0": False,
        "m1": True,
        "m2": False,
        "m3": True,
        "m4": True,
    }

    graph = MemoryGraph(embedder=_fixed_embedder)
    for memory_id, activation in activations_by_id.items():
        graph.add_memory(memory_id, "fact", f"메모리 {memory_id}", 0.5, created_tick=0)
        graph.set_activation(memory_id, activation)
        graph.set_storage_score(memory_id, activation)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(memory_id, memory_id) for memory_id in activations_by_id]

    def fake_get_query_results(query_text: str, *, top_k: int, current_tick: int | None):
        if recalled_by_id[query_text]:
            return [(query_text, 1.0)]
        return []

    monkeypatch.setattr(evaluator, "_get_query_results", fake_get_query_results)

    corr_all = evaluator.activation_recall_correlation(test_queries, threshold=0.0, top_k=5)
    corr_thresholded = evaluator.activation_recall_correlation(test_queries, threshold=0.5, top_k=5)

    expected_thresholded = float(
        np.corrcoef(
            np.array([0.6, 0.8, 0.9], dtype=np.float32),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
        )[0, 1]
    )

    assert corr_thresholded == pytest.approx(expected_thresholded)
    assert corr_thresholded != pytest.approx(corr_all)
