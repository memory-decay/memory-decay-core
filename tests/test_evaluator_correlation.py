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
    graph = MemoryGraph(embedder=_fixed_embedder)
    for i in range(5):
        graph.add_memory(f"m{i}", "fact", f"사실 {i}", 0.5, created_tick=0)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    test_queries = [(f"사실 {i}", f"m{i}") for i in range(5)]

    summary = evaluator.score_summary(test_queries)
    assert "similarity_recall_rate" in summary
