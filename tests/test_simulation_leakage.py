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


def test_scheduled_query_requires_rehearsal_targets():
    """scheduled_query without rehearsal_targets should raise ValueError."""
    graph = MemoryGraph(embedder=_fixed_embedder)
    graph.add_memory("m1", "fact", "test", 0.5, created_tick=0)
    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)

    import pytest
    with pytest.raises(ValueError, match="rehearsal_targets"):
        run_simulation(
            graph, engine, evaluator, [("q", "m1")],
            total_ticks=10, eval_interval=5,
            reactivation_policy="scheduled_query",
            seed=42,
        )
