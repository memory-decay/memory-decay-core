"""Contract tests for Eval v2 summary fields."""

import numpy as np
import pytest

from memory_decay import MemoryGraph, DecayEngine, Evaluator


def _mock_embedder(text: str):
    import numpy as np

    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def _make_summary():
    graph = MemoryGraph(embedder=_mock_embedder)
    graph.add_memory("f1", "fact", "서울은 대한민국의 수도이다", 0.8, 0)
    graph.add_memory("e1", "episode", "서울에 여행을 갔다", 0.6, 1, [("f1", 0.7)])
    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    queries = [("대한민국의 수도는?", "f1"), ("서울 여행은?", "e1")]
    return evaluator.score_summary(queries)


def test_eval_v2_summary_contains_new_primary_fields():
    summary = _make_summary()
    assert "retention_auc" in summary
    assert "selectivity_score" in summary
    assert "robustness_score" in summary
    assert "eval_v2_score" in summary
    assert "retention_curve" in summary
    assert "threshold_summary" in summary


def test_eval_v1_fields_still_exist_during_transition():
    summary = _make_summary()
    assert "overall_score" in summary
    assert "retrieval_score" in summary
    assert "plausibility_score" in summary


def test_score_summary_contains_activation_spread_metrics():
    graph = MemoryGraph(embedder=_mock_embedder)
    graph.add_memory("f1", "fact", "서울은 대한민국의 수도이다", 0.8, 0, [("ghost", 0.2)])
    graph.add_memory("f2", "fact", "부산은 항구 도시이다", 0.6, 0)
    graph.add_memory("e1", "episode", "서울에 여행을 갔다", 0.6, 0)
    graph.set_activation("f1", 0.1)
    graph.set_storage_score("f1", 0.1)
    graph.set_activation("f2", 0.4)
    graph.set_storage_score("f2", 0.4)
    graph.set_activation("e1", 0.9)
    graph.set_storage_score("e1", 0.9)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    summary = evaluator.score_summary(
        [("대한민국의 수도는?", "f1"), ("항구 도시는?", "f2"), ("서울 여행은?", "e1")]
    )

    activations = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    pairwise_diff = np.abs(activations[:, None] - activations[None, :])
    expected_gini = float(pairwise_diff.sum() / (2 * len(activations) * activations.sum()))

    assert summary["activation_std"] == pytest.approx(float(np.std(activations)))
    assert summary["activation_iqr"] == pytest.approx(
        float(np.percentile(activations, 75) - np.percentile(activations, 25))
    )
    assert summary["activation_gini"] == pytest.approx(expected_gini)


def test_score_summary_contains_storage_and_retrieval_spread_metrics():
    graph = MemoryGraph(embedder=_mock_embedder)
    graph.add_memory("f1", "fact", "서울은 대한민국의 수도이다", 0.8, 0, [("ghost", 0.2)])
    graph.add_memory("f2", "fact", "부산은 항구 도시이다", 0.6, 0)
    graph.add_memory("e1", "episode", "서울에 여행을 갔다", 0.6, 0)
    graph.set_storage_score("f1", 0.1)
    graph.set_storage_score("f2", 0.4)
    graph.set_storage_score("e1", 0.9)
    graph.set_activation("f1", 0.9)
    graph.set_activation("f2", 0.5)
    graph.set_activation("e1", 0.2)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    summary = evaluator.score_summary(
        [("대한민국의 수도는?", "f1"), ("항구 도시는?", "f2"), ("서울 여행은?", "e1")]
    )

    storage = np.array([0.1, 0.4, 0.9], dtype=np.float32)
    retrieval = np.array([0.9, 0.5, 0.2], dtype=np.float32)

    assert summary["storage_std"] == pytest.approx(float(np.std(storage)))
    assert summary["storage_iqr"] == pytest.approx(
        float(np.percentile(storage, 75) - np.percentile(storage, 25))
    )
    assert summary["retrieval_std"] == pytest.approx(float(np.std(retrieval)))
    assert summary["retrieval_iqr"] == pytest.approx(
        float(np.percentile(retrieval, 75) - np.percentile(retrieval, 25))
    )
    assert summary["activation_std"] == summary["storage_std"]
