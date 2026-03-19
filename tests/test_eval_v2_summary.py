"""Contract tests for Eval v2 summary fields."""

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
