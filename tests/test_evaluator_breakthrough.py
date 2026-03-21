"""Breakthrough evaluator regression tests."""

import numpy as np

from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.graph import MemoryGraph


def _vector_embedder_factory():
    vectors = {
        "memory high 0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "memory high 1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "memory high 2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "memory low 0": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "memory low 1": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "memory low 2": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "query high 0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "query high 1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "query high 2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "query low 0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "query low 1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "query low 2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    def _embedder(text: str) -> np.ndarray:
        return vectors[text]

    return _embedder


def _build_breakthrough_evaluator() -> tuple[Evaluator, list[tuple[str, str]]]:
    graph = MemoryGraph(embedder=_vector_embedder_factory())
    memories = [
        ("high0", "memory high 0", 0.95, "query high 0"),
        ("high1", "memory high 1", 0.85, "query high 1"),
        ("high2", "memory high 2", 0.75, "query high 2"),
        ("low0", "memory low 0", 0.25, "query low 0"),
        ("low1", "memory low 1", 0.15, "query low 1"),
        ("low2", "memory low 2", 0.05, "query low 2"),
    ]

    for memory_id, content, activation, _ in memories:
        graph.add_memory(memory_id, "fact", content, 0.5, created_tick=0)
        graph.set_activation(memory_id, activation)
        graph.set_storage_score(memory_id, activation)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    queries = [(query, memory_id) for memory_id, _, _, query in memories]
    return evaluator, queries


def test_activation_recall_correlation_changes_when_threshold_excludes_low_activation_failures():
    evaluator, queries = _build_breakthrough_evaluator()

    corr_all = evaluator.activation_recall_correlation(queries, threshold=0.0, top_k=1)
    corr_high_only = evaluator.activation_recall_correlation(queries, threshold=0.5, top_k=1)

    assert corr_all > 0.8
    assert corr_high_only == 0.0


def test_score_summary_reports_activation_spread_metrics():
    graph = MemoryGraph(embedder=_vector_embedder_factory())
    activations = [0.1, 0.2, 0.5, 0.9]

    for idx, activation in enumerate(activations):
        graph.add_memory(f"m{idx}", "fact", f"memory high {idx % 3}", 0.5, created_tick=0)
        graph.set_activation(f"m{idx}", activation)
        graph.set_storage_score(f"m{idx}", activation)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    summary = evaluator.score_summary([("query high 0", "m0")], threshold=0.1, top_k=1)

    a = np.array(activations, dtype=np.float64)
    q1, q3 = np.percentile(a, [25, 75])
    sorted_a = np.sort(a)
    index = np.arange(1, len(sorted_a) + 1, dtype=np.float64)
    expected_gini = (
        (2.0 * np.sum(index * sorted_a) - (len(sorted_a) + 1) * np.sum(sorted_a))
        / (len(sorted_a) * np.sum(sorted_a))
    )

    assert summary["activation_std"] == float(np.std(a))
    assert summary["activation_iqr"] == float(q3 - q1)
    assert summary["activation_gini"] == float(expected_gini)


def test_recall_filters_low_activation_candidates_before_applying_top_k():
    vectors = {
        "target memory": np.array([0.8, 0.6], dtype=np.float32),
        "low distractor": np.array([1.0, 0.0], dtype=np.float32),
        "other memory": np.array([-1.0, 0.0], dtype=np.float32),
        "query": np.array([1.0, 0.0], dtype=np.float32),
    }

    graph = MemoryGraph(embedder=lambda text: vectors[text])
    graph.add_memory("target", "fact", "target memory", 0.5, created_tick=0)
    graph.add_memory("low", "fact", "low distractor", 0.5, created_tick=0)
    graph.add_memory("other", "fact", "other memory", 0.5, created_tick=0)
    graph.set_activation("target", 0.9)
    graph.set_storage_score("target", 0.9)
    graph.set_activation("low", 0.1)
    graph.set_storage_score("low", 0.1)
    graph.set_activation("other", 0.9)
    graph.set_storage_score("other", 0.9)

    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine, activation_weight=0.0)

    raw_top1 = evaluator._get_query_results("query", top_k=1, current_tick=engine.current_tick)
    assert raw_top1[0][0] == "low"

    recall = evaluator.evaluate_recall([("query", "target")], threshold=0.5, top_k=1)
    assert recall == 1.0


def test_recall_uses_storage_threshold_but_ranking_uses_retrieval():
    vectors = {
        "target memory": np.array([1.0, 0.0], dtype=np.float32),
        "other memory": np.array([0.0, 1.0], dtype=np.float32),
        "query": np.array([1.0, 0.0], dtype=np.float32),
    }

    graph = MemoryGraph(embedder=lambda text: vectors[text])
    graph.add_memory("target", "fact", "target memory", 0.5, created_tick=0)
    graph.add_memory("other", "fact", "other memory", 0.5, created_tick=0)
    graph.set_storage_score("target", 0.2)
    graph.set_activation("target", 0.95)
    graph.set_storage_score("other", 0.95)
    graph.set_activation("other", 0.1)

    evaluator = Evaluator(graph, DecayEngine(graph), activation_weight=1.0)
    recall = evaluator.evaluate_recall([("query", "target")], threshold=0.5, top_k=1)

    assert recall == 0.0
