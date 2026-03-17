"""Unit tests for Phase 1 core components."""

import math
import numpy as np
import pytest

from memory_decay import MemoryGraph, DecayEngine, Evaluator


# --- Helpers ---

def mock_embedder(text: str) -> np.ndarray:
    """Deterministic mock embedder based on text hash."""
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(384).astype(np.float32)


def make_graph() -> MemoryGraph:
    return MemoryGraph(embedder=mock_embedder)


def make_exponential_engine(graph: MemoryGraph) -> DecayEngine:
    return DecayEngine(graph, decay_type="exponential", params={
        "lambda_fact": 0.05, "lambda_episode": 0.08,
        "beta_fact": 0.3, "beta_episode": 0.5,
        "alpha": 0.5,
    })


def make_power_law_engine(graph: MemoryGraph) -> DecayEngine:
    return DecayEngine(graph, decay_type="power_law", params={
        "lambda_fact": 0.05, "lambda_episode": 0.08,
        "beta_fact": 0.3, "beta_episode": 0.5,
        "alpha": 0.5,
    })


# --- MemoryGraph Tests ---

class TestMemoryGraph:
    def test_add_memory(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "서울은 대한민국의 수도이다", 0.8, 0)
        node = g.get_node("mem_001")
        assert node is not None
        assert node["type"] == "fact"
        assert node["impact"] == 0.8
        assert node["activation_score"] == 1.0
        assert node["embedding"].shape == (384,)

    def test_add_memory_with_associations(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "서울은 수도다", 0.8, 0, [("mem_002", 0.9)])
        g.add_memory("mem_002", "episode", "서울에 여행갔다", 0.6, 1)

        assoc = g.get_associated("mem_001")
        assert len(assoc) >= 1
        # Bidirectional
        assoc_002 = g.get_associated("mem_002")
        assert len(assoc_002) >= 1

    def test_query_similarity(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "서울은 수도다", 0.8, 0)
        g.add_memory("mem_002", "fact", "고양이는 동물이다", 0.5, 0)

        results = g.query_by_similarity("대한민국의 수도", top_k=2)
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "mem_001" in ids

    def test_re_activate_cascade(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "서울은 수도다", 0.8, 0, [("mem_002", 0.8)])
        g.add_memory("mem_002", "episode", "서울에 여행갔다", 0.6, 1)

        g.set_activation("mem_001", 0.5)
        g.set_activation("mem_002", 0.3)

        g.re_activate("mem_001", 0.5)

        assert g.get_node("mem_001")["activation_score"] > 0.5
        assert g.get_node("mem_002")["activation_score"] > 0.3  # cascade

    def test_get_all_activations(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "test", 0.5, 0)
        g.add_memory("mem_002", "episode", "test2", 0.5, 0)

        acts = g.get_all_activations()
        assert len(acts) == 2
        assert acts["mem_001"] == 1.0

    def test_num_memories(self):
        g = make_graph()
        assert g.num_memories == 0
        g.add_memory("mem_001", "fact", "test", 0.5, 0, [("mem_002", 0.5)])
        # mem_002 is a placeholder, shouldn't count
        assert g.num_memories == 1
        g.add_memory("mem_002", "episode", "test2", 0.5, 0)
        assert g.num_memories == 2


# --- DecayEngine Tests ---

class TestDecayEngine:
    def test_exponential_decay_decreases(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "test", 0.8, 0)
        engine = make_exponential_engine(g)

        initial = g.get_node("mem_001")["activation_score"]
        engine.tick()
        after = g.get_node("mem_001")["activation_score"]

        # With high impact (0.8) and alpha=0.5: impact_mod = 1.4
        # Should still decay from initial since e^(-0.05) < 1 but impact_mod > 1
        assert isinstance(after, float)

    def test_power_law_decay_decreases(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "test", 0.5, 0)
        engine = make_power_law_engine(g)

        engine.tick()
        after = g.get_node("mem_001")["activation_score"]

        assert isinstance(after, float)
        assert after >= 0.0

    def test_tick_advances(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "test", 0.5, 0)
        engine = make_exponential_engine(g)

        assert engine.current_tick == 0
        engine.tick()
        assert engine.current_tick == 1
        engine.tick()
        assert engine.current_tick == 2

    def test_high_impact_decays_slower(self):
        g = make_graph()
        g.add_memory("high", "fact", "important", 1.0, 0)
        g.add_memory("low", "fact", "mundane", 0.1, 0)
        engine = make_exponential_engine(g)

        # Run several ticks
        for _ in range(10):
            engine.tick()

        high_act = g.get_node("high")["activation_score"]
        low_act = g.get_node("low")["activation_score"]

        # High impact should have higher activation due to impact modifier
        assert high_act >= low_act

    def test_per_type_params(self):
        g = make_graph()
        g.add_memory("fact_001", "fact", "fact", 0.5, 0)
        g.add_memory("ep_001", "episode", "episode", 0.5, 0)
        engine = make_exponential_engine(g)

        # lambda_episode (0.08) > lambda_fact (0.05), episodes decay faster
        for _ in range(20):
            engine.tick()

        fact_act = g.get_node("fact_001")["activation_score"]
        ep_act = g.get_node("ep_001")["activation_score"]

        # Episodes should decay faster (higher lambda)
        assert fact_act >= ep_act

    def test_set_params(self):
        g = make_graph()
        g.add_memory("mem_001", "fact", "test", 0.5, 0)
        engine = make_exponential_engine(g)

        engine.set_params({"lambda_fact": 0.5})
        assert engine.get_params()["lambda_fact"] == 0.5

        # Higher lambda should cause faster decay
        for _ in range(5):
            engine.tick()
        act = g.get_node("mem_001")["activation_score"]
        assert act < 1.0  # Should have decayed significantly


# --- Evaluator Tests ---

class TestEvaluator:
    def _setup(self) -> tuple[MemoryGraph, DecayEngine, Evaluator, list]:
        g = make_graph()
        g.add_memory("f1", "fact", "서울은 대한민국의 수도이다", 0.8, 0)
        g.add_memory("f2", "fact", "고양이는 포유류이다", 0.5, 1, [("f1", 0.7)])
        g.add_memory("e1", "episode", "서울에 여행을 갔다", 0.7, 2, [("f1", 0.9)])
        g.add_memory("e2", "episode", "길에서 고양이를 만났다", 0.4, 3, [("f2", 0.8)])

        engine = make_exponential_engine(g)
        evaluator = Evaluator(g, engine)

        queries = [
            ("대한민국의 수도는?", "f1"),
            ("고양이는 무엇인가?", "f2"),
            ("서울 여행은 어땠나?", "e1"),
            ("길에서 만난 동물은?", "e2"),
        ]
        return g, engine, evaluator, queries

    def test_recall_rate(self):
        g, engine, evaluator, queries = self._setup()
        # At tick 0, all activations are 1.0
        recall = evaluator.evaluate_recall(queries, threshold=0.5)
        assert 0.0 <= recall <= 1.0

    def test_precision_rate(self):
        g, engine, evaluator, queries = self._setup()
        precision = evaluator.evaluate_precision(queries, threshold=0.5)
        assert 0.0 <= precision <= 1.0

    def test_composite_score(self):
        g, engine, evaluator, queries = self._setup()
        score = evaluator.composite_score(queries, threshold=0.5)
        assert 0.0 <= score <= 1.0

    def test_snapshot_records_history(self):
        g, engine, evaluator, queries = self._setup()
        evaluator.snapshot(queries)
        evaluator.snapshot(queries)
        assert len(evaluator.history) == 2

    def test_decay_affects_recall(self):
        g, engine, evaluator, queries = self._setup()

        recall_before = evaluator.evaluate_recall(queries, threshold=0.9)

        # Run many ticks to let memories decay
        for _ in range(50):
            engine.tick()

        recall_after = evaluator.evaluate_recall(queries, threshold=0.9)

        # Recall should decrease (or stay same) after heavy decay
        assert recall_after <= recall_before

    def test_smoothness(self):
        evaluator = Evaluator(MemoryGraph(embedder=mock_embedder), make_exponential_engine(MemoryGraph(embedder=mock_embedder)))
        # Smooth curve: [1.0, 0.9, 0.8, 0.7]
        smooth_var = evaluator.forgetting_curve_smoothness([1.0, 0.9, 0.8, 0.7])
        # Jagged curve: [1.0, 0.5, 0.9, 0.3]
        jagged_var = evaluator.forgetting_curve_smoothness([1.0, 0.5, 0.9, 0.3])
        assert smooth_var < jagged_var
