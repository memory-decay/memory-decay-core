"""Temporal correctness tests for Evaluator."""
import numpy as np
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
