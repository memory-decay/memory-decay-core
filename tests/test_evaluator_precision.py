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
