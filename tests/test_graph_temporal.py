"""Temporal filtering tests for MemoryGraph."""
import numpy as np
from memory_decay.graph import MemoryGraph


def _fixed_embedder(text: str) -> np.ndarray:
    """Deterministic embedder: hash text into a 16-d vector."""
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


class TestTemporalFiltering:
    def _build_graph(self) -> MemoryGraph:
        graph = MemoryGraph(embedder=_fixed_embedder)
        graph.add_memory("past", "fact", "서울의 날씨", 0.5, created_tick=0)
        graph.add_memory("present", "fact", "서울의 기온", 0.5, created_tick=50)
        graph.add_memory("future", "fact", "서울의 미래 기후", 0.5, created_tick=100)
        return graph

    def test_no_tick_filter_returns_all(self):
        """Without current_tick, all memories are searchable (backward compat)."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" in ids
        assert "future" in ids

    def test_tick_filter_excludes_future(self):
        """With current_tick=50, future memory (tick=100) must be excluded."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10, current_tick=50)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" in ids
        assert "future" not in ids

    def test_tick_zero_only_shows_tick_zero(self):
        """At tick=0, only memories created at tick=0 are visible."""
        graph = self._build_graph()
        results = graph.query_by_similarity("서울 날씨", top_k=10, current_tick=0)
        ids = [r[0] for r in results]
        assert "past" in ids
        assert "present" not in ids
        assert "future" not in ids
