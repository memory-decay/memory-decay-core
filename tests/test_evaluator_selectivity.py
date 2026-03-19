"""Tests for Eval v2 selectivity scoring."""

import numpy as np

from memory_decay.decay import DecayEngine
from memory_decay.evaluator import Evaluator
from memory_decay.graph import MemoryGraph


def _fixed_embedder(text: str) -> np.ndarray:
    rng = np.random.RandomState(hash(text) % 2**31)
    vec = rng.randn(16).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def _build_graph():
    graph = MemoryGraph(embedder=_fixed_embedder)
    graph.add_memory("target", "fact", "커피는 에티오피아가 원산지다", 0.9, created_tick=0)
    graph.add_memory("assoc", "fact", "에티오피아는 아프리카에 있다", 0.9, created_tick=0, associations=[("target", 0.8)])
    graph.add_memory("lure", "fact", "커피는 브라질이 원산지다", 0.9, created_tick=0)
    return graph


def test_selectivity_score_present_and_bounded():
    graph = _build_graph()
    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)
    summary = evaluator.score_summary([("커피 원산지", "target")], threshold=0.1, top_k=10)
    assert "selectivity_score" in summary
    assert 0.0 <= summary["selectivity_score"] <= 1.0


def test_selectivity_score_drops_when_strict_precision_worsens(monkeypatch):
    graph = _build_graph()
    engine = DecayEngine(graph)
    evaluator = Evaluator(graph, engine)

    monkeypatch.setattr(evaluator, "evaluate_precision", lambda *args, mode="associative", **kwargs: 0.8 if mode == "strict" else 0.9)
    strong = evaluator.score_summary([("커피 원산지", "target")], threshold=0.1, top_k=10)["selectivity_score"]

    monkeypatch.setattr(evaluator, "evaluate_precision", lambda *args, mode="associative", **kwargs: 0.2 if mode == "strict" else 0.9)
    weak = evaluator.score_summary([("커피 원산지", "target")], threshold=0.1, top_k=10)["selectivity_score"]

    assert strong > weak
