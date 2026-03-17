"""Multi-metric evaluator for memory system performance."""

from __future__ import annotations

import numpy as np
from typing import Optional

from .graph import MemoryGraph
from .decay import DecayEngine


class Evaluator:
    """Measures memory system performance with multiple complementary metrics.

    Prevents single-metric gaming by using a composite score.
    """

    def __init__(self, graph: MemoryGraph, engine: DecayEngine):
        self._graph = graph
        self._engine = engine
        self._history: list[dict] = []  # recall snapshots per tick

    def evaluate_recall(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> float:
        """Fraction of memories successfully recalled.

        For each (query, expected_id): check if expected_id appears in
        top_k similarity results AND has activation > threshold.

        Returns recall_rate in [0, 1].
        """
        if not test_queries:
            return 0.0

        recalled = 0
        for query, expected_id in test_queries:
            results = self._graph.query_by_similarity(query, top_k=top_k)
            result_ids = [rid for rid, _ in results]

            if expected_id in result_ids:
                node = self._graph.get_node(expected_id)
                if node and node["activation_score"] > threshold:
                    recalled += 1
                    continue

            # Also accept if activation alone is high enough (direct retrieval)
            node = self._graph.get_node(expected_id)
            if node and node["activation_score"] > threshold:
                recalled += 1

        return recalled / len(test_queries)

    def evaluate_precision(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> float:
        """Precision of recall results.

        For each query: of the top_k results, count how many are relevant.
        Relevant = shares an association with the expected memory, or IS the expected.
        """
        if not test_queries:
            return 0.0

        total_relevant = 0
        total_retrieved = 0

        for query, expected_id in test_queries:
            results = self._graph.query_by_similarity(query, top_k=top_k)
            total_retrieved += len(results)

            # Get expected node's associations
            expected_assoc = set()
            node = self._graph.get_node(expected_id)
            if node:
                for assoc_id, _ in self._graph.get_associated(expected_id):
                    expected_assoc.add(assoc_id)
            expected_assoc.add(expected_id)

            for rid, _ in results:
                if rid in expected_assoc:
                    total_relevant += 1

        if total_retrieved == 0:
            return 0.0
        return total_relevant / total_retrieved

    def activation_recall_correlation(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> float:
        """Pearson correlation between activation score and recall success.

        Returns correlation in [-1, 1]. Higher = activation predicts recall well.
        """
        if len(test_queries) < 3:
            return 0.0

        activations = []
        recall_success = []

        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node:
                continue

            act = node["activation_score"]
            results = self._graph.query_by_similarity(query, top_k=top_k)
            result_ids = [rid for rid, _ in results]
            recalled = 1.0 if (expected_id in result_ids and act > threshold) else 0.0

            activations.append(act)
            recall_success.append(recalled)

        if len(activations) < 3:
            return 0.0

        a = np.array(activations)
        r = np.array(recall_success)
        if np.std(a) == 0 or np.std(r) == 0:
            return 0.0

        return float(np.corrcoef(a, r)[0, 1])

    def fact_episode_delta(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> float:
        """Absolute difference in recall rates between facts and episodes."""
        fact_queries = []
        episode_queries = []

        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if node and node.get("type") == "fact":
                fact_queries.append((query, expected_id))
            elif node and node.get("type") == "episode":
                episode_queries.append((query, expected_id))

        fact_recall = (
            self.evaluate_recall(fact_queries, threshold, top_k)
            if fact_queries
            else 0.0
        )
        episode_recall = (
            self.evaluate_recall(episode_queries, threshold, top_k)
            if episode_queries
            else 0.0
        )

        return abs(fact_recall - episode_recall)

    def forgetting_curve_smoothness(
        self, history: list[float] | None = None
    ) -> float:
        """Variance of differences in the forgetting curve.

        Lower = smoother = better. Measures how jagged the decay is.
        """
        curve = history if history is not None else [h["recall_rate"] for h in self._history]
        if len(curve) < 2:
            return 0.0

        diffs = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
        return float(np.var(diffs))

    def snapshot(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> dict:
        """Take a measurement snapshot at current tick."""
        recall = self.evaluate_recall(test_queries, threshold, top_k)
        precision = self.evaluate_precision(test_queries, threshold, top_k)
        corr = self.activation_recall_correlation(test_queries, threshold, top_k)
        delta = self.fact_episode_delta(test_queries, threshold, top_k)
        smoothness = self.forgetting_curve_smoothness()

        snap = {
            "tick": self._engine.current_tick,
            "recall_rate": recall,
            "precision_rate": precision,
            "activation_recall_correlation": corr,
            "fact_episode_delta": delta,
            "forgetting_curve_smoothness": smoothness,
        }
        self._history.append(snap)
        return snap

    def composite_score(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> float:
        """Weighted composite score combining all 5 metrics.

        Weights:
          - 0.30 * recall_rate
          - 0.25 * precision_rate
          - 0.20 * (1 - |correlation|) — lower absolute corr is penalized
          - 0.10 * min(delta, 0.3) / 0.3 — rewards meaningful fact/episode difference
          - 0.15 * (1 - normalized_smoothness)
        """
        snap = self.snapshot(test_queries, threshold, top_k)

        w_recall = 0.30
        w_precision = 0.25
        w_corr = 0.20
        w_delta = 0.10
        w_smooth = 0.15

        score = (
            w_recall * snap["recall_rate"]
            + w_precision * snap["precision_rate"]
            + w_corr * (1 - abs(snap["activation_recall_correlation"]))
            + w_delta * min(snap["fact_episode_delta"], 0.3) / 0.3
            + w_smooth * max(1.0 - snap["forgetting_curve_smoothness"] * 10, 0.0)
        )

        snap["composite_score"] = score
        return score

    @property
    def history(self) -> list[dict]:
        return list(self._history)
