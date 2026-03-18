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
        self._history: list[dict] = []

    def evaluate_recall(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
        top_k: int = 5,
    ) -> float:
        """Fraction of memories successfully recalled.

        A memory is "recalled" if:
        1. Its activation_score > threshold (it hasn't decayed beyond retrievability)
        2. It appears in the top_k similarity search results

        Both conditions must be met — this models the dual requirement that
        a memory must be both stored (activation) and retrievable (similarity).
        """
        if not test_queries:
            return 0.0

        current_tick = self._engine.current_tick
        recalled = 0
        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node:
                continue

            # Skip memories that don't exist yet at current_tick
            if node.get("created_tick", 0) > current_tick:
                continue

            # Condition 1: activation above threshold
            if node["activation_score"] < threshold:
                continue

            # Condition 2: appears in similarity results
            results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
            result_ids = [rid for rid, _ in results]

            if expected_id in result_ids:
                recalled += 1

        return recalled / len(test_queries)

    def evaluate_precision(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
        top_k: int = 5,
        mode: str = "associative",
    ) -> float:
        """Precision of recall results.

        Args:
            mode: "strict" = only exact expected_id is relevant.
                  "associative" = associated nodes also count (original behavior).
        """
        if not test_queries:
            return 0.0

        current_tick = self._engine.current_tick
        total_relevant = 0
        total_retrieved = 0

        for query, expected_id in test_queries:
            results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)

            if mode == "strict":
                relevant_ids = {expected_id}
            else:
                relevant_ids = set()
                node = self._graph.get_node(expected_id)
                if node:
                    for assoc_id, _ in self._graph.get_associated(expected_id):
                        relevant_ids.add(assoc_id)
                relevant_ids.add(expected_id)

            for rid, _ in results:
                r_node = self._graph.get_node(rid)
                if not r_node or r_node.get("type") in ("unknown", None):
                    continue
                if r_node["activation_score"] < threshold:
                    continue
                total_retrieved += 1
                if rid in relevant_ids:
                    total_relevant += 1

        if total_retrieved == 0:
            return 0.0
        return total_relevant / total_retrieved

    def activation_recall_correlation(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
        top_k: int = 5,
    ) -> float:
        """Pearson correlation between activation score and recall success.

        Returns correlation in [-1, 1]. Higher = activation predicts recall well.
        """
        if len(test_queries) < 3:
            return 0.0

        current_tick = self._engine.current_tick
        activations = []
        recall_success = []

        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node:
                continue

            # Skip memories that don't exist yet at current_tick
            if node.get("created_tick", 0) > current_tick:
                continue

            act = node["activation_score"]
            results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
            result_ids = [rid for rid, _ in results]
            recalled = 1.0 if expected_id in result_ids else 0.0

            activations.append(act)
            recall_success.append(recalled)

        if len(activations) < 3:
            return 0.0

        a = np.array(activations)
        r = np.array(recall_success)
        if np.std(a) == 0 or np.std(r) == 0:
            return 0.0

        return float(np.corrcoef(a, r)[0, 1])

    def evaluate_similarity_recall(self, test_queries, top_k=5):
        """Pure similarity-based recall — no activation threshold."""
        if not test_queries:
            return 0.0
        current_tick = self._engine.current_tick
        recalled = 0
        total = 0
        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node or node.get("created_tick", 0) > current_tick:
                continue
            total += 1
            results = self._graph.query_by_similarity(query, top_k=top_k, current_tick=current_tick)
            if expected_id in [rid for rid, _ in results]:
                recalled += 1
        return recalled / max(total, 1)

    def fact_episode_delta(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
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
        threshold: float = 0.3,
        top_k: int = 5,
        record: bool = True,
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
        if record:
            self._history.append(snap)
        return snap

    def threshold_sweep(
        self,
        test_queries: list[tuple[str, str]],
        thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
        top_k: int = 5,
    ) -> dict:
        """Evaluate retrieval metrics over a fixed threshold grid."""
        threshold_metrics: dict[float, dict[str, float]] = {}
        recalls = []
        precisions = []

        for threshold in thresholds:
            recall = self.evaluate_recall(test_queries, threshold=threshold, top_k=top_k)
            precision = self.evaluate_precision(
                test_queries, threshold=threshold, top_k=top_k
            )
            threshold_metrics[threshold] = {
                "recall_rate": recall,
                "precision_rate": precision,
            }
            recalls.append(recall)
            precisions.append(precision)

        return {
            "threshold_metrics": threshold_metrics,
            "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
            "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
        }

    def _smoothness_score(self) -> float:
        if len(self._history) < 2:
            return 0.5
        return max(1.0 - self.forgetting_curve_smoothness() * 10, 0.0)

    def score_summary(
        self,
        test_queries: list[tuple[str, str]],
        thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5),
        threshold: float = 0.3,
        top_k: int = 5,
    ) -> dict:
        """Balanced summary split into retrieval and plausibility sub-scores."""
        snap = self.snapshot(test_queries, threshold=threshold, top_k=top_k, record=False)
        sweep = self.threshold_sweep(test_queries, thresholds=thresholds, top_k=top_k)
        correlations = [
            self.activation_recall_correlation(test_queries, threshold=t, top_k=top_k)
            for t in thresholds
        ]
        corr_mean = float(np.mean(correlations)) if correlations else 0.0
        corr_score = max(min(corr_mean, 1.0), 0.0)
        smoothness_score = self._smoothness_score()

        retrieval_score = (
            0.7 * sweep["recall_mean"] + 0.3 * sweep["precision_mean"]
        )
        plausibility_score = 0.6 * corr_score + 0.4 * smoothness_score
        overall_score = 0.7 * retrieval_score + 0.3 * plausibility_score

        return {
            **snap,
            **sweep,
            "corr_mean": corr_mean,
            "corr_score": corr_score,
            "smoothness_score": smoothness_score,
            "retrieval_score": retrieval_score,
            "plausibility_score": plausibility_score,
            "overall_score": overall_score,
            "composite_score": overall_score,
        }

    def composite_score(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
        top_k: int = 5,
    ) -> float:
        """Backward-compatible alias for the overall score summary."""
        return self.score_summary(test_queries, threshold=threshold, top_k=top_k)[
            "overall_score"
        ]

    @property
    def history(self) -> list[dict]:
        return list(self._history)
