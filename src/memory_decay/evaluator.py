"""Multi-metric evaluator for memory system performance."""

from __future__ import annotations

import numpy as np
from collections import OrderedDict

from .graph import MemoryGraph
from .decay import DecayEngine


class Evaluator:
    """Measures memory system performance with multiple complementary metrics.

    Prevents single-metric gaming by using a composite score.
    """

    def __init__(self, graph: MemoryGraph, engine: DecayEngine, activation_weight: float = 0.5, assoc_boost: float = 0.0):
        self._graph = graph
        self._engine = engine
        self._history: list[dict] = []
        self._activation_weight = activation_weight
        self._assoc_boost = assoc_boost
        self._query_result_cache: dict[tuple, list[tuple[str, float]]] = {}

    def _get_query_results(
        self,
        query_text: str,
        *,
        top_k: int,
        current_tick: int | None,
    ) -> list[tuple[str, float]]:
        """Return similarity results, reusing work within the same tick."""
        cache_key = (
            current_tick,
            query_text,
            top_k,
            self._activation_weight,
            self._assoc_boost,
        )
        cached = self._query_result_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        results = self._graph.query_by_similarity(
            query_text,
            top_k=top_k,
            current_tick=current_tick,
            activation_weight=self._activation_weight,
            assoc_boost=self._assoc_boost,
        )
        self._query_result_cache[cache_key] = list(results)
        return list(results)

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
        observable = 0
        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node:
                continue

            # Skip memories that don't exist yet at current_tick
            if node.get("created_tick", 0) > current_tick:
                continue

            observable += 1

            # Condition 1: activation above threshold
            if node["activation_score"] < threshold:
                continue

            # Condition 2: appears in similarity results
            results = self._get_query_results(query, top_k=top_k, current_tick=current_tick)
            result_ids = [rid for rid, _ in results]

            if expected_id in result_ids:
                recalled += 1

        return recalled / max(observable, 1)

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
            node = self._graph.get_node(expected_id)
            if not node or node.get("created_tick", 0) > current_tick:
                continue

            results = self._get_query_results(query, top_k=top_k, current_tick=current_tick)

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
            results = self._get_query_results(query, top_k=top_k, current_tick=current_tick)
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

    def evaluate_mrr(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float = 0.3,
        top_k: int = 5,
    ) -> float:
        """Mean Reciprocal Rank — measures ranking quality independently of recall.

        For each query, finds the rank of the expected_id among retrieved results
        (filtered by activation threshold). RR = 1/rank if found, 0 otherwise.
        MRR = mean of all reciprocal ranks.

        Unlike precision_strict (which equals recall/top_k when all activations
        exceed the threshold), MRR captures WHERE in the ranking the correct
        answer appears, providing independent information from recall.
        """
        if not test_queries:
            return 0.0

        current_tick = self._engine.current_tick
        reciprocal_ranks = []

        for query, expected_id in test_queries:
            node = self._graph.get_node(expected_id)
            if not node or node.get("created_tick", 0) > current_tick:
                continue

            results = self._get_query_results(query, top_k=top_k, current_tick=current_tick)

            # Filter by activation threshold and find rank of expected_id
            rank = 0
            found = False
            for rid, _ in results:
                r_node = self._graph.get_node(rid)
                if not r_node or r_node.get("type") in ("unknown", None):
                    continue
                if r_node["activation_score"] < threshold:
                    continue
                rank += 1
                if rid == expected_id:
                    found = True
                    break

            reciprocal_ranks.append(1.0 / rank if found else 0.0)

        if not reciprocal_ranks:
            return 0.0
        return float(np.mean(reciprocal_ranks))

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
            results = self._get_query_results(query, top_k=top_k, current_tick=current_tick)
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
        thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
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

        threshold_discrimination = (
            (float(np.max(recalls)) - float(np.min(recalls))) if recalls else 0.0
        )
        threshold_auc = float(np.mean(recalls)) if recalls else 0.0
        slope = (
            float(recalls[-1] - recalls[0])
            if len(recalls) >= 2
            else 0.0
        )

        return {
            "threshold_metrics": threshold_metrics,
            "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
            "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "threshold_discrimination": threshold_discrimination,
            "threshold_summary": {
                "threshold_auc": threshold_auc,
                "slope": slope,
            },
        }

    def _retention_curve(
        self,
        test_queries: list[tuple[str, str]],
        threshold: float,
        top_k: int,
    ) -> OrderedDict[str, float]:
        target_ticks = (40, 80, 120, 160, 200)
        history_by_tick = {snap["tick"]: snap for snap in self._history}
        curve: OrderedDict[str, float] = OrderedDict()

        for tick in target_ticks:
            if tick in history_by_tick:
                curve[str(tick)] = float(history_by_tick[tick].get("recall_rate", 0.0))
            elif self._engine.current_tick == tick:
                curve[str(tick)] = self.evaluate_recall(test_queries, threshold=threshold, top_k=top_k)
            else:
                curve[str(tick)] = 0.0
        return curve

    def _retention_auc(self, retention_curve: OrderedDict[str, float]) -> float:
        values = list(retention_curve.values())
        if not values:
            return 0.0
        return float(np.mean(values))

    def _smoothness_score(self) -> float:
        if len(self._history) < 2:
            return 0.5
        return max(1.0 - self.forgetting_curve_smoothness() * 10, 0.0)

    def score_summary(
        self,
        test_queries: list[tuple[str, str]],
        thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        threshold: float = 0.3,
        top_k: int = 5,
        record_snapshot: bool = False,
    ) -> dict:
        """Balanced summary split into retrieval and plausibility sub-scores."""
        snap = self.snapshot(
            test_queries,
            threshold=threshold,
            top_k=top_k,
            record=record_snapshot,
        )
        if record_snapshot:
            snap = {
                **snap,
                "forgetting_curve_smoothness": self.forgetting_curve_smoothness(),
            }
        sweep = self.threshold_sweep(test_queries, thresholds=thresholds, top_k=top_k)
        correlations = [
            self.activation_recall_correlation(test_queries, threshold=t, top_k=top_k)
            for t in thresholds
        ]
        corr_mean = float(np.mean(correlations)) if correlations else 0.0
        # Allow negative correlation to penalize anti-causal models
        corr_score = max(min(corr_mean, 1.0), -1.0)
        smoothness_score = self._smoothness_score()

        # Strict precision sweep
        strict_precisions = []
        for t in thresholds:
            sp = self.evaluate_precision(test_queries, threshold=t, top_k=top_k, mode="strict")
            strict_precisions.append(sp)

        # MRR sweep — captures ranking quality independently of recall
        mrr_scores = []
        for t in thresholds:
            mrr = self.evaluate_mrr(test_queries, threshold=t, top_k=top_k)
            mrr_scores.append(mrr)
        mrr_mean = float(np.mean(mrr_scores)) if mrr_scores else 0.0

        # Similarity recall (threshold-independent)
        sim_recall = self.evaluate_similarity_recall(test_queries, top_k=top_k)
        retention_curve = self._retention_curve(test_queries, threshold=threshold, top_k=top_k)
        retention_auc = self._retention_auc(retention_curve)

        precision_strict_mean = float(np.mean(strict_precisions)) if strict_precisions else 0.0

        # Calculate precision lift
        null_precision = sweep["recall_mean"] / max(top_k, 1)
        precision_lift = max(0.0, precision_strict_mean - null_precision)
        selectivity_score = max(precision_strict_mean - (sweep["precision_mean"] - precision_strict_mean), 0.0)
        robustness_score = 0.0
        eval_v2_score = (
            0.45 * retention_auc
            + 0.35 * selectivity_score
            + 0.20 * robustness_score
        )

        # retrieval_score: rebalanced — lower recall weight, higher precision_lift
        # to incentivize selective forgetting over "keep everything alive"
        retrieval_score = (
            0.40 * sweep["recall_mean"]
            + 0.30 * mrr_mean
            + 0.30 * precision_lift
        )

        # Plausibility: correlation now allows negative values (penalty),
        # smoothness weight reduced as it has low discriminating power
        plausibility_score = 0.50 * corr_score + 0.50 * smoothness_score

        # Multiplicative gate: retrieval must support plausibility
        overall_score = retrieval_score * (0.85 + 0.15 * plausibility_score)

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
            "mrr_mean": mrr_mean,
            "precision_strict": precision_strict_mean,
            "precision_lift": precision_lift,
            "precision_associative": sweep["precision_mean"],
            "similarity_recall_rate": sim_recall,
            "retention_curve": retention_curve,
            "retention_auc": retention_auc,
            "selectivity_score": selectivity_score,
            "robustness_score": robustness_score,
            "eval_v2_score": eval_v2_score,
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
