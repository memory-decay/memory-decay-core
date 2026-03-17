"""Evaluator: multi-metric recall evaluation.

Measures memory system performance using 5 metrics:
1. recall_rate: fraction of memories recallable at time t (activation > threshold AND found by similarity search)
2. precision_rate: of recalled results, fraction that are actually relevant
3. activation_recall_correlation: Pearson correlation between activation score and recall success
4. fact_episode_delta: absolute difference in recall rates between facts and episodes
5. forgetting_curve_smoothness: variance of the forgetting curve derivative (lower = smoother)

Computes a weighted composite score to prevent single-metric gaming.
Runs periodic recall tests at configurable tick intervals.

Anti-gaming measures:
- Memorization detection (flag if recall_rate > 0.95 at all ticks)
- Smoothness check for jagged forgetting curves
- Held-out test set for recall queries
"""
