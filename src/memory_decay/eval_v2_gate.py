"""Acceptance rules for Eval v2 candidates."""

from __future__ import annotations


def decide_eval_v2(candidate_stats: dict, champion_stats: dict, baseline_stats: dict) -> dict:
    """Return a simple accept/reject decision for Eval v2 candidates."""
    candidate_mean = candidate_stats.get("mean", {})
    champion_mean = champion_stats.get("mean", {})
    baseline_mean = baseline_stats.get("mean", {})

    mean_improvement = candidate_mean.get("eval_v2_score", 0.0) - champion_mean.get("eval_v2_score", 0.0)
    if mean_improvement < 0.01:
        return {"accept": False, "reason": "mean improvement below threshold"}

    if candidate_stats.get("lower_bound_delta", 0.0) <= 0.0:
        return {"accept": False, "reason": "fold lower bound did not clear zero"}

    improving_folds = sum(1 for delta in candidate_stats.get("fold_deltas", []) if delta > 0.0)
    if improving_folds < 4:
        return {"accept": False, "reason": "fold consistency too weak"}

    worst_fold = candidate_stats.get("worst_fold", {})
    if worst_fold.get("retention_auc", 0.0) < baseline_mean.get("retention_auc", 0.0) - 0.02:
        return {"accept": False, "reason": "worst fold retention below baseline guardrail"}

    if worst_fold.get("selectivity_score", 0.0) < baseline_mean.get("selectivity_score", 0.0) - 0.02:
        return {"accept": False, "reason": "worst fold selectivity below baseline guardrail"}

    threshold_summary = candidate_stats.get("threshold_summary", {})
    if threshold_summary.get("threshold_auc", 0.0) <= 0.0 and threshold_summary.get("slope", 0.0) == 0.0:
        return {"accept": False, "reason": "threshold diagnostics are degenerate"}

    return {"accept": True, "reason": "all eval v2 conditions satisfied"}
