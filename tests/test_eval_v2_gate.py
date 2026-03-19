"""Acceptance rule tests for Eval v2."""

from memory_decay.eval_v2_gate import decide_eval_v2


def test_candidate_accepted_when_all_conditions_hold():
    candidate_stats = {
        "mean": {"eval_v2_score": 0.35, "retention_auc": 0.40, "selectivity_score": 0.30},
        "fold_deltas": [0.02, 0.01, 0.015, 0.02, 0.01],
        "lower_bound_delta": 0.005,
        "worst_fold": {"retention_auc": 0.34, "selectivity_score": 0.25},
        "threshold_summary": {"threshold_auc": 0.28, "slope": -0.08},
    }
    champion_stats = {"mean": {"eval_v2_score": 0.33}}
    baseline_stats = {"mean": {"retention_auc": 0.30, "selectivity_score": 0.20}}

    decision = decide_eval_v2(candidate_stats, champion_stats, baseline_stats)

    assert decision["accept"] is True


def test_candidate_rejected_when_fold_consistency_is_too_weak():
    candidate_stats = {
        "mean": {"eval_v2_score": 0.34, "retention_auc": 0.38, "selectivity_score": 0.28},
        "fold_deltas": [0.02, -0.01, 0.005, 0.0, 0.01],
        "lower_bound_delta": -0.001,
        "worst_fold": {"retention_auc": 0.25, "selectivity_score": 0.17},
        "threshold_summary": {"threshold_auc": 0.0, "slope": 0.0},
    }
    champion_stats = {"mean": {"eval_v2_score": 0.33}}
    baseline_stats = {"mean": {"retention_auc": 0.30, "selectivity_score": 0.20}}

    decision = decide_eval_v2(candidate_stats, champion_stats, baseline_stats)

    assert decision["accept"] is False
    assert "fold" in decision["reason"].lower() or "threshold" in decision["reason"].lower()
