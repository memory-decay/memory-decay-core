"""Tests for human review calibration evaluation."""

from memory_decay.human_eval import HumanCalibrationEvaluator, sigmoid_probability


def _make_evaluator() -> HumanCalibrationEvaluator:
    return HumanCalibrationEvaluator(
        decay_params={
            "lambda_fact": 0.02,
            "lambda_episode": 0.035,
            "beta_fact": 0.08,
            "beta_episode": 0.12,
            "alpha": 0.5,
            "stability_weight": 0.8,
            "stability_decay": 0.01,
            "reinforcement_gain_direct": 0.2,
            "reinforcement_gain_assoc": 0.05,
            "stability_cap": 1.0,
        },
        observation_params={
            "activation_scale": 6.0,
            "bias": -3.0,
            "stability_scale": 0.0,
        },
    )


def test_sigmoid_probability_is_bounded():
    assert 0.0 < sigmoid_probability(-100) < 0.01
    assert 0.99 < sigmoid_probability(100) < 1.0


def test_replay_event_returns_probability_and_updates_state():
    evaluator = _make_evaluator()
    event = {
        "user_id": "u1",
        "item_id": "i1",
        "memory_type": "fact",
        "t_elapsed": 3.0,
        "review_index": 1,
        "outcome": 1,
        "grade": None,
        "metadata": {},
    }

    result = evaluator.replay_event(event)

    assert 0.0 <= result["predicted_probability"] <= 1.0
    assert result["activation_before_review"] < 1.0
    state = evaluator.get_state("u1", "i1")
    assert state["stability"] > 0.0
    assert state["activation"] == 1.0


def test_metrics_include_nll_brier_and_ece():
    evaluator = _make_evaluator()
    events = [
        {
            "user_id": "u1",
            "item_id": "i1",
            "memory_type": "fact",
            "t_elapsed": 1.0,
            "review_index": 1,
            "outcome": 1,
            "grade": None,
            "metadata": {},
        },
        {
            "user_id": "u1",
            "item_id": "i1",
            "memory_type": "fact",
            "t_elapsed": 12.0,
            "review_index": 2,
            "outcome": 0,
            "grade": None,
            "metadata": {},
        },
    ]

    metrics = evaluator.evaluate(events)

    assert set(metrics) >= {"nll", "brier", "ece", "num_events"}
    assert metrics["num_events"] == 2
    assert metrics["nll"] >= 0.0
    assert metrics["brier"] >= 0.0
    assert 0.0 <= metrics["ece"] <= 1.0
