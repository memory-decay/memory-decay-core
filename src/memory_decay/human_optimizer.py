"""Random-search optimizer for human review calibration."""

from __future__ import annotations

import random

from .human_eval import HumanCalibrationEvaluator


DEFAULT_PARAMS = {
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
}


def _sample_fact_params(rng: random.Random) -> dict:
    params = dict(DEFAULT_PARAMS)
    params["lambda_fact"] = rng.uniform(0.005, 0.08)
    params["beta_fact"] = rng.uniform(0.02, 0.3)
    params["alpha"] = rng.uniform(0.0, 1.5)
    params["stability_weight"] = rng.uniform(0.0, 2.0)
    params["stability_decay"] = rng.uniform(0.0, 0.1)
    params["reinforcement_gain_direct"] = rng.uniform(0.05, 0.5)
    params["stability_cap"] = rng.uniform(0.5, 2.0)
    return params


def random_search_human_params(
    *,
    train_events: list[dict],
    valid_events: list[dict],
    iterations: int = 25,
    seed: int = 42,
) -> dict:
    """Search fact-side parameters with a simple random search."""
    rng = random.Random(seed)
    trials: list[dict] = []

    if iterations <= 0:
        baseline = dict(DEFAULT_PARAMS)
        evaluator = HumanCalibrationEvaluator(
            baseline,
            {"activation_scale": 6.0, "bias": -3.0, "stability_scale": 0.0},
        )
        for event in train_events:
            evaluator.replay_event(event)
        best_metrics = evaluator.evaluate(valid_events)
        return {
            "best_params": baseline,
            "best_metrics": best_metrics,
            "trials": trials,
        }

    best_params = None
    best_metrics = None
    best_score = float("inf")

    for _ in range(iterations):
        params = _sample_fact_params(rng)
        observation_params = {
            "activation_scale": 6.0,
            "bias": -3.0,
            "stability_scale": 0.0,
        }
        evaluator = HumanCalibrationEvaluator(params, observation_params)

        for event in train_events:
            evaluator.replay_event(event)
        metrics = evaluator.evaluate(valid_events)
        score = float(metrics["nll"])

        trials.append({"params": params, "metrics": metrics})
        if score < best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "trials": trials,
    }
