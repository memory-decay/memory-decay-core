"""Continuous activation separation: low floor, slow decay, max ranking signal."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    # Low floor: let most memories decay significantly
    # Only truly important memories get meaningful floor protection
    floor_base = params.get("floor_base", 0.05)
    floor_importance_scale = params.get("floor_importance_scale", 0.30)
    floor = floor_base + floor_importance_scale * importance
    floor = min(floor, activation)

    # Very slow base decay — let activation spread out gradually
    lambda_fact = params.get("lambda_fact", 0.005)
    lambda_episode = params.get("lambda_episode", 0.020)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    # Retention
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    # Jost decay with high power for sharp separation
    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
