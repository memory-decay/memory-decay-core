"""Bi-exponential decay: fast hippocampal + slow neocortical components."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    # Floor
    floor_max = params.get("floor_max", 0.55)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)
    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)
    floor = min(floor_max * sigmoid, activation)

    # Bi-exponential: two decay rates
    # Fast component (hippocampal-like): decays quickly
    # Slow component (neocortical-like): decays slowly, weighted by importance
    lambda_fast = params.get("lambda_fast", 0.06)
    lambda_slow = params.get("lambda_slow", 0.008)

    # Type factor
    type_factor = params.get("type_fact_factor", 0.7) if mtype == "fact" else 1.0

    # Mix: high importance → more slow component
    slow_weight = min(importance * 1.5, 1.0)
    fast_weight = 1.0 - slow_weight

    # Retention
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)

    effective_fast = lambda_fast * type_factor / retention
    effective_slow = lambda_slow * type_factor / retention

    excess = max(activation - floor, 0.0)

    # Combined decay: weighted sum of two exponential decays
    decay_fast = effective_fast * excess
    decay_slow = effective_slow * excess
    decay_amount = fast_weight * decay_fast + slow_weight * decay_slow

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
