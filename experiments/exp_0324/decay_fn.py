"""Hyperbolic decay with sigmoid floor: a(t+1) = floor + excess/(1+rate)."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    # Sigmoid floor
    floor_max = params.get("floor_max", 0.60)
    sigmoid_k = params.get("sigmoid_k", 20.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.25)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    floor = floor_max * sigmoid
    floor = min(floor, activation)

    # Hyperbolic decay: excess/(1+rate) — natural fat tail
    base_rate = params.get("base_rate", 0.015)
    rate_episode_mult = params.get("rate_episode_mult", 2.5)
    rate = base_rate if mtype == "fact" else base_rate * rate_episode_mult

    # Retention from importance
    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_rate = rate / retention

    # Hyperbolic: excess decays as 1/(1+r) per step
    # This has a natural fat tail — never reaches 0
    excess = max(activation - floor, 0.0)
    new_excess = excess / (1.0 + effective_rate)

    # Apply excess-dependent acceleration like Jost
    # High excess decays proportionally faster
    excess_ratio = excess  # 0 to ~1
    acceleration = 1.0 + params.get("accel_factor", 3.0) * excess_ratio
    new_excess = excess / (1.0 + effective_rate * acceleration)

    new_activation = floor + new_excess

    return max(0.0, min(new_activation, activation))
