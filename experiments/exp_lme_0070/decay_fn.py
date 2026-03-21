"""Jost's Law with logarithmic floor instead of sigmoid floor.

Instead of sigmoid(importance) * floor_max, use a logarithmic floor:
  floor = floor_max * (1 + log(1 + importance)) / (1 + log(2))

This gives a more gradual, linear-ish floor response compared to the sharp sigmoid.

Zero-impact items bypass the floor and use pure exponential decay (lambda * 3.0).
"""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # --- Zero-impact bypass: pure exponential, no floor ---
    if impact <= 0:
        lambda_fact = params.get("lambda_fact", 0.009)
        lambda_episode = params.get("lambda_episode", 0.040)
        base_lambda = lambda_fact if mtype == "fact" else lambda_episode
        effective_lambda = base_lambda * 3.0
        new_activation = activation * math.exp(-effective_lambda)
        return max(0.0, new_activation)

    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.45)

    # Logarithmic floor: more gradual than sigmoid
    # floor = floor_max * log(1 + importance) / log(2)
    # importance=0 → floor=0, importance=1 → floor=floor_max
    floor = floor_max * math.log1p(importance) / math.log(2)
    floor = min(floor, activation)

    lambda_fact = params.get("lambda_fact", 0.009)
    lambda_episode = params.get("lambda_episode", 0.040)
    base_lambda = lambda_fact if mtype == "fact" else lambda_episode

    retention = max(1.0 + alpha * impact + rho * stability, 1.0)
    effective_lambda = base_lambda / retention

    excess = max(activation - floor, 0.0)
    jost_power = params.get("jost_power", 4.0)
    decay_amount = effective_lambda * (excess ** jost_power)

    new_activation = activation - decay_amount
    new_activation = max(new_activation, floor)

    return max(0.0, min(new_activation, activation))
