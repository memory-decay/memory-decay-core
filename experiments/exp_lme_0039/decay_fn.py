"""Jost's Law decay with additive stability floor boost."""

import math


def compute_decay(activation, impact, stability, mtype, params):
    if activation <= 0:
        return 0.0

    # --- Floor: sigmoid of importance PLUS stability boost ---
    alpha = params.get("alpha", 1.5)
    rho = params.get("stability_weight", 1.0)
    importance = (impact * alpha + stability * rho) / (alpha + rho)

    floor_max = params.get("floor_max", 0.43)
    sigmoid_k = params.get("sigmoid_k", 35.0)
    sigmoid_mid = params.get("sigmoid_mid", 0.30)

    z = sigmoid_k * (importance - sigmoid_mid)
    if z >= 0:
        sigmoid = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        sigmoid = ez / (1.0 + ez)

    base_floor = floor_max * sigmoid

    # Additive stability boost instead of multiplicative
    stability_boost = params.get("stability_boost", 0.15) * stability

    floor = base_floor + stability_boost
    floor = min(floor, activation)

    # --- Jost decay ---
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
